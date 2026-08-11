// Copyright (c) 2026 RuilinMao, ICQM, Peking University
// This file is part of GPUMD and is distributed under GPL-3.0-or-later.

#include "sed.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <climits>
#include <cstdlib>

namespace
{
constexpr int BLOCK_SIZE = 128;

inline bool sed_is_valid_int(const char* s, int* out)
{
  if (s == nullptr || out == nullptr || s[0] == '\0') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') {
    return false;
  }
  if (v < INT_MIN || v > INT_MAX) {
    return false;
  }
  *out = static_cast<int>(v);
  return true;
}

inline bool sed_is_valid_real(const char* s, double* out)
{
  if (s == nullptr || out == nullptr || s[0] == '\0') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (errno != 0 || end == s || *end != '\0') {
    return false;
  }
  *out = v;
  return true;
}

__global__ void gpu_build_group_atom_index(
  const int num_atoms,
  const int offset,
  const int* g_group_contents,
  int* g_atom_index)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < num_atoms) {
    g_atom_index[n] = g_group_contents[offset + n];
  }
}

__global__ void gpu_copy_mass_group(
  const int num_atoms,
  const int* g_atom_index,
  const double* g_mass_in,
  float* g_mass_out)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < num_atoms) {
    const int m = g_atom_index[n];
    g_mass_out[n] = static_cast<float>(g_mass_in[m]);
  }
}

__global__ void gpu_build_bin_index(
  const int num_atoms,
  const int* g_atom_index,
  const double* g_x,
  const double* g_y,
  const float xmin,
  const float ymin,
  const float dx,
  const int nx,
  const int ny,
  int* g_bin_index)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < num_atoms) {
    const int m = g_atom_index[n];
    const float x = static_cast<float>(g_x[m]);
    const float y = static_cast<float>(g_y[m]);

    const int ix = static_cast<int>(floorf((x - xmin) / dx));
    const int iy = static_cast<int>(floorf((y - ymin) / dx));

    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
      g_bin_index[n] = iy * nx + ix;
    } else {
      g_bin_index[n] = -1;
    }
  }
}

__global__ void gpu_zero_array(const int n, float* data)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = 0.0f;
  }
}

__global__ void gpu_project_velocity_to_grid(
  const int num_atoms,
  const int* g_atom_index,
  const int* g_bin_index,
  const float* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const int num_pixels,
  float* g_frame)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < num_atoms) {
    const int bin = g_bin_index[n];
    if (bin >= 0 && bin < num_pixels) {
      const int m = g_atom_index[n];
           const float mass = g_mass[n];
      const float mass_sqrt = sqrtf(mass);

      const float vx = static_cast<float>(g_vx[m]);
      const float vy = static_cast<float>(g_vy[m]);
      const float vz = static_cast<float>(g_vz[m]);

      // frame layout: [3][num_pixels]
      atomicAdd(&g_frame[bin], mass_sqrt * vx);
      atomicAdd(&g_frame[num_pixels + bin], mass_sqrt * vy);
      atomicAdd(&g_frame[2 * num_pixels + bin], mass_sqrt * vz);
    }
  }
}

} // namespace

SED::SED(const char** param, int num_param, const std::vector<Group>& groups)
{
  property_name = "compute_SED";
  parse(param, num_param, groups);
}

void SED::parse(const char** param, int num_param, const std::vector<Group>& groups)
{
  printf("Compute projected SED grid (reference-position binning + mass-weighted velocity).\n");
  compute_ = true;

  if (num_param < 6) {
    PRINT_INPUT_ERROR("compute_SED requires at least 5 parameters.\n");
  }

  if (!sed_is_valid_int(param[1], &sed_length_)) {
    PRINT_INPUT_ERROR("SED_length should be an integer.\n");
  }
  if (sed_length_ <= 0) {
    PRINT_INPUT_ERROR("SED_length should be positive.\n");
  }

  if (!sed_is_valid_real(param[2], &max_frequency_)) {
    PRINT_INPUT_ERROR("max_frequency should be a real number.\n");
  }
  if (max_frequency_ <= 0.0) {
    PRINT_INPUT_ERROR("max_frequency should be positive.\n");
  }

  if (!sed_is_valid_real(param[3], &cell_x_)) {
    PRINT_INPUT_ERROR("cell_x should be a real number.\n");
  }
  if (!sed_is_valid_real(param[4], &cell_y_)) {
    PRINT_INPUT_ERROR("cell_y should be a real number.\n");
  }
  if (!sed_is_valid_real(param[5], &dx_)) {
    PRINT_INPUT_ERROR("dx should be a real number.\n");
  }
  if (cell_x_ <= 0.0 || cell_y_ <= 0.0 || dx_ <= 0.0) {
    PRINT_INPUT_ERROR("cell_x, cell_y, dx should all be positive.\n");
  }

  use_group_ = false;
  grouping_method_ = -1;
  group_id_ = -1;
  group_ = nullptr;

  for (int k = 6; k < num_param; ++k) {
    if (strcmp(param[k], "group") == 0) {
      use_group_ = true;

      ++k;
      if (k >= num_param) {
        PRINT_INPUT_ERROR("Missing grouping_method after 'group'.\n");
      }
      if (!sed_is_valid_int(param[k], &grouping_method_)) {
        PRINT_INPUT_ERROR("grouping_method should be an integer.\n");
      }

      ++k;
      if (k >= num_param) {
        PRINT_INPUT_ERROR("Missing group_id after grouping_method.\n");
      }
      if (!sed_is_valid_int(param[k], &group_id_)) {
        PRINT_INPUT_ERROR("group_id should be an integer.\n");
      }
    } else {
      PRINT_INPUT_ERROR("Unknown argument in compute_SED.\n");
    }
  }

  if (use_group_) {
    if (grouping_method_ < 0 || grouping_method_ >= static_cast<int>(groups.size())) {
      PRINT_INPUT_ERROR("Invalid grouping_method in compute_SED.\n");
    }
    group_ = &groups[grouping_method_];
    if (group_id_ < 0 || group_id_ >= group_->number) {
      PRINT_INPUT_ERROR("Invalid group_id in compute_SED.\n");
    }
  }
}

void SED::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  (void)number_of_steps;
  (void)integrate;
  (void)box;
  (void)force;

  if (!compute_) {
    return;
  }

  initialize_parameters(time_step, group, atom);
  allocate_memory();
  build_bin_index(atom);
  copy_mass(atom);
  open_output_file();
}

void SED::initialize_parameters(
  const double time_step,
  const std::vector<Group>& groups,
  Atom& atom)
{
  // GPUMD time_step is interpreted here in fs.
  // Convert fs -> ps directly, and interpret max_frequency_ as ordinary frequency in THz.
  // IMPORTANT:
// In run.cu, time_step has already been divided by TIME_UNIT_CONVERSION
// before Property::preprocess() is called.
// Therefore we must multiply TIME_UNIT_CONVERSION back here, then convert to ps.
// max_frequency_ is interpreted as ordinary frequency in THz (1/ps), not angular frequency.
dt_md_ps_ = time_step * TIME_UNIT_CONVERSION / 1000.0;

sample_interval_ = static_cast<int>(floor(1.0 / (2.0 * max_frequency_ * dt_md_ps_)));
if (sample_interval_ < 1) {
  sample_interval_ = 1;
}

dt_sample_ps_ = dt_md_ps_ * sample_interval_;
nyquist_frequency_ = 1.0 / (2.0 * dt_sample_ps_);

// Guard against integer floor placing Nyquist slightly below the requested max frequency
while (sample_interval_ > 1 && nyquist_frequency_ < max_frequency_) {
  --sample_interval_;
  dt_sample_ps_ = dt_md_ps_ * sample_interval_;
  nyquist_frequency_ = 1.0 / (2.0 * dt_sample_ps_);
}

num_atoms_total_ = atom.number_of_atoms;

if (!use_group_) {
  num_atoms_ = num_atoms_total_;
  group_ = nullptr;
} else {
  if (grouping_method_ < 0 || grouping_method_ >= static_cast<int>(groups.size())) {
    PRINT_INPUT_ERROR("Invalid grouping_method in compute_SED.\n");
  }
  group_ = &groups[grouping_method_];
  if (group_id_ < 0 || group_id_ >= group_->number) {
    PRINT_INPUT_ERROR("Invalid group_id in compute_SED.\n");
  }
  num_atoms_ = group_->cpu_size[group_id_];
}
  nx_ = static_cast<int>(floor(cell_x_ / dx_)) + 1;
  ny_ = static_cast<int>(floor(cell_y_ / dx_)) + 1;
  if (nx_ <= 0 || ny_ <= 0) {
    PRINT_INPUT_ERROR("Invalid nx or ny derived from cell_x/cell_y/dx.\n");
  }
  num_pixels_ = nx_ * ny_;

  current_local_step_ = 0;
  num_segments_written_ = 0;
  num_sampled_frames_ = 0;

  header_.sed_length = sed_length_;
  header_.num_segments = 0;
  header_.nx = nx_;
  header_.ny = ny_;
  header_.sample_interval = sample_interval_;
  header_.dt_sample_ps = static_cast<float>(dt_sample_ps_);
  header_.max_frequency = static_cast<float>(max_frequency_);
  header_.nyquist_frequency = static_cast<float>(nyquist_frequency_);
  header_.cell_x = static_cast<float>(cell_x_);
  header_.cell_y = static_cast<float>(cell_y_);
  header_.dx = static_cast<float>(dx_);
  header_.group_id = group_id_;
}

void SED::allocate_memory()
{
  atom_index_.resize(num_atoms_);
  bin_index_.resize(num_atoms_);
  mass_.resize(num_atoms_);

  frame_buffer_.resize(3 * num_pixels_);
  host_frame_buffer_.resize(static_cast<size_t>(3) * num_pixels_, 0.0f);
  host_segment_buffer_.resize(static_cast<size_t>(sed_length_) * 3 * num_pixels_, 0.0f);
}

void SED::build_bin_index(Atom& atom)
{
  if (!use_group_) {
    std::vector<int> h_atom_index(num_atoms_);
    for (int i = 0; i < num_atoms_; ++i) {
      h_atom_index[i] = i;
    }
    atom_index_.copy_from_host(h_atom_index.data());
  } else {
    const int offset = group_->cpu_size_sum[group_id_];
    gpu_build_group_atom_index<<<(num_atoms_ - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
      num_atoms_, offset, group_->contents.data(), atom_index_.data());
    GPU_CHECK_KERNEL
  }

  const int N = atom.number_of_atoms;
  const double* x_cpu = atom.cpu_position_per_atom.data();
  const double* y_cpu = atom.cpu_position_per_atom.data() + N;

  std::vector<int> h_atom_index(num_atoms_);
  atom_index_.copy_to_host(h_atom_index.data());

  double sx = 0.0;
  double sy = 0.0;
  for (int i = 0; i < num_atoms_; ++i) {
    const int m = h_atom_index[i];
    sx += x_cpu[m];
    sy += y_cpu[m];
  }
  cx_ = static_cast<float>(sx / num_atoms_);
  cy_ = static_cast<float>(sy / num_atoms_);

  xmin_ = cx_ - static_cast<float>(0.5 * cell_x_);
  ymin_ = cy_ - static_cast<float>(0.5 * cell_y_);

  header_.cx = cx_;
  header_.cy = cy_;

  const double* x_gpu = atom.position_per_atom.data();
  const double* y_gpu = atom.position_per_atom.data() + N;

  gpu_build_bin_index<<<(num_atoms_ - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    num_atoms_,
    atom_index_.data(),
    x_gpu,
    y_gpu,
    xmin_,
    ymin_,
    static_cast<float>(dx_),
    nx_,
    ny_,
    bin_index_.data());
  GPU_CHECK_KERNEL
}

void SED::copy_mass(Atom& atom)
{
  gpu_copy_mass_group<<<(num_atoms_ - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    num_atoms_, atom_index_.data(), atom.mass.data(), mass_.data());
  GPU_CHECK_KERNEL
}

void SED::open_output_file()
{
  fid_ = fopen(output_filename_.c_str(), "wb");
  if (fid_ == nullptr) {
    PRINT_INPUT_ERROR("Failed to open sed.bin for writing.\n");
  }
  write_header_placeholder();
}

void SED::write_header_placeholder()
{
  fwrite(&header_, sizeof(Header), 1, fid_);
}

void SED::rewrite_header()
{
  if (fid_ == nullptr) {
    return;
  }
  header_.num_segments = num_segments_written_;
  fseek(fid_, 0, SEEK_SET);
  fwrite(&header_, sizeof(Header), 1, fid_);
  fseek(fid_, 0, SEEK_END);
}

void SED::close_output_file()
{
  if (fid_ != nullptr) {
    rewrite_header();
    fclose(fid_);
    fid_ = nullptr;
  }
}

void SED::project_frame(Atom& atom)
{
  gpu_zero_array<<<(3 * num_pixels_ - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    3 * num_pixels_, frame_buffer_.data());
  GPU_CHECK_KERNEL

  const int N = atom.number_of_atoms;
  const double* vx = atom.velocity_per_atom.data();
  const double* vy = atom.velocity_per_atom.data() + N;
  const double* vz = atom.velocity_per_atom.data() + 2 * N;

  gpu_project_velocity_to_grid<<<(num_atoms_ - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    num_atoms_,
    atom_index_.data(),
    bin_index_.data(),
    mass_.data(),
    vx,
    vy,
    vz,
    num_pixels_,
    frame_buffer_.data());
  GPU_CHECK_KERNEL
}

void SED::append_current_frame_to_segment()
{
  frame_buffer_.copy_to_host(host_frame_buffer_.data());

  const size_t frame_size = static_cast<size_t>(3) * num_pixels_;
  const size_t offset = static_cast<size_t>(current_local_step_) * frame_size;
  std::memcpy(
    host_segment_buffer_.data() + offset,
    host_frame_buffer_.data(),
    frame_size * sizeof(float));
}

void SED::flush_segment_to_file()
{
  if (fid_ == nullptr) {
    PRINT_INPUT_ERROR("sed.bin file handle is null.\n");
  }

  const size_t count = static_cast<size_t>(sed_length_) * 3 * num_pixels_;
  fwrite(host_segment_buffer_.data(), sizeof(float), count, fid_);
  ++num_segments_written_;
  current_local_step_ = 0;

  std::fill(host_segment_buffer_.begin(), host_segment_buffer_.end(), 0.0f);
}

void SED::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  (void)number_of_steps;
  (void)fixed_group;
  (void)move_group;
  (void)global_time;
  (void)temperature;
  (void)integrate;
  (void)box;
  (void)group;
  (void)thermo;
  (void)force;

  if (!compute_) {
    return;
  }
  if ((step + 1) % sample_interval_ != 0) {
    return;
  }

  project_frame(atom);
  append_current_frame_to_segment();
  ++current_local_step_;
  ++num_sampled_frames_;

  if (current_local_step_ == sed_length_) {
    flush_segment_to_file();
  }
}

void SED::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  (void)atom;
  (void)box;
  (void)integrate;
  (void)number_of_steps;
  (void)time_step;
  (void)temperature;

  if (!compute_) {
    return;
  }

  CHECK(gpuDeviceSynchronize());
  close_output_file();

  printf("SED finished.\n");
  printf("    sample interval   = %d\n", sample_interval_);
  printf("    dt_sample_ps      = %g\n", dt_sample_ps_);
  printf("    Nyquist frequency = %g THz\n", nyquist_frequency_);
  printf("    grid              = %d x %d\n", nx_, ny_);
  printf("    sampled frames    = %lld\n", num_sampled_frames_);
  printf("    written segments  = %d\n", num_segments_written_);
  if (current_local_step_ != 0) {
    printf("    discarded tail    = %d frames (incomplete segment)\n", current_local_step_);
  }

  compute_ = false;
}
