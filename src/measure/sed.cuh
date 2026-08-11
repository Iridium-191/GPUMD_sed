// Copyright (c) 2026 RuilinMao, ICQM, Peking University
// This file is part of GPUMD and is distributed under GPL-3.0-or-later.

#pragma once

#include "measure/property.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <string>
#include <vector>

class SED : public Property
{
public:
  SED(const char** param, int num_param, const std::vector<Group>& groups);
  ~SED() override = default;

  void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void process(
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
    Force& force) override;

  void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  void parse(const char** param, int num_param, const std::vector<Group>& groups);
  void initialize_parameters(const double time_step, const std::vector<Group>& groups, Atom& atom);
  void allocate_memory();
  void build_bin_index(Atom& atom);
  void copy_mass(Atom& atom);

  void project_frame(Atom& atom);
  void append_current_frame_to_segment();
  void flush_segment_to_file();
  void open_output_file();
  void write_header_placeholder();
  void rewrite_header();
  void close_output_file();

private:
  bool compute_ = false;

  // input parameters
  int sed_length_ = 0;
  double max_frequency_ = 0.0; // THz
  double cell_x_ = 0.0;
  double cell_y_ = 0.0;
  double dx_ = 0.0;

  // optional group selection, matching dos.cu style:
  // group <grouping_method> <group_id>
  bool use_group_ = false;
  int grouping_method_ = -1;
  int group_id_ = -1;
  const Group* group_ = nullptr;

  // derived parameters
  int num_atoms_total_ = 0;
  int num_atoms_ = 0;
  int sample_interval_ = 1;
  double dt_md_ps_ = 0.0;
  double dt_sample_ps_ = 0.0;
  double nyquist_frequency_ = 0.0;

  float cx_ = 0.0f;
  float cy_ = 0.0f;
  float xmin_ = 0.0f;
  float ymin_ = 0.0f;
  int nx_ = 0;
  int ny_ = 0;
  int num_pixels_ = 0;

  // runtime counters
  long long num_sampled_frames_ = 0;
  int current_local_step_ = 0;
  int num_segments_written_ = 0;

  // group-local arrays on GPU
  GPU_Vector<int> atom_index_; // local atom -> global atom index
  GPU_Vector<int> bin_index_;  // local atom -> pixel index, -1 if outside window
  GPU_Vector<float> mass_;     // local atom mass in float

  // current projected frame on GPU: [3][num_pixels]
  GPU_Vector<float> frame_buffer_;

  // host staging buffers
  std::vector<float> host_frame_buffer_;   // [3][num_pixels]
  std::vector<float> host_segment_buffer_; // [sed_length][3][num_pixels]

  // binary output
  FILE* fid_ = nullptr;
  std::string output_filename_ = "sed.bin";

  struct Header
  {
    int magic = 0x31444553; // "SED1" in little-endian int
    int version = 1;
    int sed_length = 0;
    int num_segments = 0;
    int nx = 0;
    int ny = 0;
    int sample_interval = 0;
    float dt_sample_ps = 0.0f;
    float max_frequency = 0.0f;
    float nyquist_frequency = 0.0f;
    float cell_x = 0.0f;
    float cell_y = 0.0f;
    float dx = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    int group_id = -1;
    int reserved[15] = {0};
  } header_;
};
