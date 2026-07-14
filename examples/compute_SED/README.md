# `compute_SED`: Si[001] Full Reciprocal-Space Projected Spectrum Example

This directory documents the `compute_SED` implementation in `src/measure/sed.cu` and provides a reproducible 6 nm single-crystal Si[001] example. The CUDA code writes mass-weighted, grid-projected velocity fields to `sed.bin`; the Python code calculates and visualizes their reciprocal-space spectrum.

The current implementation is a projected grid spectrum with `q_z = 0`. It is not an atom-resolved, basis-preserving lattice SED, because velocities from atoms in the same `x-y` bin are summed before the FFT.

## 1. CUDA and post-processing formulas

Let the MD time step be `time_step = Delta t_MD` in fs and let `max_frequency = f_max` in THz (`1/ps`). `sed.cu` chooses an integer sampling interval

$$
s = \max\!\left(1,\left\lfloor\frac{1}{2f_{\max}\Delta t_{\mathrm{MD}}}\right\rfloor\right),
\qquad
\Delta t_{\mathrm{MD}}(\mathrm{ps})=\frac{\Delta t_{\mathrm{MD}}(\mathrm{fs})}{1000}.
$$

If integer rounding makes the Nyquist frequency too low, the code decreases $s$. The actual sampling interval, Nyquist frequency, and frequency resolution for a segment of length $L=\texttt{SED\_length}$ are

$$
\Delta t=s\Delta t_{\mathrm{MD}},\qquad
f_{\mathrm{Nyq}}=\frac{1}{2\Delta t},\qquad
\Delta f=\frac{1}{L\Delta t},\qquad
f_j=\frac{j}{L\Delta t}.
$$

At `preprocess()`, CUDA builds a fixed `x-y` grid from the reference coordinates $(x_i^0,y_i^0)$ of the selected atoms:

$$
c_x=\frac{1}{N}\sum_i x_i^0,\qquad c_y=\frac{1}{N}\sum_i y_i^0,
$$

$$
x_{\min}=c_x-\frac{\texttt{cell\_x}}{2},\qquad
y_{\min}=c_y-\frac{\texttt{cell\_y}}{2},
$$

$$
n_x=\left\lfloor\frac{\texttt{cell\_x}}{\texttt{dx}}\right\rfloor+1,\qquad
n_y=\left\lfloor\frac{\texttt{cell\_y}}{\texttt{dx}}\right\rfloor+1.
$$

Atom $i$ is assigned to $l_i=\lfloor(x_i^0-x_{\min})/\texttt{dx}\rfloor$ and $m_i=\lfloor(y_i^0-y_{\min})/\texttt{dx}\rfloor$. The CUDA kernel `gpu_project_velocity_to_grid()` writes the mass-weighted velocity projection

$$
u_\alpha(l,m,t_n)=
\sum_{i\in\mathcal G}\sqrt{M_i}\,v_{i\alpha}(t_n)
\,\mathbf 1(l_i=l)\,\mathbf 1(m_i=m),
\qquad \alpha\in\{x,y,z\}.
$$

Here $\mathcal G$ is either all atoms or the group selected by the optional `group` argument. The binary data layout is `[segment][time][component][iy][ix]` in `float32`.

The Python post-processing removes the time average of each segment and velocity component, then performs an FFT over time and the two spatial grid axes:

$$
A_{s,\alpha}(j,a,b)=
\sum_{n=0}^{L-1}\sum_{m=0}^{n_y-1}\sum_{l=0}^{n_x-1}
u_{s,\alpha}(l,m,t_n)
\exp\left[-2\pi i\left(\frac{jn}{L}+\frac{am}{n_y}+\frac{bl}{n_x}\right)\right].
$$

The normalized projected power spectrum is

$$
\Phi(j,a,b)=\frac{1}{N_s\,3\,L\,n_y\,n_x}
\sum_{s=0}^{N_s-1}\sum_{\alpha=x,y,z}
\left|A_{s,\alpha}(j,a,b)\right|^2,
$$

with reciprocal coordinates

$$
q_x(b)=\frac{2\pi b}{n_x\,\texttt{dx}},\qquad
q_y(a)=\frac{2\pi a}{n_y\,\texttt{dx}}.
$$

For diamond Si along `[100]`, $X=2\pi/a$ and the reciprocal lattice vector is $G=4\pi/a$. This example uses `dx` close to $a/4$, giving a `44 x 44` grid and an FFT range approximately equal to $[-G,G]$. The plotted high-symmetry sequence is $\Gamma-X-\Gamma-X-\Gamma$. The energy slices show the full `q_x-q_y` FFT cell at selected frequencies.

## 2. `compute_SED` usage

```text
compute_SED <SED_length> <max_frequency> <cell_x> <cell_y> <dx> [group <grouping_method> <group_id>]
```

| Parameter | Meaning |
| --- | --- |
| `SED_length` | Number of sampled frames per segment, $L$, which sets $\Delta f$. |
| `max_frequency` | Requested maximum ordinary frequency, $f_{\max}$, in THz; it is not an angular frequency. |
| `cell_x`, `cell_y` | Reference-coordinate projection-window dimensions in A. |
| `dx` | Shared `x` and `y` grid spacing in A. |
| `group <grouping_method> <group_id>` | Optional atom selection. `group_id` must be a valid non-negative ID. |

Use NVE for the production segment to avoid thermostat broadening. The grid is established at the beginning of the `run` containing `compute_SED`; if the system undergoes large diffusion or deformation during equilibration, make sure that the projection window still covers all target atoms. To avoid a discarded incomplete tail, choose the production length as

$$
N_{\mathrm{run}}=s\,L\,N_s,
$$

where $N_s$ is a positive integer. The `sed.bin` header stores `dt_sample_ps`, `nx`, `ny`, `dx`, and the number of complete segments, so the Python analysis does not need those values to be entered again.

## 3. 6 nm Si[001] full-reciprocal-space example

`si001_6nm/generate_si001_model.py` creates an unstrained diamond-Si supercell of `11 x 11 x 11` conventional cells with $a=5.431$ A. The box length is $11a=59.741$ A and the model contains $8\times11^3=10648$ atoms, with $z\parallel[001]$. The input file uses the repository [Si Tersoff 1989 potential](../../potentials/tersoff/Si_Tersoff_1989.txt).

```bash
cd examples/compute_SED/si001_6nm
python3 generate_si001_model.py
../../../src/gpumd
python3 ../sed_dispersion.py sed.bin --fmax 20 --output full_qx_dispersion_20THz.png --slice-output full_q_slices.png
```

In PowerShell, the final two commands can be issued through WSL as follows:

```powershell
wsl bash -lc 'cd /mnt/e/gpumd/c2f020de-0b0d-4f0e-a62b-4a8d5e121007/gpumd_sed/examples/compute_SED/si001_6nm && ../../../src/gpumd && python3 ../sed_dispersion.py sed.bin --fmax 20 --output full_qx_dispersion_20THz.png --slice-output full_q_slices.png'
```

The input equilibrates for 20 ps with Langevin NVT at 300 K, then uses NVE production:

```text
time_step   1
compute_SED 4096 100 59.742 59.742 1.35778
run         20480
```

Therefore $s=5$, $\Delta t=0.005$ ps, $f_{\mathrm{Nyq}}=100$ THz, and $\Delta f=0.048828125$ THz. The spatial grid is `44 x 44`, with $q_{\max}=\pi/\texttt{dx}=2.31373$ rad/A, approximately $G=4\pi/a$. The production run writes one complete segment and produces a `sed.bin` file of about 91 MiB.

The following plots were generated from an actual run in this directory. The first is the complete $\Gamma-X-\Gamma-X-\Gamma$ extended-zone dispersion from 0 to 20 THz. The second shows complete two-dimensional momentum slices at 0.977, 4.980, 10.010, and 14.990 THz.

![Si[001] full extended-zone projected spectrum](si001_6nm/full_qx_dispersion_20THz.png)

![Si[001] full reciprocal-space energy slices](si001_6nm/full_q_slices.png)

## 4. Python analysis and notebook workflow

`sed_dispersion.py` reads the binary file with NumPy and computes the complete projected $\Phi(q_x,q_y,f)$. It requires Python 3, NumPy, and Matplotlib:

```bash
python3 -m pip install numpy matplotlib
python3 ../sed_dispersion.py sed.bin --fmax 20 --output full_qx_dispersion_20THz.png --slice-output full_q_slices.png
```

`--output` saves the complete extended-zone dispersion and `--slice-output` saves the two-dimensional momentum slices. For Si, `--lattice-constant 5.431` labels repeated $\Gamma$ and $X$ points. Use `--slices` to select slice frequencies and `--fmax` to set the maximum displayed frequency:

```bash
python3 ../sed_dispersion.py sed.bin --fmax 20 --slices 2 6 10 15 --output full_qx_20THz.png --slice-output full_q_slices_custom.png
```

For an interactive workflow, start Jupyter Lab in `si001_6nm` and run [sed_workflow.ipynb](si001_6nm/sed_workflow.ipynb) from top to bottom. The notebook covers model generation, optional GPUMD execution, metadata inspection, the complete extended-zone dispersion, two-dimensional momentum slices, and direct access to one numerical slice. The committed notebook already contains outputs from a successful run.

`read_sed_bin()` returns data shaped `[segment, time, component, y, x]`. `projected_spectrum()` returns the frequency axis, full `q_x` and `q_y` axes, and an intensity tensor shaped `[frequency, qy, qx]` for integration into other analyses.
