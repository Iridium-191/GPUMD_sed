#!/usr/bin/env python3
"""Visualize the complete FFT reciprocal cell from compute_SED output."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


HEADER_FORMAT = "<7i8fi15i"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
SED_MAGIC = 0x31444553


def read_sed_bin(filename: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    """Return the compute_SED header and data shaped [segment, time, component, y, x]."""
    path = Path(filename)
    with path.open("rb") as stream:
        values = struct.unpack(HEADER_FORMAT, stream.read(HEADER_SIZE))
        header = {
            "magic": values[0],
            "version": values[1],
            "sed_length": values[2],
            "num_segments": values[3],
            "nx": values[4],
            "ny": values[5],
            "sample_interval": values[6],
            "dt_sample_ps": values[7],
            "max_frequency": values[8],
            "nyquist_frequency": values[9],
            "cell_x": values[10],
            "cell_y": values[11],
            "dx": values[12],
            "cx": values[13],
            "cy": values[14],
            "group_id": values[15],
        }
        if header["magic"] != SED_MAGIC:
            raise ValueError(f"{path} is not a compute_SED file (unexpected magic number).")

        shape = (header["num_segments"], header["sed_length"], 3, header["ny"], header["nx"])
        expected = int(np.prod(shape))
        data = np.fromfile(stream, dtype="<f4", count=expected)

    if data.size != expected:
        raise ValueError(f"{path} is truncated: expected {expected} float32 values, got {data.size}.")
    return header, data.reshape(shape)


def projected_spectrum(data: np.ndarray, header: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return f, qx, qy and Phi(f, qy, qx) over the complete FFT reciprocal cell."""
    if data.ndim != 5 or data.shape[2] != 3:
        raise ValueError("data must have shape [segment, time, 3, y, x].")

    nseg, length, ncomp, ny, nx = data.shape
    if nseg == 0:
        raise ValueError("sed.bin contains no complete SED segment.")

    velocity = data.astype(np.float64, copy=False)
    velocity = velocity - velocity.mean(axis=1, keepdims=True)
    amplitude = np.fft.fftn(velocity, axes=(1, 3, 4))
    power = np.abs(amplitude) ** 2
    power = power.sum(axis=(0, 2)) / (nseg * ncomp * length * ny * nx)

    nfreq = length // 2 + 1
    dt_sample_ps = float(header["dt_sample_ps"])
    dx = float(header["dx"])
    frequency_thz = np.arange(nfreq) / (length * dt_sample_ps)
    qx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    qy = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    power = np.fft.fftshift(power[:nfreq], axes=(1, 2))
    return frequency_thz, qx, qy, power


def close_periodic_axis(axis: np.ndarray, values: np.ndarray, dimension: int, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Append the first periodic data plane at the high-q boundary for plotting."""
    closed_axis = np.append(axis, axis[0] + period)
    first_plane = np.take(values, [0], axis=dimension)
    return closed_axis, np.concatenate((values, first_plane), axis=dimension)


def add_si_high_symmetry_ticks(axis: plt.Axes, qmin: float, qmax: float, lattice_constant: float, direction: str) -> None:
    """Label repeated Gamma and X points along a cubic Si [100]/[010] reciprocal axis."""
    x_point = 2.0 * np.pi / lattice_constant
    indices = np.arange(np.ceil(qmin / x_point - 1.0e-3), np.floor(qmax / x_point + 1.0e-3) + 1, dtype=int)
    positions = indices * x_point
    labels = [r"$\Gamma$" if index % 2 == 0 else r"$X$" for index in indices]
    if direction == "x":
        axis.set_xticks(positions, labels)
    else:
        axis.set_yticks(positions, labels)


def full_qx_dispersion(
    frequency_thz: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full periodic qx dispersion at qy=0, including both X boundaries."""
    iy0 = int(np.argmin(np.abs(qy)))
    qx_period = (qx[1] - qx[0]) * len(qx)
    qx_closed, intensity = close_periodic_axis(qx, power[:, iy0, :], 1, qx_period)
    return qx_closed, intensity


def plot_full_qx_dispersion(
    frequency_thz: np.ndarray,
    qx: np.ndarray,
    intensity: np.ndarray,
    output: str | Path,
    fmax: float | None,
    lattice_constant: float | None,
) -> None:
    """Save the entire qx periodic cell, labelled X-Gamma-X."""
    keep = np.ones_like(frequency_thz, dtype=bool) if fmax is None else frequency_thz <= fmax
    if not np.any(keep):
        raise ValueError("fmax is below the first non-negative frequency.")

    fig, axis = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    image = axis.pcolormesh(
        qx,
        frequency_thz[keep],
        np.log10(intensity[keep] + np.finfo(float).eps),
        shading="auto",
        cmap="magma",
    )
    axis.set_xlabel(r"$q_x$ at $q_y=q_z=0$ (rad/A)")
    axis.set_ylabel("Frequency (THz)")
    q_margin = 0.015 * (qx[-1] - qx[0])
    axis.set_xlim(qx[0] - q_margin, qx[-1] + q_margin)
    axis.set_ylim(0.0, frequency_thz[keep][-1])
    if lattice_constant is None:
        axis.set_xticks((qx[0], 0.0, qx[-1]), (f"{qx[0]:.2f}", r"$\Gamma$", f"{qx[-1]:.2f}"))
    else:
        add_si_high_symmetry_ticks(axis, qx[0], qx[-1], lattice_constant, "x")
    axis.axvline(0.0, color="white", linewidth=0.7, alpha=0.55)
    fig.colorbar(image, ax=axis, label=r"$\log_{10}\Phi$")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_qspace_slices(
    frequency_thz: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    power: np.ndarray,
    requested_frequencies: Sequence[float],
    output: str | Path,
    lattice_constant: float | None,
) -> None:
    """Save full qx-qy maps at the requested frequencies, with periodic boundaries closed."""
    if not requested_frequencies:
        raise ValueError("at least one slice frequency is required.")

    qx_period = (qx[1] - qx[0]) * len(qx)
    qy_period = (qy[1] - qy[0]) * len(qy)
    qx_closed, power_closed_x = close_periodic_axis(qx, power, 2, qx_period)
    qy_closed, power_closed = close_periodic_axis(qy, power_closed_x, 1, qy_period)

    indices = [int(np.argmin(np.abs(frequency_thz - target))) for target in requested_frequencies]
    log_slices = [np.log10(power_closed[index] + np.finfo(float).eps) for index in indices]
    limits = np.concatenate([image.ravel() for image in log_slices])
    vmin, vmax = np.percentile(limits, (2.0, 99.5))

    ncols = min(2, len(indices))
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows), constrained_layout=True)
    axes_flat = np.asarray(axes).ravel()
    image = None
    for axis, index, log_slice in zip(axes_flat, indices, log_slices):
        image = axis.pcolormesh(qx_closed, qy_closed, log_slice, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        axis.set_title(f"{frequency_thz[index]:.3f} THz")
        axis.set_xlabel(r"$q_x$ (rad/A)")
        axis.set_ylabel(r"$q_y$ (rad/A)")
        if lattice_constant is not None:
            add_si_high_symmetry_ticks(axis, qx_closed[0], qx_closed[-1], lattice_constant, "x")
            add_si_high_symmetry_ticks(axis, qy_closed[0], qy_closed[-1], lattice_constant, "y")
        axis.axhline(0.0, color="white", linewidth=0.6, alpha=0.5)
        axis.axvline(0.0, color="white", linewidth=0.6, alpha=0.5)
        qx_margin = 0.015 * (qx_closed[-1] - qx_closed[0])
        qy_margin = 0.015 * (qy_closed[-1] - qy_closed[0])
        axis.set_xlim(qx_closed[0] - qx_margin, qx_closed[-1] + qx_margin)
        axis.set_ylim(qy_closed[0] - qy_margin, qy_closed[-1] + qy_margin)
        axis.set_aspect("equal")
    for axis in axes_flat[len(indices):]:
        axis.remove()
    fig.colorbar(image, ax=axes_flat[: len(indices)], label=r"$\log_{10}\Phi$")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot complete reciprocal-space FFT maps from compute_SED sed.bin.")
    parser.add_argument("sed_bin", type=Path, help="sed.bin produced by compute_SED")
    parser.add_argument("--output", type=Path, default=Path("full_qx_dispersion.png"), help="full qx dispersion PNG")
    parser.add_argument("--slice-output", type=Path, default=Path("full_q_slices.png"), help="full qx-qy slice PNG")
    parser.add_argument("--slices", type=float, nargs="+", default=(1.0, 5.0, 10.0, 15.0), help="requested slice frequencies in THz")
    parser.add_argument("--fmax", type=float, default=None, help="maximum displayed frequency in the qx dispersion")
    parser.add_argument("--lattice-constant", type=float, default=5.431, help="cubic lattice constant in A for Gamma/X labels; use 0 to disable labels")
    args = parser.parse_args()

    header, data = read_sed_bin(args.sed_bin)
    frequency_thz, qx, qy, power = projected_spectrum(data, header)
    qx_closed, intensity = full_qx_dispersion(frequency_thz, qx, qy, power)
    lattice_constant = args.lattice_constant if args.lattice_constant > 0.0 else None
    plot_full_qx_dispersion(frequency_thz, qx_closed, intensity, args.output, args.fmax, lattice_constant)
    plot_qspace_slices(frequency_thz, qx, qy, power, args.slices, args.slice_output, lattice_constant)
    print(f"Wrote {args.output} and {args.slice_output} from {header['num_segments']} segment(s), grid {header['nx']} x {header['ny']}.")


if __name__ == "__main__":
    main()
