#!/usr/bin/env python3
"""Generate a periodic 11 x 11 x 11 conventional-cell diamond-Si crystal."""

from pathlib import Path


A0 = 5.431
REPEATS = 11
BASIS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)


def main() -> None:
    length = REPEATS * A0
    atoms = []
    for iz in range(REPEATS):
        for iy in range(REPEATS):
            for ix in range(REPEATS):
                for bx, by, bz in BASIS:
                    atoms.append(((ix + bx) * A0, (iy + by) * A0, (iz + bz) * A0))

    model = Path(__file__).with_name("model.xyz")
    with model.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{len(atoms)}\n")
        stream.write(
            'pbc="T T T" '
            f'Lattice="{length:.6f} 0 0 0 {length:.6f} 0 0 0 {length:.6f}" '
            'Properties=species:S:1:pos:R:3:mass:R:1\n'
        )
        for x, y, z in atoms:
            stream.write(f"Si {x:.6f} {y:.6f} {z:.6f} 28.0855\n")
    print(f"Wrote {model} with {len(atoms)} atoms and [001] along z.")


if __name__ == "__main__":
    main()
