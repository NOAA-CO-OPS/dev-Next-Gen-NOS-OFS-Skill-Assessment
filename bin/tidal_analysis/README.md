# Tidal analysis CLI tools

Command-line drivers for the harmonic-analysis package
(`src/ofs_skill/tidal_analysis/`). See the
[Harmonic Analysis wiki page](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/wiki/Harmonic-Analysis)
for the full module documentation.

## `run_harmonic_analysis.py`

Main pipeline driver: runs harmonic analysis on paired model/observation series,
builds the constituent comparison table, and (optionally) generates tidal
predictions and non-tidal residuals. Usage is documented on the wiki.

## `validate_fortran_ha.py`

One-to-one validation of the Python HA package against legacy NOS **Fortran** HA
output. Given a Fortran HA log (`W*.log`) and the exact `.obs` series it analysed,
it runs two checks and writes a statistics table, figures, and a markdown report:

- **Validation A — same-window real-data comparison.** Runs the Python HA on the
  identical observed series and compares the resolved constituents to the Fortran
  constants (amplitude, Greenwich-phase, and NOS vector difference). Because the
  input, record length, and epoch are identical, residuals isolate the analysis
  method alone.
- **Validation B — closed-loop round-trip.** Injects the Fortran amplitudes and
  phases into a UTide coefficient set, reconstructs a clean signal, and re-solves
  to confirm the two packages share the same amplitude units and Greenwich-phase
  convention.

### Example (station 8638901, Chesapeake Channel)

```bash
python bin/tidal_analysis/validate_fortran_ha.py \
    --fortran-log W8638901.log \
    --obs-file 8638901W_CB.obs \
    --latitude 37.033 \
    --station-id 8638901 \
    --station-name "Chesapeake Channel (CBBT)" \
    --out-dir ha_validation
```

### Inputs

- `--fortran-log` — a NOS Fortran HA log whose first result block has the
  `Constituent (H) (K) (K'-K) (K')` table.
- `--obs-file` — whitespace-delimited columns
  `day-of-year year month day hour minute value` (metres; values `<= -99` are
  treated as missing).

Run `python bin/tidal_analysis/validate_fortran_ha.py --help` for all options
(`--min-duration`, `--amp-tol`, `--phase-tol`, `--no-roundtrip`, ...).

The results of the 8638901 validation are summarised in the
[Harmonic Analysis wiki page](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/wiki/Harmonic-Analysis#validation-against-the-legacy-fortran-package).
