# VFD-CBD — Coherence Boundary Diagnostic

![Failure Mode Map](images/mode_map.png)

# VFD-CBD — Coherence Boundary Diagnostic

A reproducible simulator for mapping coherence and stability boundaries in coupled phase-oscillator networks.

This tool analyzes how detuning and coupling-to-loss ratios affect global coherence and identifies structured failure regimes.

VFD-CBD is a deterministic numerical simulator for studying coherence, stability,
and failure modes in coupled phase-oscillator networks. The tool maps stability
regions and collapse boundaries as a function of detuning and coupling-to-loss,
and classifies distinct failure regimes such as decoherence, cluster splitting,
and phase inversion.

## What This Repository Contains

This repository includes:

- A fixed-step, reproducible Kuramoto-type simulator
- A parameter-sweep engine for stability mapping
- Diagnostics for coherence collapse and failure modes
- A fully reproducible reference run (v0.1.0 artifact)

![Failure Mode Map](images/mode_map.png)

## Scope and Non-Claims

This code implements a stability and coherence diagnostic for abstract
coupled oscillator systems.

It does NOT:

- model or predict power output, thrust, or energy extraction
- describe or enable construction of physical devices
- make claims about propulsion, gravity control, or exotic technologies
- reference or depend on classified systems or data

All parameters are dimensionless or normalized. Results should be interpreted
strictly as nonlinear dynamical systems diagnostics.

## Reproducing the v0.1.0 Results

The directory:

```
out/run_20260201_001137_b37ac85f4c4c/
```

contains a fully resolved configuration, raw data, and plots for the reference
81x81 sweep used in the initial public release.

To regenerate plots from this run:

```bash
vfd-cbd plot --run out/run_20260201_001137_b37ac85f4c4c/
```

## Running a New Sweep

```bash
vfd-cbd sweep --config configs/killer_final.yaml
```

Results will be written to `out/run_<timestamp>_<hash>/`.

## Installation

```bash
pip install -e ".[dev]"
```

## Status

This is an initial public diagnostic release (v0.1.0). The API, configuration
schema, and diagnostics are expected to evolve.

## License

MIT License - see [LICENSE](LICENSE) file.

