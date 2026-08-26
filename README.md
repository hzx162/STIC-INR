# STIC-INR

## Spatiotemporal In-Situ Compression of Unsteady CFD Data via Implicit Neural Representations

This repository contains the code associated with the paper

*STIC-INR: Spatiotemporal In-Situ Compression of Unsteady CFD Data via Implicit Neural Representations*.

Citation information will be added upon publication.

---

### Requirements

- Python v3.10+
- PyTorch v2.0+
- NumPy v1.24+
- OpenFOAM v2506
- SmartSim and SmartRedis

The Python dependencies are listed in `requirements.txt`. Dependencies used by
the SmartSim/OpenFOAM interface are listed in `requirements-smartsim.txt`.

### Running the Model

The `stic_inr/model.py` file defines the SIREN network used to represent each
flow-field snapshot. The `stic_inr/pretraining.py` and
`examples/offline_pretraining.py` files provide the offline meta-pretraining
procedure used to obtain the model initialization from historical CFD
snapshots.

The `stic_inr/adaptation.py` file performs online adaptation when a new snapshot
arrives. The `stic_inr/parameters.py` file converts each adapted model into a
weight vector. The `stic_inr/weight_trajectory.py` file performs streaming
piecewise compression of the resulting model-weight trajectory.

Offline pretraining can be started with

```bash
python examples/offline_pretraining.py \
  --coordinates coordinates.npy \
  --snapshots historical_snapshots.npy \
  --output initial_model.pth
```

The `examples/smartsim_openfoam_reference.py` file shows how to start OpenFOAM
with SmartSim, receive flow-field snapshots through SmartRedis, perform online
adaptation, and compress the adapted model-weight trajectory using streaming
piecewise segments. The corresponding
OpenFOAM function-object configuration is described in `docs/integration.md`.

### OpenFOAM Case

The `cases/depthCharge3D/` directory contains the source files for the
three-dimensional depth-charge case used in the paper. It includes the initial
conditions, physical properties, mesh definition, numerical schemes, solver
settings, and `fieldsToSmartRedis` configuration.

The mesh and initial nonuniform fields are generated from `blockMeshDict` and
`setFieldsDict` by the supplied `Allrun` script.

### Data and Model Weights

Historical flow-field snapshots can be supplied to
`examples/offline_pretraining.py` as NumPy arrays. The coordinates have shape
`[N, d]`, while scalar and vector snapshot sequences have shape `[T, N]` and
`[T, N, components]`, respectively.

The resulting offline initialization is saved as `initial_model.pth` and is
loaded by the SmartSim/OpenFOAM online compression script.
