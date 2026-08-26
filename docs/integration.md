# In-situ integration boundary

STIC-INR receives one flow-field snapshot at a time through an in-situ
integration layer:

```text
simulation / in-situ transport
        |
        | step: int
        | coordinates: float[N, d]
        | values: float[N, c]
        v
normalize -> online_adapt -> flatten_parameters -> compressor.update
        |
        | adapted model and compressed weight-trajectory segments
        v
external persistence or reconstruction
```

The orchestration layer is responsible for starting the solver, receiving
snapshots, checking mesh/field compatibility, and storing outputs. The method
layer can be called as follows:

```python
from stic_inr import (
    StreamingWeightTrajectoryCompressor,
    flatten_parameters,
    online_adapt,
)
from stic_inr.adaptation import normalize_coordinates, normalize_field

model = load_offline_initialization()
compressor = StreamingWeightTrajectoryCompressor(
    residual_threshold=chosen_threshold
)

for snapshot in simulation_stream:
    coordinates = normalize_coordinates(snapshot.coordinates)
    values = normalize_field(snapshot.values)

    adapted = online_adapt(
        model,
        coordinates,
        values,
        learning_rate=1e-4,
        adaptation_steps=20,
        batch_size=50_000,
        in_place=True,
    )
    model = adapted.model

    vector, layout = flatten_parameters(model)
    update = compressor.update(snapshot.step, vector)

    if update.starts_new_segment:
        persist_closed_trajectory_segment(update.segments[-2])

segments = compressor.finalize()
```

To reconstruct model parameters for a time step, locate its segment and call:

```python
vector = segment.reconstruct(step)
restore_parameters(model, vector, layout)
```

The integration must preserve the `ParameterLayout` associated with the model
architecture. Changing layer widths, output components, or parameter order in
the middle of a sequence invalidates the temporal representation.

## SmartSim/OpenFOAM reference

`examples/smartsim_openfoam_reference.py` provides the corresponding control
script at API level. The OpenFOAM case is expected to contain a function object
equivalent to:

```text
functions
{
    pUPhiTest
    {
        type fieldsToSmartRedis;
        libs ("libsmartredisFunctionObjects.so");
        clusterMode off;
        fields (U);
        patches (internal);
        executeControl writeTime;
        writeControl writeTime;
    }
}
```

The function object publishes its dataset and tensor name templates through
`pUPhiTest_metadata`. The reference adapter reads these templates, so their
concrete naming remains the responsibility of the OpenFOAM/SmartRedis bridge.
