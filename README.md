# Mäher

Python bindings for [REAPER](https://github.com/google/REAPER) using pybind11. This is meant to be an alternative to [pyreaper](https://github.com/r9y9/pyreaper) for quicker maintenence as part of integration in the [Besra](https://codeberg.org/besra) audio annotator. The original REAPER C++ algorithm is wrapped without algorithmic changes (only some upstream stdout diagnostic prints are removed). Numerical output is identical to the original REAPER CLI.

## Install

```bash
pip install maeher
```

Or build from source (requires CMake ≥ 3.15 and a C++14 compiler):

```bash
pip install ".[dev]"
```

For an editable/development install, pre-install the build backend and use
`--no-build-isolation`:

```bash
pip install scikit-build-core pybind11 numpy
pip install --no-build-isolation -e ".[dev]"
```

## Usage

```python
import numpy as np
import maeher

audio, sr = load_audio("speech.wav") # audio as mono float32, range [-1, 1]
result = maeher.track(audio, sample_rate=sr, min_f0=40, max_f0=500)

print(result.keys()) # dict_keys(['time', 'f0', 'voiced', 'corr', 'epochs', 'epoch_voiced'])

f0 = result["f0"]
voiced = result["voiced"]
epochs = result["epochs"]
```

## API

```python
maeher.track(
    audio, # 1-D float32 NumPy array
    *,
    sample_rate=16000, # Hz
    min_f0=40.0, # Hz
    max_f0=500.0, # Hz
    frame_interval=0.005, # seconds
    unvoiced_pulse_interval=0.01,
    unvoiced_cost=0.9,
    do_highpass=True,
    do_hilbert_transform=False,
) -> dict[str, np.ndarray]
```

| Output key      | dtype   | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `time`          | float32 | Timestamp of each frame (seconds)                |
| `f0`            | float32 | F0 estimate in Hz; 0 in unvoiced frames          |
| `voiced`        | bool    | Voicing decision per frame                       |
| `corr`          | float32 | NCCF correlation value per frame                 |
| `epochs`        | float32 | Glottal closure instant times (seconds)          |
| `epoch_voiced`  | bool    | Voicing state at each epoch                      |

## Migrating from pyreaper

[pyreaper](https://github.com/r9y9/pyreaper) returns a 5-tuple and takes
**int16** audio; maeher returns a dict and takes **float32** audio in `[-1, 1]`.
The underlying algorithm and parameters are the same, so migration is mechanical.

```python
# Before — pyreaper (x is int16 PCM)
from pyreaper import reaper
pm_times, pm, f0_times, f0, corr = reaper(
    x, fs, minf0=40.0, maxf0=500.0,
    frame_period=0.005, unvoiced_cost=0.9,
)

# After — maeher (audio is float32 in [-1, 1])
import maeher
audio = x.astype("float32") / 32768.0          # int16 → float32, only if your input was int16
r = maeher.track(
    audio, sample_rate=fs, min_f0=40.0, max_f0=500.0,
    frame_interval=0.005, unvoiced_cost=0.9,
)
pm_times, pm   = r["epochs"], r["epoch_voiced"]
f0_times, f0   = r["time"], r["f0"]
corr           = r["corr"]
```

Parameter names (same defaults):

| pyreaper              | maeher                     |
|-----------------------|----------------------------|
| `fs` (positional)     | `sample_rate=`             |
| `minf0`               | `min_f0`                   |
| `maxf0`               | `max_f0`                   |
| `frame_period`        | `frame_interval`           |
| `inter_pulse`         | `unvoiced_pulse_interval`  |
| `do_high_pass`        | `do_highpass`              |
| `do_hilbert_transform`| `do_hilbert_transform`     |
| `unvoiced_cost`       | `unvoiced_cost`            |

Behavioural differences to watch for:

- **Input dtype.** pyreaper expects `int16`; maeher expects `float32` scaled to
  `[-1, 1]`. Divide int16 by `32768.0` (as above). Already-float audio needs no scaling.
- **Unvoiced F0 sentinel.** pyreaper sets `f0 = -1.0` in unvoiced frames; maeher
  sets `f0 = 0.0` and additionally exposes a boolean `r["voiced"]` mask. Replace
  any `f0 > 0` / `f0 == -1` checks with `r["voiced"]` / `~r["voiced"]`.
- **Pitch-mark voicing dtype.** pyreaper's `pm` is `int32` (1/0); maeher's
  `epoch_voiced` is `bool`. Comparisons like `pm == 1` become `r["epoch_voiced"]`.

If you want a literal drop-in while migrating, wrap maeher to mimic the old tuple:

```python
import numpy as np, maeher

def reaper(x, fs, **kw):
    audio = x.astype("float32") / 32768.0 if np.issubdtype(x.dtype, np.integer) else x.astype("float32")
    name_map = {"minf0": "min_f0", "maxf0": "max_f0", "frame_period": "frame_interval",
                "inter_pulse": "unvoiced_pulse_interval", "do_high_pass": "do_highpass"}
    kw = {name_map.get(k, k): v for k, v in kw.items()}
    r = maeher.track(audio, sample_rate=fs, **kw)
    f0 = np.where(r["voiced"], r["f0"], -1.0).astype("float32")   # restore pyreaper's -1 sentinel
    pm = r["epoch_voiced"].astype("int32")
    return r["epochs"], pm, r["time"], f0, r["corr"]
```

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
