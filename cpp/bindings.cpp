#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "reaper/core/track.h"
#include "reaper/epoch_tracker/epoch_tracker.h"

namespace py = pybind11;

// Float32 audio [-1, 1] → int16 PCM, which REAPER's Init() expects.
static std::vector<int16_t> float_to_int16(const float* data, int n) {
    std::vector<int16_t> out(n);
    for (int i = 0; i < n; ++i) {
        float s = data[i] * 32768.0f;
        if (s > 32767.0f)  s = 32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        out[i] = static_cast<int16_t>(s);
    }
    return out;
}

py::dict track(
    py::array_t<float, py::array::c_style | py::array::forcecast> audio,
    int sample_rate = 16000,
    float min_f0 = 40.0f,
    float max_f0 = 500.0f,
    float frame_interval = 0.005f,
    float unvoiced_pulse_interval = 0.01f,
    float unvoiced_cost = 0.9f,
    bool do_highpass = true,
    bool do_hilbert_transform = false
) {
    py::buffer_info buf = audio.request();
    if (buf.ndim != 1)
        throw std::invalid_argument("audio must be a 1-D array");
    if (buf.size == 0)
        throw std::invalid_argument("audio array is empty");

    const float* ptr = static_cast<const float*>(buf.ptr);
    const int n = static_cast<int>(buf.size);

    std::vector<int16_t> pcm = float_to_int16(ptr, n);

    EpochTracker et;
    et.set_unvoiced_cost(unvoiced_cost);

    if (!et.Init(pcm.data(), n, static_cast<float>(sample_rate),
                 min_f0, max_f0, do_highpass, do_hilbert_transform)) {
        throw std::runtime_error("EpochTracker::Init failed");
    }

    if (!et.ComputeFeatures()) {
        throw std::runtime_error("EpochTracker::ComputeFeatures failed");
    }

    if (!et.TrackEpochs()) {
        throw std::runtime_error("EpochTracker::TrackEpochs failed");
    }

    // F0 and correlation resampled to a regular grid.
    std::vector<float> f0_vec, corr_vec;
    if (!et.ResampleAndReturnResults(frame_interval, &f0_vec, &corr_vec)) {
        throw std::runtime_error("EpochTracker::ResampleAndReturnResults failed");
    }

    const int nf = static_cast<int>(f0_vec.size());

    // Epoch/pitchmark times and their voicing flags.
    std::vector<float> epoch_times_vec;
    std::vector<int16_t> epoch_voicing_vec;
    et.GetFilledEpochs(unvoiced_pulse_interval, &epoch_times_vec, &epoch_voicing_vec);
    const int ne = static_cast<int>(epoch_times_vec.size());

    // Build output arrays.
    py::array_t<float>  time_arr(nf);
    py::array_t<float>  f0_arr(nf);
    py::array_t<bool>   voiced_arr(nf);
    py::array_t<float>  corr_arr(nf);
    py::array_t<float>  epochs_arr(ne);
    py::array_t<bool>   epoch_voiced_arr(ne);

    auto time_buf   = time_arr.mutable_unchecked<1>();
    auto f0_buf     = f0_arr.mutable_unchecked<1>();
    auto voiced_buf = voiced_arr.mutable_unchecked<1>();
    auto corr_buf   = corr_arr.mutable_unchecked<1>();
    auto ep_buf     = epochs_arr.mutable_unchecked<1>();
    auto epv_buf    = epoch_voiced_arr.mutable_unchecked<1>();

    for (int i = 0; i < nf; ++i) {
        time_buf(i)   = frame_interval * i;
        f0_buf(i)     = (f0_vec[i] > 0.0f) ? f0_vec[i] : 0.0f;
        voiced_buf(i) = (f0_vec[i] > 0.0f);
        corr_buf(i)   = corr_vec[i];
    }

    for (int i = 0; i < ne; ++i) {
        ep_buf(i)  = epoch_times_vec[i];
        epv_buf(i) = static_cast<bool>(epoch_voicing_vec[i]);
    }

    py::dict result;
    result["time"]          = time_arr;
    result["f0"]            = f0_arr;
    result["voiced"]        = voiced_arr;
    result["corr"]          = corr_arr;
    result["epochs"]        = epochs_arr;
    result["epoch_voiced"]  = epoch_voiced_arr;
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for the REAPER F0 tracker";
    m.def("track", &track,
        py::arg("audio"),
        py::arg("sample_rate")            = 16000,
        py::arg("min_f0")                 = 40.0f,
        py::arg("max_f0")                 = 500.0f,
        py::arg("frame_interval")         = 0.005f,
        py::arg("unvoiced_pulse_interval")= 0.01f,
        py::arg("unvoiced_cost")          = 0.9f,
        py::arg("do_highpass")            = true,
        py::arg("do_hilbert_transform")   = false,
        R"(
Estimate F0 (fundamental frequency) from a mono audio signal.

Parameters
----------
audio : np.ndarray, float32, shape (N,)
    Mono audio samples, expected range [-1, 1].
sample_rate : int
    Sample rate of the audio in Hz (default 16000).
min_f0 : float
    Minimum F0 to search for, in Hz (default 40).
max_f0 : float
    Maximum F0 to search for, in Hz (default 500).
frame_interval : float
    Output frame period in seconds (default 0.005).
unvoiced_pulse_interval : float
    Pulse interval in unvoiced regions, seconds (default 0.01).
unvoiced_cost : float
    Cost for unvoiced hypothesis (default 0.9).
do_highpass : bool
    Apply 80 Hz highpass filter to remove rumble (default True).
do_hilbert_transform : bool
    Apply Hilbert transform to reduce phase distortion (default False).

Returns
-------
dict with keys:
    time          : float32 array, timestamps of F0 frames (seconds)
    f0            : float32 array, F0 in Hz (0 in unvoiced frames)
    voiced        : bool array, voicing decision per frame
    corr          : float32 array, NCCF correlation value per frame
    epochs        : float32 array, glottal closure instant times (seconds)
    epoch_voiced  : bool array, voicing state at each epoch
        )");
}
