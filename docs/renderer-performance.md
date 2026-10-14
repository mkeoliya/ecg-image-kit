# ECG-Image-Kit: measured optimization

The final candidate renders the same signal samples, DPI, layouts, calibration pulses, lead labels, grids and augmentations faster. All five configurations use six workers, 12 distinct PTB-XL patients repeated five times (60 images per configuration), with a synchronized start after one warm-up per worker. Source originals and shared generation are unchanged.

| Configuration | Original images/s | Optimized images/s | Steady speedup | Mean seconds/image original → optimized | PNG size ratio |
|---|---:|---:|---:|---:|---:|
| 100 DPI / 100 Hz / 1 column / grid | 2.45 | 23.09 | 9.43× | 2.189 → 0.235 | 1.000× |
| 150 DPI / 250 Hz / 2 columns / grid | 2.01 | 14.96 | 7.44× | 2.557 → 0.352 | 1.000× |
| 200 DPI / 500 Hz / 4 columns / grid | 1.69 | 10.35 | 6.11× | 3.155 → 0.537 | 1.000× |
| 150 DPI / 250 Hz / 2 columns / noise + blur + JPEG | 1.21 | 6.93 | 5.71× | 4.763 → 0.772 | 1.142× |
| 200 DPI / 500 Hz / 4 columns / grid off / BW + blur | 2.47 | 10.56 | 4.27× | 2.349 → 0.547 | 1.164× |

These are rendering-stage steady rates, including WFDB signal loading, plotting, augmentations, PNG encoding and annotation writes. Selecting records, resampling and writing the source/target snapshots occur before this benchmark. `results.json` also reports startup-inclusive timings, p95 latency, CPU time, peak RSS and file-size ratios. Other jobs run on the same host; this is not an idle-machine ceiling. The grid-off workload does not reach 5× in the final test.

## What changed

- Batch all waveform coordinate transforms and float64 rounding, retaining every annotated sample and exact values.
- Use `Figure.savefig` instead of pyplot save, eliminating a second full Agg draw; avoid recreating a discarded figure after close.
- Retain at most four Figure/Axes objects and cache only the static axis/grid raster. Waveforms, spines and text retain the original draw order. Cache keys include dimensions, DPI, grid visibility/colors and axis limits; style changes invalidate the raster.
- Apply tick/grid styles only when they change.
- Compile the layout adapter and read YAML once per persistent worker, using fresh copied settings per record.
- Apply augmentations directly to the Agg raster, eliminating the intermediate PNG encode/decode and annotation rewrite.
- Chunk noise generation without changing its seeded random sequence; clamp in place to reduce temporary memory.
- Compact annotations without dropping any fields. Optional PNG filter None and zlib settings trade modest file size for lossless encoding speed. Noise/JPEG uses level 1 with Huffman-only; BW/blur uses level 1 with the default strategy. Clean images retain normal PNG compression and their bytes are identical to the original outputs.

## Validation

All 300 primary images match every decoded raster value and the full parsed annotation object. Validation checks point counts, source sample intervals, sampling frequency, lead names and 25 mm/s / 10 mm/mV calibration against the calibrated WFDB snapshots. Clean PNG files also match byte for byte. Noise, blur and JPEG parameters are identical; the optional PNG encoder changes compression, not pixels. A separate mixed sequence alternates grids, colors, pulses, layouts and 1/2/6/12 leads within one worker. The standalone production adapter is self-contained and was compared with the profiled adapter.

PNG encoder experiments in `profiles/png_codecs.json` and `profiles/png_no_filter.json` decode their outputs and compare the original RGB samples exactly. PNG signature, dimensions, bit depth, color type, zlib stream and chunk CRCs are standard. No model training or GPU computation was launched.

## Usage and scope

Based on upstream commit `27b90f56896c9fc78b05a83ca14844ea2637aa0b`.
The timings above use persistent, isolated worker processes; they are not
benchmarks of every upstream command-line augmentation. The core renderer
changes are in `ecg_plot.py` and `extract_leads.py`.

`tools/render_record.py` supplies the measured persistent-worker path. It
accepts line-delimited JSON jobs on stdin and emits one JSON result per job:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/render_record.py --kit "$PWD" --server
```

Each job has absolute `record` (WFDB record path without extension), `output`
(directory), and `config` fields. Example config for a 10-second record:

```json
{"columns":4,"resolution":200,"duration_s":10,"rhythm_lead":"II","grid":true,"black_white":false,"grid_color":5,"seed":20261013,"calibration_pulse":true,"image_noise_std":0,"blur_radius":0,"jpeg_quality":100}
```

For the measured fast, lossless PNG path on augmented images, add
`"png_encoder":"nofilter", "png_compress_level":1` and
`"png_compress_type":2` for noise/JPEG or `0` for blur-only images.
Clean images retain the default PNG encoder. The optional encoder changes
compression and file size, not decoded pixels. Noise standard deviation is
in 8-bit pixel values, blur radius in pixels, and JPEG quality is 1–100.
The worker preserves original annotation fields and includes augmentation
settings. It disables the upstream duplicate WFDB export: callers retain
source signals and use annotations to establish which samples are visible.

The worker is single-threaded. Run separate workers for parallel generation;
Matplotlib's figure cache is not intended for concurrent threads. The
`image2ecg`/`ecgjev` generator uses this process model and records source,
renderer, configuration, and worker hashes for reproducibility. No patient
images, acquired waveforms, or credentials are included in this fork.

Test environment: Python 3.10, NumPy 1.26.4, Matplotlib 3.10.9,
Pillow 12.3.0, SciPy 1.15.3, WFDB 4.3.1, PyYAML 6.0.3, imgaug 0.4.0.
Keep NumPy below 2 for the upstream imgaug dependency. Research benchmark
scripts, source snapshots, detailed profiles, and equivalence results remain
in the parent project's `tmp/ecg-image-kit-optimization/` workspace.
