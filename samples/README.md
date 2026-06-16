# Test samples

The test suite synthesizes its own clip, so you don't need a real gameplay video.

## Generate a synthetic clip with known loud bursts

This creates a 45-second video: a quiet base tone with loud bursts at **t=5s, 20s,
40s** (plus a color change at each burst so the montage is visually obvious).
Because the loud moments are at known timestamps, detection can be asserted exactly.

```bash
ffmpeg -y \
  -f lavfi -i "color=c=navy:s=640x360:d=45" \
  -f lavfi -i "sine=frequency=220:duration=45:sample_rate=16000" \
  -f lavfi -i "sine=frequency=880:duration=45:sample_rate=16000" \
  -filter_complex "\
    [1:a]volume=0.05[base]; \
    [2:a]volume=4.0, \
      aselect='between(t,5,6)+between(t,20,21)+between(t,40,41)':sample_rate=16000, \
      asetpts=N/SR/TB[bursts]; \
    [base][bursts]amix=inputs=2:duration=first[a]" \
  -map 0:v -map "[a]" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  samples/sample.mp4
```

The automated tests build an equivalent clip on the fly (see
`tests/conftest.py`), so running them needs only ffmpeg on PATH.
