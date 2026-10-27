# CIAO multirate telemetry example

This example inspects the CIAO1 recording in Tiago Gomes, Paulo Garcia, Carlos
Correia and Nuno Morujão's [Proof-of-concept AO telemetry data using the AOT
standard format](https://doi.org/10.5281/zenodo.8192742), published 28 July 2023
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The authors
attribute the CIAO data to ESO programme `60.A-9278(C)` and describe its
translation through `aotpy`.

The original FITS bytes remain unchanged and are not redistributed here.
Noether Replay retains derived metadata and qualification summaries. This use
does not imply endorsement by the dataset authors or the observatory.

| Source identity | Value |
| --- | --- |
| File | `CIAO1_2019-12-06_DATA_EXPO-015808.fits` |
| Size | 19,751,040 bytes |
| Publisher MD5, verified | `069b37b24997bf55c5312a7bad469502` |
| Computed SHA-256 | `cd9e05a65c90e459df28fb8c2d4671381a08060d10b012c472dc41a6bdfd3d0e` |
| Implementation | `0176f741a8c93dcbf727370f198e4641e3c21ace` |

## What the recording establishes

The [report](../examples/reports/ciao.md) retains four streams: 15,000 WFS
measurements, 15,000 high-order command records, 15,000 tip-tilt command records
and 316 detector frames. The [counter summary](../examples/reports/ciao-coverage.json)
contains these independently computed set relationships:

| Counter coverage | Loop measurements | Pixel frames |
| --- | ---: | ---: |
| First counter | 215827 | 215900 |
| Last counter | 230826 | 231650 |
| Consecutive counter increment | 1 | 50 |
| Pixel counter values also in the loop record | — | 299 |
| Pixel counter values beyond the loop record | — | 17 |

The report counts 15,435 unrepresented integer pixel IDs, with an explicit
warning that these are not proof of detector-frame loss. The regular increment
of 50 is compatible with subsampling; the file does not establish that policy.
Shared counter values also do not establish physical dependency or event timing.

The source supplies generic timestamps for the loop streams and none for the
pixel stream. Exposure boundaries, observation availability, actuator-effective
times and clock uncertainty are not established. No physical event sequence is
inferred from the nominal 499.962 Hz loop rate. Both command streams remain
`command_recorded`; applied-command timing qualification rejects them. Calibration
matrices do not supply calibration validity intervals, so temporal calibration
coverage remains unqualified. No probe response or alternative-controller
performance is estimated. The report's 19 findings are evidence gaps in this
demonstration file, not demonstrated instrument faults.

## Reproduce

Use the implementation commit and dependency versions in the separate
[manifest](../examples/reports/ciao-manifest.json). Its reports were generated
from source with Python 3.11.15, NumPy 2.4.6, Astropy 8.0.1 and `aotpy` 3.2.1.
The earlier examples retain their original implementation manifests.

From a checkout with the AOT dependencies installed, this shell block downloads
one file into a new temporary directory. It rejects a response beyond the stated
size, verifies both hashes, and only then invokes the offline CLI. The download
is separate from the test suite and does not fetch external FITS references.

```sh
set -eu
python -m pip install -e '.[aot]' -r requirements-reproduce.txt
CIAO_DIR=$(mktemp -d)
export CIAO_DIR
python - <<'PY'
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

name = "CIAO1_2019-12-06_DATA_EXPO-015808.fits"
url = f"https://zenodo.org/api/records/8192742/files/{name}/content"
path = Path(os.environ["CIAO_DIR"]) / name
limit = 19_751_040
md5, sha256 = hashlib.md5(), hashlib.sha256()
size = 0
created = False
try:
    with urlopen(url, timeout=60) as response, path.open("xb") as output:
        created = True
        while block := response.read(min(65536, limit - size + 1)):
            size += len(block)
            if size > limit:
                raise ValueError("Source exceeds the expected size")
            md5.update(block)
            sha256.update(block)
            output.write(block)
    if (size != limit
        or md5.hexdigest() != "069b37b24997bf55c5312a7bad469502"
        or sha256.hexdigest() != "cd9e05a65c90e459df28fb8c2d4671381a08060d10b012c472dc41a6bdfd3d0e"):
        raise ValueError("Source identity check failed")
except Exception:
    if created:
        path.unlink(missing_ok=True)
    raise
PY
noether-replay "$CIAO_DIR/CIAO1_2019-12-06_DATA_EXPO-015808.fits" \
  --output "$CIAO_DIR/ciao.json"
noether-replay "$CIAO_DIR/CIAO1_2019-12-06_DATA_EXPO-015808.fits" \
  --format markdown --output "$CIAO_DIR/ciao.md"
```

The [JSON report](../examples/reports/ciao.json) binds the imported recording and
event timeline to content hashes. The manifest also hashes every retained
artifact. Content hashes establish consistency with the named source, not
independent authentication of acquisition or cross-platform bitwise identity.

To recompute the counter overlap in the same shell:

```sh
python - <<'PY'
import os
from pathlib import Path
from noether_replay.aot import read_aot

recording = read_aot(Path(os.environ["CIAO_DIR"]) / "CIAO1_2019-12-06_DATA_EXPO-015808.fits")
streams = {stream.id: stream for stream in recording.streams}
loop = set(map(int, streams["wfs:WFS:measurements"].frame_ids))
pixels = list(map(int, streams["wfs:WFS:pixels"].frame_ids))
print("shared:", len(set(pixels) & loop))  # 299
print("outside:", len(set(pixels) - loop))  # 17
print("steps:", sorted({b - a for a, b in zip(pixels, pixels[1:])}))  # [50]
print("all outside values exceed loop maximum:", min(set(pixels) - loop) > max(loop))
PY
```
