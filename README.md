# mcap_anonymizer

Blurs the faces in the camera images of [MCAP](https://mcap.dev/) files (ROS 2 bags), so
datasets recorded among people can be published without privacy issues, while keeping
everything else exactly as recorded.

- **Minimal changes**: only the pixels inside the detected faces change. For raw images
  (`sensor_msgs/Image`) every other pixel stays bit-exact, with the same encoding, `step`,
  header and message size. Bayer images are processed on each color plane of the mosaic, so
  they can still be demosaiced as usual.
- **Everything else untouched**: all other messages are copied byte by byte, with their log and
  publish times and sequence numbers; schemas, channels (also those without messages),
  metadata, attachments and the chunk compression are kept. For a rosbag2 directory, its
  `metadata.yaml` is copied.
- **Compressed images** (`sensor_msgs/CompressedImage`): PNG is re-encoded losslessly; JPEG is
  re-encoded with its own quantization tables and chroma subsampling (a small re-encoding
  loss in the whole image).
- **No face left behind on purpose**: low detection threshold, boxes enlarged by a margin and
  propagated to the neighboring frames (covers frames where the detector misses the face),
  and a review video to check the result. Detections can be edited by hand (add boxes for
  missed faces) before applying them.

Faces are detected with [CenterFace](https://github.com/Star-Clouds/CenterFace) (ONNX), on the
CPU or on an NVIDIA GPU.

## Install

```bash
git clone https://github.com/MOLAorg/mcap_anonymizer.git
cd mcap_anonymizer
python3 -m venv .venv
.venv/bin/pip install -e .
```

**GPU (optional, much faster):** install `onnxruntime-gpu` instead of `onnxruntime`:

```bash
.venv/bin/pip uninstall -y onnxruntime
.venv/bin/pip install 'onnxruntime-gpu[cuda,cudnn]'
```

Older GPUs (Pascal, e.g. GTX 10xx, Quadro P series) are not supported by the CUDA 13 libraries
of recent versions; this combination works on them:
`pip install 'onnxruntime-gpu[cuda,cudnn]==1.22.0' 'nvidia-cudnn-cu12==9.5.1.17'`.

## Usage

All in one go (all the image topics of the input):

```bash
mcap-anonymize run input.mcap output.mcap --detections detections.json
mcap-anonymize run rosbag2_dir/ output_dir/            # rosbag2 directory with .mcap files
```

Recommended workflow for a dataset to publish:

```bash
# 1. Detect the faces (slow part)
mcap-anonymize detect input.mcap detections.json

# 2. Anonymize
mcap-anonymize apply input.mcap detections.json output.mcap

# 3. Check the result: video of each camera, with the detections outlined
mcap-anonymize review output.mcap /camera/left/image_raw review.mp4 --detections detections.json

# 4. If a face was missed, add a manual box to detections.json ("manual" list, see below)
#    and apply again
```

Main options:

| Option | Default | Description |
|---|---|---|
| `--include REGEX`, `--exclude REGEX` | all | Image topics to process |
| `--threshold` | 0.2 | Detection score threshold; lower blurs more (and more false positives) |
| `--upscale` | 1.0 | Run the detector on the image scaled by this factor, for very small faces (2: ~4x slower) |
| `--device` | auto | `cuda` (with `onnxruntime-gpu`) or `cpu` |
| `--jobs`, `--threads` | 1, 0 | Parallel detector processes, and threads for each (CPU) |
| `--window` | 0.3 s | Detections are also applied to the frames within +-window |
| `--margin` | 0.2 | Boxes are enlarged by this fraction of their size on each side |
| `--mode` | blur | `blur` (strong box blur), `pixelate` or `solid` |

The anonymized region is the ellipse inscribed in each (enlarged) box.

### Detections file

```json
{
  "version": 1,
  "topics": {
    "/camera/image_raw": {"1793639112324938012": [[x1, y1, x2, y2, score]]}
  },
  "manual": [
    {"topic": "/camera/image_raw", "from": 1793639112000000000, "to": 1793639115000000000,
     "box": [400, 120, 460, 190]}
  ],
  "frames": {"/camera/image_raw": 12187}
}
```

Keys of `topics` are message log times (ns); only frames with detections are listed. Manual
boxes apply to all the frames of the topic with log time in `[from, to]`.

## Supported images

- `sensor_msgs/msg/Image`: `mono8/16`, `rgb8/16`, `bgr8/16`, `rgba8/16`, `bgra8/16`, `8UC1/3/4`,
  `16UC1`, and Bayer `bayer_{rggb,bggr,gbrg,grbg}{8,16}`.
- `sensor_msgs/msg/CompressedImage`: JPEG and PNG.

Other encodings (e.g. YUV 4:2:2, float depth images) stop the process with an error, so no
camera is silently left unprocessed: skip them with `--exclude` if they cannot show faces.
Video topics (`foxglove_msgs/CompressedVideo`) are not supported.

## Limitations

No face detector finds every face: faces seen from far away, from the side or backlit can be
missed in some frames. The temporal window and the margin reduce this, but **review the result**
before publishing (step 3), and add manual boxes where needed.

## License

BSD 3-Clause (see `LICENSE`). The detector code is based on
[CenterFace](https://github.com/Star-Clouds/CenterFace) and its adaptation in
[deface](https://github.com/ORB-HD/deface), both MIT; `models/centerface.onnx` is an
unmodified copy of CenterFace's `centerface_bnmerged.onnx`. Their licenses are in
`src/mcap_anonymizer/models/`.
