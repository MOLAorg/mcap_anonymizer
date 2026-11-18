# mcap_anonymizer

[![test](https://github.com/MOLAorg/mcap_anonymizer/actions/workflows/test.yml/badge.svg)](https://github.com/MOLAorg/mcap_anonymizer/actions/workflows/test.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Blurs the faces in the camera images of [MCAP](https://mcap.dev/) files (ROS 2 bags), so
datasets recorded among people can be shared without privacy issues, while keeping everything
else exactly as it was recorded.

- **Only the faces change.** In raw images (`sensor_msgs/Image`), every pixel outside the
  blurred faces stays bit-exact, and so do the encoding, `step`, header and message size. Each
  encoding is processed in its own format: a Bayer image is still a valid mosaic, a YUV 4:2:2
  image keeps its chroma subsampling, 16-bit images keep their depth and byte order.
- **Everything else is untouched.** All other messages are copied byte by byte, with their log
  and publish times and sequence numbers, in the same order. Schemas, channels (also those
  without messages, with their QoS metadata), metadata and attachment records, and the chunk
  compression are kept. For a rosbag2 directory, its `metadata.yaml` is copied.
- **Compressed images** (`sensor_msgs/CompressedImage`): PNG is re-encoded losslessly (same
  channels and bit depth); JPEG is re-encoded with its own quantization tables and chroma
  subsampling, so the extra loss in the rest of the image is minimal.
- **Privacy first.** A low detection threshold, boxes enlarged by a margin and propagated to the
  neighboring frames (so a face missed by the detector in a few frames is still covered), a
  review video to check the result, and hand-added boxes for anything missed. Topics that may
  contain images but cannot be processed (e.g. compressed video) stop the tool, instead of being
  silently copied.
- **Large and unfinished bags.** Files are streamed (memory use is about one MCAP chunk, whatever
  the bag size), and recordings that were never closed (no summary section, truncated last
  chunk) are processed up to their last complete record. The output is always a complete,
  indexed MCAP file.

Faces are detected with [CenterFace](https://github.com/Star-Clouds/CenterFace) at a low
threshold (high recall), and doubtful detections are kept only if a person detector
([YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX)) finds a person around them, which
removes most false positives (foliage, wheels, machinery...). Both are ONNX models that run with
OpenCV (no extra dependencies), or with [onnxruntime](https://onnxruntime.ai/) on the CPU or an
NVIDIA GPU. All the bundled code and models have permissive licenses (BSD, MIT, Apache-2.0).

## Install

### ROS 2 (Humble, Jazzy, Kilted, Rolling)

From source, in a colcon workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/MOLAorg/mcap_anonymizer.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --packages-select mcap_anonymizer
source install/setup.bash
ros2 run mcap_anonymizer mcap_anonymize --help
```

Its only dependencies are `python3-numpy`, `python3-opencv` and `python3-pil`; the MCAP reader
and writer (the [mcap](https://github.com/foxglove/mcap) Python library, MIT) is bundled.

### pip (no ROS needed)

```bash
pip install 'mcap-anonymizer[compression] @ git+https://github.com/MOLAorg/mcap_anonymizer.git'
mcap_anonymize --help
```

### Optional extras

- **zstd or lz4 compressed MCAP chunks** need the Python packages `zstandard` or `lz4`
  (Ubuntu: `sudo apt install python3-zstandard python3-lz4`; pip: the `[compression]` extra).
  Bags recorded with rosbag2's default MCAP settings are not compressed.
- **Faster detection with onnxruntime**: `pip install onnxruntime`. It is used automatically
  when installed (`--backend auto`).
- **NVIDIA GPU**: `pip install 'onnxruntime-gpu[cuda,cudnn]'` (and not `onnxruntime`), then
  `--device cuda` (or `auto`). Older GPUs (Pascal: GTX 10xx, Quadro Pxxx) are not supported by
  the CUDA 13 libraries of the latest versions; on them, use
  `pip install 'onnxruntime-gpu[cuda,cudnn]==1.22.0' 'nvidia-cudnn-cu12==9.5.1.17'`.

## Quick start

```bash
mcap_anonymize run input.mcap output.mcap --detections detections.json
mcap_anonymize run my_rosbag2_dir/ my_rosbag2_dir_anonymized/   # all the .mcap files of a bag
```

With ROS 2, use `ros2 run mcap_anonymizer mcap_anonymize` instead of `mcap_anonymize`.

All the image topics are processed. The tool prints how many images of each topic were
anonymized, and `detections.json` keeps the detected faces.

## Recommended workflow to publish a dataset

```bash
# 1. Detect the faces (the slow part; see "Performance")
mcap_anonymize detect input.mcap detections.json

# 2. Anonymize (fast: only reads, blurs and writes)
mcap_anonymize apply input.mcap detections.json output.mcap

# 3. Check the result: one video per camera, with the anonymized faces outlined in red, and the
#    doubtful detections rejected by the person check in yellow
mcap_anonymize review output.mcap /camera/left/image_raw left.mp4 --detections detections.json

# 4. If a face was missed, add a manual box to detections.json (see below) and apply again
```

No face detector finds every face (see "Limitations"), so do not skip the review before
publishing. Keep the original files: anonymization cannot be undone.

## Commands and options

- `mcap_anonymize run INPUT OUTPUT [--detections FILE]`: detect + apply.
- `mcap_anonymize detect INPUT DETECTIONS`: writes the detections file.
- `mcap_anonymize verify INPUT DETECTIONS OUTPUT_DETECTIONS`: applies the person check to an
  existing detections file (e.g. one made with `--verify none`).
- `mcap_anonymize apply INPUT DETECTIONS OUTPUT`: anonymizes with a detections file.
- `mcap_anonymize review INPUT TOPIC OUTPUT.mp4`: video of one image topic.

`INPUT` is a `.mcap` file or a rosbag2 directory (all its `.mcap` files; `OUTPUT` is then a
directory, with the same file names and a copy of `metadata.yaml`).

| Option | Default | Commands | Description |
|---|---|---|---|
| `--include REGEX` | all | run, detect, apply | Only anonymize the image topics matching this regex |
| `--exclude REGEX` | none | run, detect, apply | Leave the image topics matching this regex untouched |
| `--drop REGEX` | none | run, detect, apply | Remove the topics matching this regex from the output |
| `--ignore-unsupported` | off | run, detect, apply | Copy topics that may carry images but cannot be anonymized as they are (with a warning), instead of stopping |
| `--threshold T` | 0.2 | run, detect | Detection score threshold. Lower finds more faces, and more false positives |
| `--verify` | person | run, detect | `person`: faces scoring below `--verify-below` are only kept with a person around them (their center in the head region of a person box); `none`: keep every detection (maximum recall, more false positives) |
| `--verify-below S` | 0.7 | run, detect, verify | Faces with a lower score need a person around them |
| `--person-threshold T` | 0.2 | run, detect, verify | Person detection score threshold |
| `--upscale F` | 1.0 | run, detect | Run the detector on the image scaled by F (e.g. 2 for small faces in low-resolution images; about F² times slower) |
| `--backend` | auto | run, detect, verify | `opencv`, `onnxruntime`, or `auto` (onnxruntime if installed) |
| `--device` | auto | run, detect, verify | `cpu` or `cuda` (onnxruntime-gpu) |
| `--jobs N` | 1 | run, detect | Parallel detector processes (CPU) |
| `--threads N` | 0 | run, detect, verify | Threads of each detector process (0: library default) |
| `--window S` | 0.3 | run, apply | The detections of each frame are also applied to the frames within ±S seconds |
| `--margin M` | 0.2 | run, apply | Each box is enlarged by M times its size on each side |
| `--mode` | blur | run, apply | `blur` (strong box blur), `pixelate` (about 6 blocks across the face) or `solid` (black) |
| `--every N` | 1 | review | Use one image out of N |
| `--scale S` | 0.5 | review | Image scale in the video |
| `--only-detections` | off | review | Only frames within 1 s of a detection |

The anonymized region is the ellipse inscribed in each (enlarged) box. The blur kernel is a
third of the face size, applied twice, which leaves no recognizable detail.

### Detections file

```json
{
  "version": 1,
  "topics": {
    "/camera/image_raw": {"1793639112324938012": [[412.5, 120.0, 448.0, 165.5, 0.81]]}
  },
  "rejected": {
    "/camera/image_raw": {"1793639113324938012": [[10.0, 300.5, 25.0, 318.0, 0.23]]}
  },
  "manual": [
    {"topic": "/camera/image_raw", "from": 1793639112000000000, "to": 1793639115000000000,
     "box": [400, 120, 460, 190]}
  ],
  "frames": {"/camera/image_raw": 12187}
}
```

`topics` lists, for each image topic, the frames with detections: message log time (ns) ->
boxes `[x1, y1, x2, y2, score]` in pixels. `rejected` lists, in the same way, the doubtful
detections discarded by the person check: they are not anonymized (move a box from `rejected`
to `topics` to anonymize it anyway). `manual` boxes (written by hand) apply to every
frame of their topic with a log time in `[from, to]` (ns), as they are (no margin). `frames`
counts the processed images.

## Supported messages and encodings

| Message | Encodings | Notes |
|---|---|---|
| `sensor_msgs/msg/Image` | `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, `8UC1`, `8UC3`, `8UC4` | |
| | `rgb16`, `bgr16`, `rgba16`, `bgra16`, `mono16`, `16UC3`, `16UC4` | Little or big endian. For detection, 16-bit images are stretched to 8 bits |
| | `bayer_rggb8`, `bayer_bggr8`, `bayer_gbrg8`, `bayer_grbg8` and their `16` versions | Each color plane of the mosaic is blurred on its own |
| | `uyvy` / `yuv422`, `yuyv` / `yuv422_yuy2` | Y, U and V samples blurred on their own |
| | `nv21`, `nv12`, `nv24` | Y plane and chroma plane blurred on their own |
| `sensor_msgs/msg/CompressedImage` | JPEG and PNG, any format string (`jpeg`, `bgr8; jpeg compressed bgr8`, `mono16; png compressed mono16`...) | |

- **Not camera images** (left untouched, with a notice): depth images (`16UC1`, the ROS
  convention for depth in millimeters, `32FC1`, and `compressedDepth` CompressedImages) and other
  data encodings (`32FC*`, `64FC*`, `8SC*`, `16SC*`, `32SC*`, `8UC2`, `16UC2`). 16-bit cameras
  publish `mono16`, which is processed.
- **Unknown encodings** stop the tool with an error, so no camera is silently left unprocessed.
  Use `--exclude` for topics that cannot show faces.
- **Other image-like messages** (`foxglove_msgs/CompressedVideo`, `ffmpeg_image_transport`,
  `theora_image_transport` packets, `foxglove_msgs` images, or any non-CDR image) cannot be
  anonymized: the tool stops and lists them. Remove them with `--drop`, or keep them with
  `--ignore-unsupported` if you know they show no people.
- Only CDR-encoded messages (ROS 2) are processed. ROS 1 bags and non-ROS MCAP files are not
  supported.
- LiDAR "images" published as `mono16`/`mono8` (e.g. range, signal or near-IR images of
  Ouster sensors) are processed like any camera; exclude them with `--exclude` if a false
  positive must not alter their values.

## Performance

Detection dominates the run time. Rough figures per image, single process:

| Image | OpenCV, CPU, 1 thread | onnxruntime, GPU (Quadro P620) |
|---|---|---|
| 1024x768 | ~0.5 s | ~0.08 s |

Use `--jobs` to run several detector processes on a multi-core CPU, or a GPU. `apply` is
limited by the disk (it rewrites the whole file); memory use is small and does not depend on
the bag size.

## Limitations

- No face detector finds every face. Faces that are very small (below ~10 pixels), seen from
  behind or from the side, backlit, blurred by motion or partially occluded can be missed in
  some frames. The temporal window and the margin reduce this; the review step and manual
  boxes cover the rest.
- The person check discards doubtful faces with no detectable body around them: a face seen
  through a window or behind an obstacle, with only the face visible, may then be missed if it
  also scores low. Use `--verify none` for maximum recall (at the cost of blurring false
  positives too), and check the yellow boxes in the review video.
- Remaining false positives are blurred too; this only affects small regions.
- Faces are the only personal data handled: license plates, screens or documents are not.
- Depth images, point clouds and other sensor data are not modified.

## How it works

1. **Reading**: the MCAP file is read record by record, in file order. Image messages
   (`sensor_msgs` Image/CompressedImage, CDR) are located; their `data` field is found by
   parsing the CDR header fields, so no ROS message libraries are needed.
2. **Detection**: each image is converted to RGB (demosaicing Bayer, converting YUV, stretching
   16-bit) only to run CenterFace on it. In images with doubtful faces (score below
   `--verify-below`), YOLOX-s looks for people, and each doubtful face is kept only if its center
   is in the head region (upper 45 %) of a person box.
3. **Anonymization**: for each frame, the boxes of the frame and of its neighbors in time are
   enlarged and blurred on the original buffer: each plane of the encoding (the image, the four
   Bayer color planes, or the Y, U and V samples) is filtered in its own resolution and data
   type, only inside the ellipses. Only those bytes change; the `data` field is replaced in the
   serialized message, keeping all the other bytes.
4. **Writing**: a new MCAP file with the same schemas, channels, messages (same times and
   sequence numbers), metadata, attachments and chunk compression.

## License

BSD 3-Clause (see [LICENSE](LICENSE)).

Bundled third-party code and models (licenses next to them):

- `mcap_anonymizer/_vendor/mcap`: the [mcap](https://github.com/foxglove/mcap) Python library
  1.6.0, by Foxglove (MIT; one import made relative, version string fixed).
- `mcap_anonymizer/models/centerface.onnx`: CenterFace's `centerface_bnmerged.onnx`, with its
  input and output sizes declared as dynamic (MIT).
- `mcap_anonymizer/models/yolox_s.onnx`: the YOLOX-s model released by Megvii
  ([YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), Apache-2.0), unmodified.
- The face detector decoding code is based on CenterFace and on its adaptation in
  [deface](https://github.com/ORB-HD/deface) (MIT).
