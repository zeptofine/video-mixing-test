import argparse
import random
import tempfile
from pathlib import Path

import ffmpeg
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

parser = argparse.ArgumentParser("video_shuffle", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("file", type=str, default="test.mp4", help="the file to read")
parser.add_argument("output", type=str, default="out.mp4", help="the filepath to output to")
parser.add_argument("--frames", type=int, help="the number of frames per chunk", default=2)
args = parser.parse_args()

file = Path(args.file)
assert file.exists()
split_size_frames = args.frames
output_path = Path(args.output)
output_path.parent.mkdir(exist_ok=True, parents=True)
assert output_path.parent.exists()


with tempfile.TemporaryDirectory(suffix="video-mixing-test") as tmpdir:
    tmpdir = Path(tmpdir)

    audiopath = tmpdir / "audio.wav"
    imagepaths = tmpdir / "%06d.png"

    filedata = ffmpeg.probe(file)
    video_stream_data: dict | None = None
    for stream in filedata["streams"]:
        if stream["codec_type"] == "video":
            video_stream_data = stream
            break
    assert video_stream_data is not None
    numerator, denominator = video_stream_data["avg_frame_rate"].split("/")
    fps: float = float(numerator) / float(denominator)
    single_frame_duration = 1 / fps

    fi = ffmpeg.input(file)
    print(f"separating audio to {audiopath}...")
    (
        fi.audio.output(str(audiopath), format="wav", ar="44100", loglevel="error")
        .global_args("-hide_banner", "-stats")
        .run()
    )

    samplerate, data = wavread(audiopath)
    total_length = len(data) / samplerate
    samples_per_frame: float = samplerate / fps
    audio_chunks = np.split(
        data,
        range(
            int(samples_per_frame),
            len(data),
            int(samples_per_frame * split_size_frames),
        ),
        axis=0,
    )

    print(f"saving video frames to {tmpdir}...")
    fi.video.output(str(imagepaths), loglevel="error").global_args("-hide_banner", "-stats").run()
    images = sorted(tmpdir.glob("*.png"))
    video_chunks = (images[i : i + split_size_frames] for i in range(0, len(images), split_size_frames))

    # bind video and audio chunks, shuffle them, then split again
    v_chunks_, a_chunks = zip(
        *sorted(
            zip(
                video_chunks,
                audio_chunks,
            ),
            key=lambda _: random.random(),
        )
    )
    v_chunks = (y for x in v_chunks_ for y in x)  # [[a,b],[c,d]] -> [a,b,c,d]
    shuffled_audio = np.vstack(a_chunks)

    print(f"saving shuffled audio to {audiopath}...")
    wavwrite(audiopath, samplerate, shuffled_audio)

    audioinput = ffmpeg.input(audiopath).audio
    # create new folder of symlinks with the files in order
    with tempfile.TemporaryDirectory(suffix="video-mixing-test-links") as linkdir:
        print(f"symlinking ordered images to {linkdir}...")
        linkdir = Path(linkdir)
        for filename, origin in zip([i.relative_to(tmpdir) for i in images], v_chunks):
            (linkdir / filename).symlink_to(origin)

        vidinput = ffmpeg.input(linkdir / "%06d.png", r=fps).video.filter("format", "yuv420p")

        print(f"saving new video to {output_path}...")
        # save the new file
        
        
        x=(
            ffmpeg.output(
                vidinput,
                audioinput,
                str(output_path),
                vcodec="h264",
            )
            .global_args("-hide_banner", "-stats")
            .overwrite_output()
        )
        print(ffmpeg.compile(x))
        x.run()
