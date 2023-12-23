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
    audio_streams = []
    for stream in filedata["streams"]:
        if stream["codec_type"] == "video":
            video_stream_data = stream
        else:
            audio_streams.append(stream)
    assert video_stream_data is not None
    numerator, denominator = video_stream_data["avg_frame_rate"].split("/")
    fps: float = float(numerator) / float(denominator)
    single_frame_duration = 1 / fps

    fi = ffmpeg.input(file)
    from pprint import pprint

    print(fi.video)
    pprint(audio_streams)
    all_audio = []
    samplerate = 44100
    for i, stream in enumerate(audio_streams):
        filepath = audiopath.with_stem(f"audio_{i}")
        print(f"separating audio stream {i} to {filepath}")
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
        all_audio.append(audio_chunks)

    print(f"saving video frames to {tmpdir}...")
    fi.video.output(str(imagepaths), loglevel="error").global_args("-hide_banner", "-stats").run()
    images = sorted(tmpdir.glob("*.png"))
    video_chunks = (images[i : i + split_size_frames] for i in range(0, len(images), split_size_frames))

    # bind video and audio chunks, shuffle them, then split again
    v_chunks_, *a_chunks = zip(
        *sorted(
            zip(
                video_chunks,
                *all_audio,
            ),
            key=lambda _: random.random(),
        )
    )

    v_chunks = (y for x in v_chunks_ for y in x)  # [[a,b],[c,d]] -> [a,b,c,d]

    shuffled_audios: list[np.ndarray] = [
        np.vstack(a_chunk) if len(a_chunk[0].shape) != 1 else np.hstack(a_chunk) for a_chunk in a_chunks
    ]

    audiopaths = []
    for i, stream in enumerate(shuffled_audios):
        filepath = audiopath.with_stem(f"audio_{i}")
        print(f"separating audio stream {i} to {filepath}")
        wavwrite(filepath, samplerate, stream)
        audiopaths.append(filepath)

    pprint(shuffled_audios)

    # create new folder of symlinks with the files in order
    with tempfile.TemporaryDirectory(suffix="video-mixing-test-links") as linkdir:
        print(f"symlinking ordered images to {linkdir}...")
        linkdir = Path(linkdir)
        for filename, origin in zip([i.relative_to(tmpdir) for i in images], v_chunks):
            (linkdir / filename).symlink_to(origin)

        vidinput = ffmpeg.input(linkdir / "%06d.png", r=fps).video.filter("format", "yuv420p")

        print(f"saving new video to {output_path}...")
        # save the new file

        x = (
            ffmpeg.output(
                vidinput,
                *[ffmpeg.input(pth).audio for pth in audiopaths],
                str(output_path),
            )
            .global_args("-hide_banner", "-stats")
            .overwrite_output()
        )
        print(ffmpeg.compile(x))
        x.run()

    exit()

