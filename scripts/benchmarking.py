from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from noob import SynchronousRunner, Tube

from cala.nodes.io import stream
from cala.nodes.prep import (
    Anchor,
    blur,
    butter,
    package_frame,
    remove_mean,
    GlowRemover,
)
from cala.nodes.prep.motion import _prepare
from cala.testing.util import total_gradient_magnitude

sns.set_style("whitegrid")
font = {"family": "normal", "weight": "regular", "size": 15}

matplotlib.rc("font", **font)


VIDEOS = [f"long_recording/{i}.avi" for i in range(20)]  # minian/msCam{i}


def preprocess(arr, idx):
    frame = package_frame(arr, idx)
    frame = blur(frame, method="median", kwargs={"ksize": 3})
    frame = butter(frame, {})
    return remove_mean(frame, orient="both")


def test_write_raw_movie():
    gen = stream(VIDEOS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("120fps.avi", fourcc, 120.0, (600, 600))

    for arr in gen:
        frame_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()


def test_write_stable_movie():
    """
    For testing how well motion correction performs with real movie
    See for yourself
    """
    gen = stream(VIDEOS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("motion_test.avi", fourcc, 60.0, (752, 960))

    stab = Anchor()

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        combined = np.concat([frame.array.values, matched.array.values], axis=0)

        frame_bgr = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()


def test_write_signal_movie():
    """
    For testing how well the entire preprocessing line performs with real movie
    See for yourself
    """
    gen = stream(VIDEOS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("signal_test.avi", fourcc, 60.0, (752, 960))

    stab = Anchor()
    bgrm = GlowRemover()

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        signal = bgrm.process(matched)
        combined = np.concat([matched.array.values, signal.array.values], axis=0)

        frame_bgr = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()


def test_write_all_movies():
    gen = stream(VIDEOS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("4x_long2.mp4", fourcc, 30.0, (2000, 350))
    stab = Anchor()
    bgrm = GlowRemover()
    max_signal = 0
    for idx, arr in enumerate(gen):
        if idx % 4 == 0:
            frame = package_frame(arr[200:550, 50:-50], idx)
            blurred = blur(frame, method="median", kwargs={"ksize": 3})
            flat = butter(blurred, {})
            delined = remove_mean(flat, orient="both")
            matched = stab.stabilize(delined)
            denoised = blur(matched, method="gaussian", kwargs={"ksize": (9, 9), "sigmaX": 0})
            signal = bgrm.process(denoised)
            # top = np.concat([frame.array.values, flat.array.values], axis=1)
            # bottom = np.concat([matched.array.values, signal.array.values], axis=1)
            # combined = np.concatenate((top, bottom), axis=0)
            combined = np.concat(
                [
                    frame.array.values,
                    flat.array.values,
                    matched.array.values,
                    signal.array.values * 3,
                ],
                axis=1,
            )
            max_signal = max(np.max(signal.array.values), max_signal)
            frame_bgr = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            out.write(frame_bgr)

    out.release()
    print(f"{max_signal = }")


def test_motion_crisp_pics():
    """
    quantify how crisp is a summary image before and after registration.
    This can be done by computing the norm of the gradient field of the image at all pixels.
    Intuitively, a dataset with non-registered motion will have a blurred mean image,
    resulting in a lower value for the total gradient field norm.

    Currently not working properly because of the remaining optical artifacts. (vertical bands)
    These artifacts move with camera, which becomes clearer if the image is NOT motion-corrected.

    References:
        [1] NormCorre, https://www.sciencedirect.com/science/article/pii/S0165027017302753
    """

    gen = stream(VIDEOS)

    stab = Anchor()
    raws = []
    stabs = []

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        raws.append(frame.array.values)
        stabs.append(matched.array.values)

    raw = np.stack(raws)
    stab = np.stack(stabs)

    raw_mean = np.mean(raw, axis=0)
    stab_mean = np.mean(stab, axis=0)

    crisp_raw = total_gradient_magnitude(raw_mean)
    crisp_stab = total_gradient_magnitude(stab_mean)

    print(f"{crisp_raw = }, {crisp_stab = }")

    mean = np.concatenate((raw_mean, stab_mean), axis=0)
    plt.imsave("motion_crisp_pics.png", mean, cmap="gray")


def test_motion_mean_corr():
    """
    To evaluate the results of the motion correction algorithm across the different frames,
    we use a metric that is based on the similarity (pixel-wise, Pearson's correlation coefficient r)
    between the mean image across time and each individual frame.
    Intuitively, an increase in the correlation coefficient for a given frame
    indicates a better alignment with the mean.

    Similar metrics have been used before for assessing the quality of registration algorithms
    in the context of fluorescence microscopy (Lee et al., 2014).
    To account for border effects during registration,
    a number of pixels around each boundary (e.g., equal to the maximum shift in each direction over time)
    is removed prior to computing the correlation coefficients.

    References:
        [1] NormCorre, https://www.sciencedirect.com/science/article/pii/S0165027017302753
    """
    figure_dir = Path("figures/motion")

    gen = stream(VIDEOS)

    stab = Anchor()
    raws = []
    stabs = []

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        if idx == 500:
            curr = _prepare(frame.array, stab.dog_kwargs, stab.gauss_kwargs).values
            cv2.imwrite(
                figure_dir / "anchors.png",
                np.concatenate([curr, stab._local.values, stab._global.values], axis=1),
            )
            cv2.imwrite(figure_dir / "curr.png", curr)
            cv2.imwrite(figure_dir / "local_anchor.png", stab._local.values)
            cv2.imwrite(figure_dir / "global_anchor.png", stab._global.values)
        raws.append(frame.array.values)
        stabs.append(matched.array.values)

    raw = np.stack(raws)
    stab = np.stack(stabs)

    raw_mean = np.mean(raw[:, 20:-20, 20:-20], axis=0)
    stab_mean = np.mean(stab[:, 20:-20, 20:-20], axis=0)

    raw_cms = [np.corrcoef(r[20:-20, 20:-20].flatten(), raw_mean.flatten())[0, 1] for r in raws]
    stab_cms = [np.corrcoef(s[20:-20, 20:-20].flatten(), stab_mean.flatten())[0, 1] for s in stabs]

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.plot(raw_cms)
    plt.plot(stab_cms)
    plt.legend(["raw", "stabilized"], loc="upper right")
    plt.xlabel("Frame Index")
    plt.ylabel("Correlation")
    plt.tight_layout()

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / "mean_corr.png")

    assert False


def test_whole_pipeline():
    """
    Run the entire pipeline with Minian data
    """
    tube = Tube.from_specification("with-minian")
    runner = SynchronousRunner(tube=tube)
    processed_vid = runner.run()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("motion_test.avi", fourcc, 20.0, (100, 100))

    for arr in processed_vid:
        frame_bgr = cv2.cvtColor(arr.array.values.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    out.release()


def test_speed_per_frame():
    """
    Test how long it takes for the entire cala to process each frame.
    Export a plot showing time taken.
    """
    gen = stream(VIDEOS)
    tube = Tube.from_specification("cala-unraveled", {"cell_size": 10})
    runner = SynchronousRunner(tube)
    frame_speed = []

    for idx, arr in enumerate(gen):
        try:
            start = datetime.now()
            runner.process(frame=arr, epoch=idx)
            duration = datetime.now() - start
            frame_speed.append(round(duration.total_seconds(), 2))
            if idx % 100 == 0:
                print(f"{idx} frames processed")
            if idx == 100:
                break
        except RuntimeError as e:
            print(e)
            break
    fig, ax = plt.subplots(figsize=(20, 4))
    sns.histplot(np.array(frame_speed) / 2)
    plt.ylabel("percent", fontsize=20)
    plt.xlabel("time taken (s)", fontsize=20)
    plt.tight_layout()
    plt.savefig("frame_speed.png")


def test_deglow_write_image():
    """
    Write a deglowed frame by accumulating the frames that are preprocessed until motion correction,
    grab the last frame and subtract the global minimum pixel value through time.
    """
    gen = stream(VIDEOS[:3])

    stab = Anchor()
    stabs = []

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        stabs.append(matched.array.values)

    deglowed = stabs[-1] - np.min(stabs, axis=0)
    plt.imsave("deglowed.png", deglowed, cmap="gray")


def test_preprocessing_write_image():
    """
    A temporary function for hotswapping preprocessing steps
    and plot results / diffs fast.
    """
    images = {}
    gen = stream(VIDEOS)

    stab = Anchor()
    eraser = GlowRemover()
    raws = []
    stabs = []

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        signal = eraser.process(matched)
        if idx == 500:
            break
    arr = next(gen)
    frame = preprocess(arr, idx)
    matched = stab.stabilize(frame)
    signal = eraser.process(matched)

    frame = package_frame(arr, 0)
    hot_pixel = blur(frame, method="median", kwargs={"ksize": 3})
    buttered = butter(hot_pixel, {})
    delined = remove_mean(buttered, "both")
    images["pre"] = buttered.array.values
    images["post"] = delined.array.values
    images["diff"] = buttered.array.values - delined.array.values
    # crop = slice(None, int(arr.shape[0] / 3)), slice(int(arr.shape[1] * 3 / 4), None)
    # for k, v in images.items():
    #     images[k] = circle_points(
    #         np.clip(v, 0, None).astype(np.uint8),
    #         zip(*np.where(diff > 50)[::-1]),
    #         radius=20,
    #         color=(0, 0, 255),
    #         thickness=2,
    #     )[crop]
    images["diff"] = images["diff"] / images["diff"].max() * 255
    all = np.concatenate([image for image in images.values()], axis=1)
    figure_dir = Path("figures/deline")
    figure_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(figure_dir / "all.png", all)
    for k in images:
        cv2.imwrite(figure_dir / f"{k}.png", images[k])


def test_recursive():
    gen = stream(VIDEOS[:3])
    tube = Tube.from_specification("cala-demo", {"cell_size": 8})
    runner = SynchronousRunner(tube)
    for idx, arr in enumerate(gen):
        res = runner.process(frame=arr, index=idx)


if __name__ == "__main__":
    test_write_all_movies()
