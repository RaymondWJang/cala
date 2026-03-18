import os

import psutil
from matplotlib import pyplot as plt
from noob import Tube, SynchronousRunner

from cala.nodes.io import stream

VIDEOS = [f"long_recording/{i}.avi" for i in range(30)]


def main():
    process = psutil.Process(os.getpid())
    gen = stream(VIDEOS)
    tube = Tube.from_specification("cala-unraveled", {"cell_size": 10})
    runner = SynchronousRunner(tube)
    ram_use_frame = []
    for idx, arr in enumerate(gen):
        try:
            runner.process(frame=arr, epoch=idx)
            ram_used = process.memory_info().rss / (1024 * 1024)  # in MB
            ram_use_frame.append(round(ram_used, 2))
            if idx % 100 == 0:
                print(f"{idx} frames processed")
        except RuntimeError as e:
            print(e)
            break
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.plot(ram_use_frame)
    plt.xlabel("frame")
    plt.ylabel("memory used (MB)")
    plt.tight_layout()
    plt.savefig("ram_use.png", format="png")
    print(f"{idx} frames complete!")


# def main():
#     from pympler.classtracker import ClassTracker
#
#     tracker = ClassTracker()
#     tracker.track_class(Asset)
#     gen = stream(VIDEOS)
#     tube = Tube.from_specification("cala-unraveled", {"cell_size": 10})
#     runner = SynchronousRunner(tube)
#     for idx, arr in enumerate(gen):
#         try:
#             tracker.create_snapshot()
#             runner.process(frame=arr, epoch=idx)
#             tracker.create_snapshot()
#             tracker.stats.print_summary()
#             next(gen)
#             if idx % 100 == 0:
#                 print(f"{idx} frames processed")
#         except RuntimeError as e:
#             print(e)
#             break
#     fig, ax = plt.subplots(figsize=(12, 5))
#     plt.xlabel("frame")
#     plt.ylabel("memory used (MB)")
#     plt.tight_layout()
#     plt.savefig("ram_use.png", format="png")
#     print(f"{idx} frames complete!")


if __name__ == "__main__":
    main()
