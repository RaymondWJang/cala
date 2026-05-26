from pathlib import Path


def add_noob_sources() -> list[Path]:
    """
    Expose packages tubes to noob via its ``noob.add_sources`` entrypoint.
    """
    return [Path(__file__).parent / "data"]
