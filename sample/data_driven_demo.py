"""Run the standalone NumPy reference example from any working directory."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    example = (Path(__file__).resolve().parents[1] / "external_libraries" /
               "MCAP_python_adaptive_control" / "sample" / "data_driven" /
               "pi_tuning.py")
    runpy.run_path(str(example), run_name="__main__")
