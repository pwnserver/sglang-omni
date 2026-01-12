# SGLang-Omni

## Get Started

```bash
# create a virtual environment
uv venv .venv -p 3.11
source .venv/bin/activate

# install the package
uv pip install -v -e .
```

### Run Demo

You can execute the folloing commands to run the demos.

```bash
# Use SHMRelay (default)
python examples/run_two_stage_demo.py

# Use NIXLRelay
python examples/run_two_stage_demo.py --relay nixl


# Use SHMRelay (default)
python run_three_stage_demo.py

# Use NIXLRelay
python run_three_stage_demo.py --relay nixl
```
