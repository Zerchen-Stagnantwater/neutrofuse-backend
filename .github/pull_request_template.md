## What this changes

## Why

## Checklist

- [ ] `pytest tests/ pipeline_tests/ -m "not network"` passes (215 tests)
- [ ] Any change to fusion logic includes updated ablation results if the headline metrics shift
- [ ] `requirements.txt` updated if new dependencies added
- [ ] Docker build verified locally if touching `Dockerfile` or `requirements.txt`
