# Test Suite for SeasonsGAN

This directory contains unit and integration tests for the CycleGAN implementation.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_generator.py        # Generator architecture tests
├── test_discriminator.py    # Discriminator architecture tests
└── test_integration.py      # End-to-end pipeline tests
```

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_generator.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run individual test:
```bash
pytest tests/test_generator.py::test_generator_output_shape
```

## Test Coverage

### Generator Tests (`test_generator.py`)
- ✅ **Output Shape**: Verifies 256x256 input → 256x256 output
- ✅ **No NaNs**: Checks for numerical stability
- ✅ **Batch Size**: Tests flexibility with different batch sizes
- ✅ **Output Range**: Verifies Tanh output is in [-1, 1]
- ✅ **Deterministic**: Ensures reproducibility in eval mode

### Discriminator Tests (`test_discriminator.py`)
- ✅ **Output Shape**: Verifies PatchGAN output dimensions
- ✅ **Output Range**: Checks Sigmoid output is in [0, 1]
- ✅ **Batch Size**: Tests with multiple batch sizes
- ✅ **No NaNs**: Validates numerical stability

### Integration Tests (`test_integration.py`)
- ✅ **CycleGAN Forward Pass**: Tests complete Summer ↔ Winter cycle
- ✅ **Cycle Consistency**: Verifies reconstruction pipeline
- ✅ **Parameter Count**: Validates model architecture hasn't changed

## Requirements

Install test dependencies:
```bash
pip install pytest torch torchvision
```

## Expected Output

When all tests pass, you should see:
```
tests/test_generator.py ✓✓✓✓✓         5 passed
tests/test_discriminator.py ✓✓✓✓      4 passed
tests/test_integration.py ✓✓✓         3 passed

============ 12 passed in 2.34s ============
```

## CI/CD Integration

These tests can be integrated into GitHub Actions:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/
```

## Notes

- Tests use **CPU-only** PyTorch (no GPU required)
- Tests are **deterministic** (use fixed random seeds)
- Tests are **fast** (~2-3 seconds total)
- Tests validate **architecture correctness**, not training quality
