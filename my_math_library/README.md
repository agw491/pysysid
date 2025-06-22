# My Math Library

A high-performance math library implemented in C++ with Python bindings using pybind11, managed with UV.

## Features

- **Fast C++ Implementation**: Core mathematical operations implemented in optimized C++
- **Python Integration**: Seamless Python interface using pybind11
- **Modern Tooling**: Managed with UV for fast dependency resolution and virtual environments
- **Type Safety**: Full type hints and mypy support
- **Comprehensive Testing**: Complete test suite with pytest

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd my_math_library

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
```

## Usage

```python
from my_math_library import add, Matrix, mean

# Basic operations
result = add(5.0, 3.0)  # 8.0

# Statistical functions
data = [1.0, 2.0, 3.0, 4.0, 5.0]
avg = mean(data)  # 3.0

# Matrix operations
m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
result = m1 * m2
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Building

The C++ extensions are automatically built when you install the package. If you need to rebuild:

```bash
uv pip install -e . --force-reinstall --no-deps
```

## Requirements

- Python 3.8+
- C++14 compatible compiler
- CMake (for building)
- UV package manager

## License

MIT License
```

## Setup Instructions

1. **Create the project directory:**
```bash
mkdir my_math_library
cd my_math_library
```

2. **Initialize UV project:**
```bash
uv init
```

3. **Create the directory structure and files as shown above**

4. **Install dependencies:**
```bash
uv add --dev pytest pytest-cov black isort mypy
uv add pybind11
```

5. **Build and install:**
```bash
uv pip install -e .
```

6. **Run tests:**
```bash
uv run pytest
```

7. **Run the example:**
```bash
uv run python examples/usage_example.py
```

This project demonstrates:
- C++ implementation with header/source separation
- pybind11 bindings for Python integration
- UV project management with proper dependencies
- Type hints and documentation
- Comprehensive testing
- Modern Python packaging with pyproject.toml
- Development tools integration (black, isort, mypy)
