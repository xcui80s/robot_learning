# Agent Guidelines

This document provides guidance for agentic coding agents working in this repository.

## Build Commands

- **Install dependencies**: `pip install -r requirements.txt` (Python) or `npm install` (Node.js)
- **Build**: Check for `Makefile`, `setup.py`, or `pyproject.toml` for project-specific commands
- **Lint**: `ruff check .` or `flake8` (Python), `eslint` (JavaScript/TypeScript)
- **Format**: `black .` or `ruff format .` (Python), `prettier --write .` (JavaScript/TypeScript)
- **Type check**: `mypy` (Python), `tsc --noEmit` (TypeScript)

## Testing Commands

- **Run all tests**: `pytest` (Python) or `npm test` (Node.js)
- **Run single test file**: `pytest path/to/test_file.py`
- **Run single test**: `pytest path/to/test_file.py::test_function_name`
- **Run with coverage**: `pytest --cov=. --cov-report=term-missing`
- **Run specific test class**: `pytest path/to/test_file.py::TestClassName`
- **Verbose mode**: `pytest -v` or `pytest -vv` for more detail

## Code Style Guidelines

### Python

- Use **type hints** for all function parameters and return values
- Follow **PEP 8** naming conventions:
  - `snake_case` for functions, variables, modules
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
- Keep functions focused and under 50 lines when possible
- Use **docstrings** for all public functions, classes, and modules
- Prefer **f-strings** over string concatenation
- Handle exceptions explicitly; never use bare `except:`

### JavaScript/TypeScript

- Use **const** by default; **let** when reassignment needed
- Prefer **async/await** over callbacks
- Use **TypeScript** strict mode
- Follow naming conventions:
  - `camelCase` for variables, functions
  - `PascalCase` for classes, interfaces, types
  - `SCREAMING_SNAKE_CASE` for constants

### General

- Keep line length under 100 characters
- Use 2 or 4 spaces for indentation (check existing files)
- Remove unused imports and variables
- Add trailing newlines to files

## Import Organization

### Python
1. Standard library imports
2. Third-party library imports (alphabetical)
3. Local application imports (alphabetical)

Separate each group with a blank line:

```python
import os
import sys
from typing import List

import numpy as np
import pandas as pd

from mymodule import myfunction
```

### JavaScript/TypeScript
1. Built-in modules
2. Third-party packages (alphabetical)
3. Local imports (alphabetical, by path depth)

## Error Handling

- Always catch specific exceptions, not generic `Exception`
- Use context managers (`with` statements) for resource management
- Log errors with appropriate severity levels
- Include meaningful error messages
- Never swallow exceptions silently

## Testing

- Write tests for all new functionality
- Use descriptive test names: `test_function_name_does_x_when_y`
- Follow AAA pattern: Arrange, Act, Assert
- Mock external dependencies
- Aim for >80% code coverage

## Git

- **NEVER** commit secrets, API keys, or credentials
- **NEVER** commit `.env` files
- Make atomic commits (one logical change per commit)
- Write clear commit messages following conventional commits format
- Only commit when explicitly requested by user

## Before Completing Tasks

1. Run linting tools (`ruff`, `flake8`, `eslint`)
2. Run type checking (`mypy`, `tsc`)
3. Run tests and ensure they pass
4. Remove any debug code or print statements
5. Check for unused imports or variables

## File Organization

- Group related functionality in modules/packages
- Keep configuration at root level
- Place tests in `tests/` directory, mirroring source structure
- Use `__init__.py` for Python packages
- Separate concerns: models, views, controllers, utils

## Performance

- Avoid premature optimization
- Profile before optimizing
- Use appropriate data structures
- Consider memory usage for large datasets
- Cache expensive computations when appropriate

## Security

- Validate all inputs
- Sanitize user-provided data
- Use parameterized queries (prevent SQL injection)
- Never expose sensitive data in logs or errors
- Follow OWASP guidelines for web applications
