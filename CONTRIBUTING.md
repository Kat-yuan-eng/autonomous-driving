# Contributing to Autonomous Driving

Thank you for your interest in contributing to the autonomous driving project!

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd autonomous driving

# Install development dependencies
pip install -r requirements/dev.txt

# Run tests
bash runtests.sh
```

## Code Style

- Follow PEP 8 (enforced by ruff)
- Use type hints where possible (checked by mypy)
- Maximum line length: 88 characters
- Use numpy docstring convention

## Testing

- All new features must include tests
- Run `pytest -l -Werror` before submitting a PR
- Ensure all tests pass on Python 3.10+

## Pull Request Process

1. Update the CHANGELOG.md with details of changes
2. Ensure all tests pass
3. Update documentation if needed
4. Request review from maintainers
