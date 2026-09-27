# Releasing ppc-forecaster

Wheels for Linux (glibc + musl, x86_64 + aarch64), macOS (Intel + Apple Silicon) and Windows x64 are
built and tested by `.github/workflows/CI.yml` on every push. Pushing a `v*` tag also publishes to PyPI.

## One-time setup

1. Create a GitHub repository (e.g. `ppc-forecaster`) and push this directory to it.
2. On GitHub: Settings → Environments → New environment → name it `pypi`. Optionally require your approval
   before each release.
3. On pypi.org (log in or create an account) go to Your account → Publishing → "Add a new pending publisher":
   - PyPI project name: `ppc-forecaster`
   - Owner: your GitHub user name, Repository: the repo name
   - Workflow name: `CI.yml`, Environment name: `pypi`

   No API token is needed. PyPI trusts this workflow directly ("Trusted Publishing").
4. Optional dry run: do the same on test.pypi.org, and temporarily change the publish step to
   `uv publish --trusted-publishing always --publish-url https://test.pypi.org/legacy/ 'wheels-*/*'`.

## Each release

1. Bump the version in **both** `pyproject.toml` and `ppc_rs/Cargo.toml`.
2. Push to `main` and wait for CI to go green on every platform.
3. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`
4. The `release` job uploads all wheels plus the sdist to PyPI. Then check: `pip install ppc-forecaster`.

## Local builds (for testing only)

```bash
maturin build --release                               # this machine
maturin build --release --target x86_64-apple-darwin  # Intel Mac from Apple Silicon
maturin build --release --target x86_64-unknown-linux-gnu --zig --compatibility manylinux2014   # needs zig
maturin sdist
```
