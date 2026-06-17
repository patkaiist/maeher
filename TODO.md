
## One-time PyPI / GitHub setup (OIDC trusted publishing)

The `publish.yml` workflow uses OIDC trusted publishing — no API token is
stored in the repo. This requires configuration on both sides:

- [ ] On PyPI: create a **pending publisher** for the project `maeher`
      (https://pypi.org/manage/account/publishing/), pointing at:
      - Owner/repo: `patkaiist/maeher`
      - Workflow filename: `publish.yml`
      - Environment name: `pypi`
- [ ] On GitHub: create a repository **environment** named `pypi`
      (Settings → Environments). The workflow already requests
      `id-token: write` and sets `environment: pypi`.

## Release

- [ ] Commit everything to `main` and push; confirm the CI workflow
      (`ci.yml`) is green across the Python 3.10–3.13 / OS matrix.
- [ ] Tag the release and push the tag — this triggers `publish.yml`:
      ```bash
      git tag v0.1.0
      git push origin v0.1.0
      ```
- [ ] Watch the Actions run: it builds the sdist + wheels (Linux x86_64/aarch64,
      macOS x86_64/arm64, Windows AMD64) and uploads to PyPI.
- [ ] Verify the install in a clean environment:
      ```bash
      pip install maeher
      python -c "import maeher, numpy as np; print(maeher.track(np.zeros(16000, 'float32'), sample_rate=16000).keys())"
      ```

## Optional / nice-to-have (non-blocking)

- [ ] Add `cp314-*` to `[tool.cibuildwheel] build` once Python 3.14 is widely
      available. Currently 3.14 users fall back to a source build (works, but
      needs CMake + a C++14 compiler on their machine).
- [ ] Consider a `pip install -e .` smoke test or test run inside `publish.yml`
      before upload, so a broken wheel can't ship.
