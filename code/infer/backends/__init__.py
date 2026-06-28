"""Per-backend inference callers.

Each backend module exposes a `make_backend(...)` factory returning a
`Backend` instance (see `base.Backend`). The unified runner in
`infer/run_infer.py` dispatches on `--backend`.
"""
