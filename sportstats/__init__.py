from __future__ import annotations


def create_app(*args, **kwargs):
    from sportstats.web import create_app as _create_app

    return _create_app(*args, **kwargs)
