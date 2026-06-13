"""Tests for the CLI entrypoint (__main__).

``main`` is a thin wrapper over the renderer's shared ``run_cli``; ``_render``
dispatches between the test-render and full-export paths. We stub the heavy
collaborators (context creation, track export) so the dispatch logic is covered
without Embree or disk rendering.
"""

import argparse
import types

from openrct2_track_generator import __main__ as cli


def _args(input_path, test=False, skip_render=False):
    return argparse.Namespace(input=input_path, test=test, skip_render=skip_render)


def _patch_common(monkeypatch, calls):
    monkeypatch.setattr(
        cli, "build_tracks", lambda root: [types.SimpleNamespace(units_per_tile=3.3)]
    )
    monkeypatch.setattr(cli, "make_context", lambda lights, upt, test, root: ("ctx", upt, test))
    monkeypatch.setattr(cli, "output_directory_of", lambda root: "out-dir")

    def fake_export_tracks(tracks, ctx, out, skip_render):
        calls["export"] = {"n": len(tracks), "ctx": ctx, "out": out, "skip_render": skip_render}

    def fake_export_track_test(track, ctx):
        calls["export_test"] = {"ctx": ctx}

    monkeypatch.setattr(cli, "export_tracks", fake_export_tracks)
    monkeypatch.setattr(cli, "export_track_test", fake_export_track_test)


def test_render_full_export_path(monkeypatch):
    calls = {}
    _patch_common(monkeypatch, calls)
    cli._render(_args("track.yaml", skip_render=True), {}, [])
    assert "export_test" not in calls
    assert calls["export"]["n"] == 1
    assert calls["export"]["out"] == "out-dir"
    assert calls["export"]["skip_render"] is True
    assert calls["export"]["ctx"] == ("ctx", 3.3, False)


def test_render_test_path(monkeypatch):
    calls = {}
    _patch_common(monkeypatch, calls)
    cli._render(_args("track.yaml", test=True), {}, [])
    assert "export" not in calls
    assert calls["export_test"]["ctx"] == ("ctx", 3.3, True)


def test_main_delegates_to_run_cli(monkeypatch):
    captured = {}

    def fake_run_cli(prog, argv, render):
        captured["prog"] = prog
        captured["argv"] = argv
        captured["render"] = render
        return 0

    monkeypatch.setattr(cli, "run_cli", fake_run_cli)
    rc = cli.main(["track.yaml"])
    assert rc == 0
    assert captured["prog"] == "openrct2-track-generator"
    assert captured["argv"] == ["track.yaml"]
    assert captured["render"] is cli._render


def test_main_returns_run_cli_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "run_cli", lambda prog, argv, render: 1)
    assert cli.main([]) == 1


def test_dunder_main_guard_invokes_sys_exit(monkeypatch):
    # Cover the `sys.exit(main())` body inside `if __name__ == "__main__":`.
    import runpy
    import sys

    exits = []
    monkeypatch.setattr(sys, "exit", exits.append)

    import openrct2_object_common.cli as _cli
    monkeypatch.setattr(_cli, "run_cli", lambda prog, argv, render: 7)

    sys.modules.pop("openrct2_track_generator.__main__", None)
    runpy.run_module("openrct2_track_generator", run_name="__main__")

    assert exits == [7]
