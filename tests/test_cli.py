import json

import pytest

from hyperopt_kit.cli import build_parser, main

SPACE = json.dumps({"a": [0.1, 5.0], "b": [0.1, 5.0]})


def test_build_parser_has_subcommands():
    parser = build_parser()
    sub = {action.dest: action for action in parser._actions}
    assert "search" in parser._subparsers._group_actions[0].choices


def test_search_random_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "random", "--seed", "1"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: random" in out
    assert "trials: 10" in out
    assert "best score:" in out
    assert "best parameters:" in out


def test_search_bayesian_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "12",
         "--strategy", "bayesian", "--seed", "2"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: bayesian" in out


def test_search_grid_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "grid"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: grid" in out


def test_search_deterministic_with_same_seed(capsys):
    main(["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
          "--strategy", "bayesian", "--seed", "5"])
    first = capsys.readouterr().out
    main(["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
          "--strategy", "bayesian", "--seed", "5"])
    second = capsys.readouterr().out
    assert first == second


def test_search_floor_stops_early(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "30",
         "--strategy", "random", "--seed", "1", "--floor", "1000.0"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "trials: 1" in out


def test_search_patience_config_accepted(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "30",
         "--strategy", "random", "--seed", "1", "--patience", "5"]
    )
    assert rc == 0


def test_compare_subcommand_prints_table(capsys):
    rc = main(
        ["compare", "--space", SPACE, "--objective", "demo", "--budget", "8", "--seed", "3"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| strategy |" in out
    assert "| grid |" in out
    assert "| random |" in out
    assert "| bayesian |" in out
    assert "| tpe |" in out
    assert "| cmaes |" in out
    assert "| hyperband |" in out
    assert "| bohb |" in out


def test_search_hyperband_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "hyperband", "--seed", "2"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: hyperband" in out
    assert "trials: 10" in out


def test_search_random_successive_halving_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "random_successive_halving", "--seed", "2", "--eta", "3",
         "--max-resource", "9"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: random_successive_halving" in out
    assert "trials: 10" in out


def test_compare_random_successive_halving_only(capsys):
    rc = main(
        ["compare", "--space", SPACE, "--objective", "demo", "--budget", "8",
         "--seed", "3", "--strategies", "random,random_successive_halving",
         "--eta", "2", "--max-resource", "8"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| random |" in out
    assert "| random_successive_halving |" in out
    assert "| grid |" not in out


def test_search_successive_halving_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "successive_halving", "--seed", "2", "--eta", "3",
         "--max-resource", "9"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: successive_halving" in out
    assert "trials: 10" in out


def test_search_tpe_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "tpe", "--seed", "2"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: tpe" in out
    assert "trials: 10" in out


def test_search_bohb_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "bohb", "--seed", "2", "--top-n-percent", "15",
         "--random-fraction", "0.3", "--n-candidates", "8"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: bohb" in out
    assert "trials: 10" in out


def test_search_pbt_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "8",
         "--strategy", "pbt", "--seed", "2", "--population-size", "4",
         "--exploit-interval", "1", "--eta", "3", "--max-resource", "9"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: pbt" in out
    assert "trials: 8" in out


def test_compare_pbt_only(capsys):
    rc = main(
        ["compare", "--space", SPACE, "--objective", "demo", "--budget", "8",
         "--seed", "3", "--strategies", "random,pbt", "--exploit-interval", "1"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| random |" in out
    assert "| pbt |" in out
    assert "| grid |" not in out


def test_search_cmaes_subcommand(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "cmaes", "--seed", "2"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: cmaes" in out
    assert "trials: 10" in out


def test_search_cmaes_sigma0_accepted(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "cmaes", "--seed", "2", "--sigma0", "0.2",
         "--population-size", "6"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: cmaes" in out


def test_compare_with_cmaes_only(capsys):
    rc = main(
        ["compare", "--space", SPACE, "--objective", "demo", "--budget", "8",
         "--seed", "3", "--strategies", "random,cmaes"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| random |" in out
    assert "| cmaes |" in out
    assert "| grid |" not in out


def test_search_tpe_gamma_accepted(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "demo", "--budget", "10",
         "--strategy", "tpe", "--seed", "2", "--gamma", "0.15", "--n-candidates", "12"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "strategy: tpe" in out


def test_compare_with_tpe_only(capsys):
    rc = main(
        ["compare", "--space", SPACE, "--objective", "demo", "--budget", "8",
         "--seed", "3", "--strategies", "random,tpe"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| random |" in out
    assert "| tpe |" in out
    assert "| grid |" not in out


def test_compare_with_hyperband_only(capsys):
    rc = main(
        ["compare", "--space", SPACE, "--objective", "demo", "--budget", "8",
         "--seed", "3", "--strategies", "random,hyperband"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| random |" in out
    assert "| hyperband |" in out
    assert "| grid |" not in out


def test_compare_with_custom_strategies(capsys):
    rc = main(
        ["compare", "--space", SPACE, "--objective", "demo", "--budget", "8",
         "--seed", "3", "--strategies", "random,bayesian"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| random |" in out
    assert "| bayesian |" in out
    assert "| grid |" not in out


def test_report_subcommand_writes_file(tmp_path, capsys):
    out = tmp_path / "nested" / "report.md"
    rc = main(
        ["report", "--space", SPACE, "--objective", "demo", "--budget", "8",
         "--seed", "4", "--out", str(out)]
    )
    captured = capsys.readouterr().out
    assert rc == 0
    assert "wrote report to" in captured
    text = out.read_text(encoding="utf-8")
    assert "# " in text
    assert "## Results" in text
    assert "## Learning curves" in text
    assert "## Parameter importance" in text


def test_report_default_output_file(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    rc = main(
        ["report", "--space", SPACE, "--objective", "demo", "--budget", "6", "--seed", "1"]
    )
    assert rc == 0
    assert (tmp_path / "hyperopt_report.md").exists()


def test_objective_from_dotted_path(capsys):
    rc = main(
        ["search", "--space", SPACE, "--objective", "hyperopt_kit.objectives:demo_objective",
         "--budget", "6", "--strategy", "random", "--seed", "1"]
    )
    assert rc == 0


def test_space_from_json_file(tmp_path, capsys):
    space_file = tmp_path / "space.json"
    space_file.write_text(SPACE, encoding="utf-8")
    rc = main(
        ["search", "--space", f"@{space_file}", "--objective", "demo", "--budget", "6",
         "--strategy", "random", "--seed", "1"]
    )
    assert rc == 0


def test_no_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_strategy_exits_nonzero():
    with pytest.raises(SystemExit):
        main(
            ["search", "--space", SPACE, "--objective", "demo", "--budget", "5",
             "--strategy", "grid-search"]
        )


def test_bad_objective_exits_nonzero():
    with pytest.raises(SystemExit):
        main(
            ["search", "--space", SPACE, "--objective", "no.such.module:fn",
             "--budget", "5"]
        )


def test_invalid_space_exits_nonzero():
    with pytest.raises(SystemExit):
        main(
            ["search", "--space", "{not json", "--objective", "demo", "--budget", "5"]
        )


def test_budget_zero_rejected():
    with pytest.raises(SystemExit):
        main(
            ["search", "--space", SPACE, "--objective", "demo", "--budget", "0"]
        )


def test_empty_strategies_rejected():
    with pytest.raises(SystemExit):
        main(
            ["compare", "--space", SPACE, "--objective", "demo", "--budget", "5",
             "--strategies", ","]
        )
