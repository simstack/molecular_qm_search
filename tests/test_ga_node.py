import asyncio
import importlib
from types import SimpleNamespace

import pytest

from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_search.optimization.models.ga_models import GAConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,function", [
    ("ga", "generate_ga_conformers"),
    ("ga-min", "generate_ga_min_conformers"),
    ("ga-select", "generate_ga_select_conformers"),
])
async def test_node_dispatches_modes_and_returns_molecule_list(monkeypatch, mode, function):
    module = importlib.import_module("molecular_qm_search.optimization.ga")
    received = []
    ranked = MoleculeList()
    evaluator = object()
    def generate(**kwargs):
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        received.append(kwargs)
        return ranked
    monkeypatch.setattr(module, function, generate)
    runner = SimpleNamespace(info=lambda *args: None)
    runner.succeed = lambda: runner
    config = GAConfig(initial_molecule=Molecule(), mode=mode, optimization_method="xtb")
    result = await module.run_ga_conformer_gen.__wrapped__(
        config, node_runner=runner, evaluator=evaluator,
    )
    assert result.result is ranked
    assert received[0]["evaluator"] is evaluator
    assert received[0]["initial_mol"] is config.initial_molecule
    if mode == "ga-select":
        assert received[0]["n_prune"] == config.n_prune


@pytest.mark.asyncio
async def test_node_rejects_unknown_mode():
    module = importlib.import_module("molecular_qm_search.optimization.ga")
    config = GAConfig(initial_molecule=Molecule(), mode="typo")
    with pytest.raises(ValueError, match="Unknown GA mode"):
        await module.run_ga_conformer_gen.__wrapped__(
            config, node_runner=SimpleNamespace(info=lambda *a: None),
        )
