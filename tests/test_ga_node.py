import asyncio
import importlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_search.optimization.models.ga_models import GAConfig, GAOptimizationMethod


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
    config = GAConfig(initial_molecule=Molecule(), mode=mode, optimization_method="XTB")
    result = await module.run_ga_conformer_gen.__wrapped__(
        config, node_runner=runner, evaluator=evaluator,
    )
    assert result.result is ranked
    assert received[0]["evaluator"] is evaluator
    assert received[0]["initial_mol"] is config.initial_molecule
    assert received[0]["parallel_children"] == config.parallel_children
    assert received[0]["match_double_bonds"] is True
    assert received[0]["rotatable_bond_min"] == -180.0
    assert received[0]["rotatable_bond_max"] == 180.0
    assert "forcefield" not in received[0]
    assert "threads" not in received[0]
    if mode == "ga-select":
        assert received[0]["n_prune"] == config.n_prune


def test_config_uses_method_enum_and_exposes_rotatable_bond_options():
    config = GAConfig(initial_molecule=Molecule())
    assert config.optimization_method is GAOptimizationMethod.RDKIT_MMFF
    assert config.parallel_children == 0
    assert config.match_double_bonds is True
    assert set(GAOptimizationMethod) == {
        GAOptimizationMethod.RDKIT_MMFF,
        GAOptimizationMethod.RDKIT_MMFF94S,
        GAOptimizationMethod.RDKIT_UFF,
        GAOptimizationMethod.XTB,
        GAOptimizationMethod.DFTB,
    }
    assert "forcefield" not in GAConfig.model_fields
    assert "threads" not in GAConfig.model_fields
    schema = GAConfig.model_json_schema()
    assert schema["$defs"]["GAOptimizationMethod"]["enum"] == [
        "RDKIT/mmff", "RDKIT/mmff94s", "RDKIT/uff", "XTB", "DFTB",
    ]
    assert schema["properties"]["parallel_children"]["title"] == "Parallel children"
    assert "match_double_bonds" in schema["properties"]
    assert "rotatable_bond_min" in schema["properties"]
    assert "rotatable_bond_max" in schema["properties"]
    with pytest.raises(ValidationError):
        GAConfig(initial_molecule=Molecule(), optimization_method="mmff")


@pytest.mark.asyncio
async def test_node_rejects_unknown_mode():
    module = importlib.import_module("molecular_qm_search.optimization.ga")
    config = GAConfig(initial_molecule=Molecule(), mode="typo")
    with pytest.raises(ValueError, match="Unknown GA mode"):
        await module.run_ga_conformer_gen.__wrapped__(
            config, node_runner=SimpleNamespace(info=lambda *a: None),
        )
