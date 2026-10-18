import inspect
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from simstack.util.docstring_parser import DocstringParser

from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_models import Atom, Molecule, MoleculeList
from molecular_qm_psi4.models.crest_input import CrestLevelOfTheoryEnum
from molecular_qm_search.optimization.ga_relax import (
    GARelaxInput,
    GARelaxOptimizer,
    XTBOptimizerParameters,
    ga_relax,
)
from molecular_qm_search.optimization.models.ga_models import GAConfig, GAMode


def _carbon():
    return Molecule(atoms=[Atom(element="C", x=0.0, y=0.0, z=0.0)])


def _ga():
    return GAConfig(initial_molecule=_carbon(), num_confs=3, generations=2)


def _dftb(**overrides):
    values = {"optimization": True, "max_optimization_steps": 1000}
    values.update(overrides)
    return DftbInput(**values)


def _runner():
    warnings = []
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: warnings.append(args),
        warnings=warnings,
        succeed=lambda *args, **kwargs: SimpleNamespace(status="completed"),
    )


def test_node_name_and_called_nodes():
    assert ga_relax._node_name == "ga-relax"
    parsed = DocstringParser(inspect.getdoc(ga_relax))
    assert parsed.called_nodes() == [
        "run_ga_conformer_gen",
        "dftb_list_calculator",
        "xtb_optimize_molecule_list",
    ]
    results = parsed.simstack_results()
    assert results["molecules"]["type"] == "MoleculeList"


def test_schema_exposes_only_the_selected_optimizer():
    schema = GARelaxInput.json_schema()
    assert "ga_config" in schema["properties"]
    assert "initial_molecule" in schema["properties"]["ga_config"]["properties"]
    assert "xtb_parameters" not in schema["properties"]
    assert "dftb_parameters" not in schema["properties"]
    assert schema["properties"]["optimizer"]["default"] == "dftb"
    assert "ga_config" in schema["required"]

    by_method = {
        arm["properties"]["optimizer"]["const"]: arm
        for arm in schema["dependencies"]["optimizer"]["oneOf"]
    }
    xtb_arm = by_method["xtb"]
    assert xtb_arm["required"] == ["xtb_parameters"]
    assert "dftb_parameters" not in xtb_arm["properties"]
    xtb_schema = xtb_arm["properties"]["xtb_parameters"]
    assert "anyOf" not in xtb_schema
    assert "molecules" not in xtb_schema["properties"]
    assert "optimize" not in xtb_schema["properties"]
    for name in ("level_of_theory", "compute_gradients", "additional_keywords"):
        assert name in xtb_schema["properties"]
    assert xtb_schema["default"]["level_of_theory"]["method"] == CrestLevelOfTheoryEnum.GFN2_XTB.value

    dftb_arm = by_method["dftb"]
    assert dftb_arm["required"] == ["dftb_parameters"]
    assert "xtb_parameters" not in dftb_arm["properties"]
    dftb_schema = dftb_arm["properties"]["dftb_parameters"]
    assert "anyOf" not in dftb_schema
    optimization_on = dftb_schema["dependencies"]["optimization"]["oneOf"][1]
    for name in ("optimization_method", "max_optimization_steps", "force_tolerance"):
        assert name in optimization_on["properties"]
    assert dftb_schema["default"]["optimization"] is True
    assert dftb_schema["default"]["max_optimization_steps"] == 1000


def test_ui_conditions_follow_the_selected_optimizer():
    ui = GARelaxInput.ui_schema()
    assert ui["ga_config"]["ui:field"] == "GenericFormField"
    assert "ui:condition" not in ui["ga_config"]
    assert ui["ga_config"]["ui:options"]["model"].endswith("GAConfig")
    assert ui["optimizer"]["ui:widget"] == "select"
    assert ui["xtb_parameters"]["ui:field"] == "GenericFormField"
    assert ui["xtb_parameters"]["ui:condition"] == {"optimizer": "xtb"}
    assert ui["xtb_parameters"]["ui:options"]["model"].endswith("XTBOptimizerParameters")
    assert ui["dftb_parameters"]["ui:field"] == "GenericFormField"
    assert ui["dftb_parameters"]["ui:condition"] == {"optimizer": "dftb"}
    assert ui["dftb_parameters"]["ui:options"]["model"].endswith("DftbInput")


def test_unselected_optimizer_parameters_are_cleared_and_missing_ones_rejected():
    ga = _ga()
    with pytest.raises(ValidationError, match="GA settings are required"):
        GARelaxInput()
    with pytest.raises(ValidationError, match="DFTB optimizer parameters are required"):
        GARelaxInput(ga_config=ga)
    with pytest.raises(ValidationError, match="xTB optimizer parameters are required"):
        GARelaxInput(ga_config=ga, optimizer=GARelaxOptimizer.XTB)
    with pytest.raises(ValidationError, match="DFTB optimization must be enabled"):
        GARelaxInput(ga_config=ga, dftb_parameters=DftbInput(optimization=False))
    with pytest.raises(ValidationError, match="optimizer is required"):
        GARelaxInput(ga_config=ga, optimizer=None, dftb_parameters=_dftb())

    dftb_only = GARelaxInput(
        ga_config=ga,
        dftb_parameters=_dftb(charge=-1),
        xtb_parameters=XTBOptimizerParameters(compute_gradients=True),
    )
    assert dftb_only.optimizer is GARelaxOptimizer.DFTB
    assert dftb_only.xtb_parameters is None
    assert dftb_only.dftb_parameters.charge == -1
    assert dftb_only.dftb_parameters.optimization is True
    assert dftb_only.ga_config.mode is GAMode.GA

    xtb_only = GARelaxInput(
        ga_config=ga,
        optimizer="xtb",
        xtb_parameters=XTBOptimizerParameters(compute_gradients=True),
        dftb_parameters=_dftb(),
    )
    assert xtb_only.dftb_parameters is None
    assert xtb_only.xtb_parameters.compute_gradients is True


@pytest.mark.asyncio
async def test_ga_relax_runs_ga_then_the_selected_optimizer(monkeypatch):
    ga = _ga()
    calls = []
    higher = Molecule(atoms=[Atom(element="C", x=1.0, y=0.0, z=0.0)])
    lower = Molecule(atoms=[Atom(element="C", x=2.0, y=0.0, z=0.0)])

    async def fake_ga(config, **kwargs):
        calls.append(("ga", config))
        generated = MoleculeList()
        generated.append(higher)
        generated.append(lower)
        return SimpleNamespace(molecules=generated)

    async def fake_dftb(molecules, opts, **kwargs):
        calls.append(("dftb", opts, len(molecules)))
        rows = {
            "high": {
                "success": True,
                "result_qm_result": SimpleNamespace(final_structure=higher, final_energy=1.5),
            },
            "low": {
                "success": True,
                "result_qm_result": SimpleNamespace(final_structure=lower, final_energy=-0.5),
            },
        }
        return SimpleNamespace(dataset={"results": rows})

    async def fake_xtb(xtb_input, **kwargs):
        calls.append(("xtb", xtb_input))
        relaxed = MoleculeList()
        for index, molecule in enumerate(xtb_input.molecules):
            molecule.properties["energy"] = 0.2 if index == 0 else -0.8
            relaxed.append(molecule)
        return SimpleNamespace(molecule_list=relaxed)

    monkeypatch.setattr(
        "molecular_qm_search.optimization.ga.run_ga_conformer_gen",
        fake_ga,
    )
    import molecular_qm_dftb.nodes.dftb_list_calculator  # noqa: F401

    monkeypatch.setattr(
        sys.modules["molecular_qm_dftb.nodes.dftb_list_calculator"],
        "dftb_list_calculator",
        fake_dftb,
    )
    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.crest.xtb_optimize_molecule_list",
        fake_xtb,
    )

    dftb_opts = _dftb(max_optimization_steps=40, charge=2)
    runner = _runner()
    result = await ga_relax.__wrapped__(
        GARelaxInput(ga_config=ga, dftb_parameters=dftb_opts),
        node_runner=runner,
    )
    assert result.status == "completed"
    assert [molecule.atoms[0].x for molecule in runner.molecules] == [2.0, 1.0]
    assert [molecule.properties["energy"] for molecule in runner.molecules] == [-0.5, 1.5]
    assert calls[0][0] == "ga"
    assert calls[0][1].num_confs == 3
    assert calls[1] == ("dftb", dftb_opts, 2)

    calls.clear()
    xtb_opts = XTBOptimizerParameters(compute_gradients=True, additional_keywords="--gfn 1")
    xtb_runner = _runner()
    await ga_relax.__wrapped__(
        GARelaxInput(ga_config=ga, optimizer=GARelaxOptimizer.XTB, xtb_parameters=xtb_opts),
        node_runner=xtb_runner,
    )
    assert [call[0] for call in calls] == ["ga", "xtb"]
    xtb_input = calls[1][1]
    assert xtb_input.optimize is True
    assert xtb_input.compute_gradients is True
    assert xtb_input.additional_keywords == "--gfn 1"
    assert xtb_input.level_of_theory.method is CrestLevelOfTheoryEnum.GFN2_XTB
    assert len(xtb_input.molecules) == 2
    assert [molecule.properties["energy"] for molecule in xtb_runner.molecules] == [-0.8, 0.2]


@pytest.mark.asyncio
async def test_ga_relax_rejects_an_empty_population_and_a_failed_relaxation(monkeypatch):
    ga = _ga()

    async def empty_ga(config, **kwargs):
        return SimpleNamespace(molecules=MoleculeList())

    monkeypatch.setattr(
        "molecular_qm_search.optimization.ga.run_ga_conformer_gen",
        empty_ga,
    )
    with pytest.raises(ValueError, match="returned no conformers"):
        await ga_relax.__wrapped__(
            GARelaxInput(ga_config=ga, dftb_parameters=_dftb()),
            node_runner=_runner(),
        )

    async def fake_ga(config, **kwargs):
        generated = MoleculeList()
        generated.append(_carbon())
        return SimpleNamespace(molecules=generated)

    async def failed_dftb(molecules, opts, **kwargs):
        return SimpleNamespace(
            dataset={"results": {"row": {"success": False, "error": "scf failed"}}}
        )

    monkeypatch.setattr(
        "molecular_qm_search.optimization.ga.run_ga_conformer_gen",
        fake_ga,
    )
    import molecular_qm_dftb.nodes.dftb_list_calculator  # noqa: F401

    monkeypatch.setattr(
        sys.modules["molecular_qm_dftb.nodes.dftb_list_calculator"],
        "dftb_list_calculator",
        failed_dftb,
    )
    runner = _runner()
    with pytest.raises(ValueError, match="produced no conformers"):
        await ga_relax.__wrapped__(
            GARelaxInput(ga_config=ga, dftb_parameters=_dftb()),
            node_runner=runner,
        )
    assert runner.warnings
