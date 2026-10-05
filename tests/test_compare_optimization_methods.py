from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from molecular_qm_models import Atom, Molecule, MoleculeList
from molecular_qm_psi4.models.crest_input import CrestInput
from molecular_qm_search.optimization.compare_optimization_methods import (
    CompareConformers,
    compare_optimization_methods,
)
from molecular_qm_search.optimization.models.ga_models import (
    GAConfig,
    GAConformerParameters,
    GAOptimizationMethod,
)


def _carbon():
    return Molecule(atoms=[Atom(element="C", x=0.0, y=0.0, z=0.0)])


def _runner():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        fail=lambda *args, **kwargs: None,
    )


def test_ga_conformer_parameters_match_ga_config_except_molecule():
    assert set(GAConformerParameters.model_fields) == set(GAConfig.model_fields) - {
        "initial_molecule"
    }
    for name, field in GAConformerParameters.model_fields.items():
        if name in {"field_name", "id"}:
            continue
        config_field = GAConfig.model_fields[name]
        assert field.annotation == config_field.annotation
        assert field.default == config_field.default


def test_schema_exposes_ga_parameters_only_when_selected_and_crest_only_when_on():
    schema = CompareConformers.json_schema()
    assert "ga_config" not in schema["properties"]
    assert "crest_input" not in schema["properties"]
    assert "molecule" not in schema["properties"]

    ga_off, ga_on = schema["dependencies"]["use_ga"]["oneOf"]
    assert ga_off["properties"]["use_ga"]["const"] is False
    assert "ga_config" not in ga_off["properties"]
    ga_schema = ga_on["properties"]["ga_config"]
    assert ga_on["properties"]["use_ga"]["const"] is True
    assert "anyOf" not in ga_schema
    assert "initial_molecule" not in ga_schema["properties"]
    for name in (
        "num_confs",
        "mode",
        "pop_size",
        "generations",
        "mutation_rate",
        "optimization_method",
        "parallel_children",
        "match_double_bonds",
    ):
        assert name in ga_schema["properties"]
    assert ga_schema["default"]["generations"] == 50
    assert ga_schema["default"]["optimization_method"] == GAOptimizationMethod.RDKIT_MMFF.value

    crest_off, crest_on = schema["dependencies"]["use_crest"]["oneOf"]
    assert crest_off["properties"]["use_crest"]["const"] is False
    assert "molecule" in crest_off["properties"]
    assert "crest_input" not in crest_off["properties"]
    assert crest_off["required"] == ["molecule"]
    crest_schema = crest_on["properties"]["crest_input"]
    assert crest_on["properties"]["use_crest"]["const"] is True
    assert "anyOf" not in crest_schema
    assert "molecule" in crest_schema["properties"]
    assert "level_of_theory" in crest_schema["properties"]
    assert "md_options" in crest_schema["properties"]
    assert "molecule" not in crest_on["properties"]


def test_ui_conditions_follow_the_selected_generator():
    ui = CompareConformers.ui_schema()
    assert ui["ga_config"]["ui:field"] == "GenericFormField"
    assert ui["ga_config"]["ui:condition"] == {"use_ga": True}
    assert ui["ga_config"]["ui:options"]["model"].endswith("GAConformerParameters")
    assert ui["crest_input"]["ui:field"] == "GenericFormField"
    assert ui["crest_input"]["ui:condition"] == {"use_crest": True}
    assert ui["crest_input"]["ui:options"]["model"].endswith("CrestInput")
    assert ui["molecule"]["ui:field"] == "MoleculeField"
    assert ui["molecule"]["ui:condition"] == {"use_crest": False}
    assert ui["use_ga"]["ui:widget"] == "checkbox"
    assert ui["use_crest"]["ui:widget"] == "checkbox"


def test_disabled_generator_parameters_are_cleared_and_missing_ones_rejected():
    molecule = _carbon()
    ga_config = GAConformerParameters(num_confs=12, generations=4)
    crest_input = CrestInput(molecule=molecule)

    with pytest.raises(ValidationError, match="GA parameters are required"):
        CompareConformers(molecule=molecule)

    with pytest.raises(ValidationError, match="A molecule is required"):
        CompareConformers(use_ga=False, use_crest=False)

    with pytest.raises(ValidationError, match="CREST parameters are required"):
        CompareConformers(use_ga=False, use_crest=True)

    crest_only = CompareConformers(
        use_ga=False,
        use_crest=True,
        molecule=molecule,
        crest_input=crest_input,
        ga_config=ga_config,
    )
    assert crest_only.molecule is None
    assert crest_only.ga_config is None
    assert crest_only.crest_input.molecule.atoms[0].element == "C"

    ga_only = CompareConformers(
        use_crest=False,
        molecule=molecule,
        ga_config=ga_config,
        crest_input=crest_input,
    )
    assert ga_only.use_ga is True
    assert ga_only.crest_input is None
    assert ga_only.molecule is molecule
    assert ga_only.ga_config.num_confs == 12
    assert ga_only.ga_config.generations == 4


@pytest.mark.asyncio
async def test_ga_selection_calls_ga_conformer_gen_with_its_parameters(monkeypatch):
    molecule = _carbon()
    ga_config = GAConformerParameters(
        num_confs=12,
        generations=4,
        mode="ga-min",
        optimization_method=GAOptimizationMethod.XTB,
        parallel_children=3,
    )
    seen = []

    async def fake_ga(config, **kwargs):
        seen.append(config)
        return SimpleNamespace(molecules=MoleculeList())

    def fail_if_called(*args, **kwargs):
        raise AssertionError("this generator should not run")

    monkeypatch.setattr(
        "molecular_qm_search.optimization.ga.run_ga_conformer_gen", fake_ga
    )
    monkeypatch.setattr(
        "molecular_qm_psi4.nodes.crest.crest", fail_if_called
    )
    monkeypatch.setattr(
        "molecular_qm_util.obabel_scripts.openbabel_conformers.conformers_openbabel",
        fail_if_called,
    )
    monkeypatch.setattr(
        "molecular_qm_util.rdkit_scripts.rdkit_conformers.conformers_rdkit",
        fail_if_called,
    )

    opts = CompareConformers(
        use_ga=True,
        use_babel=False,
        use_rdkit=False,
        use_crest=False,
        molecule=molecule,
        ga_config=ga_config,
    )
    result = await compare_optimization_methods.__wrapped__(opts, node_runner=_runner())

    assert isinstance(result, MoleculeList)
    assert len(result) == 0
    assert len(seen) == 1
    config = seen[0]
    assert isinstance(config, GAConfig)
    assert config.initial_molecule is molecule
    assert config.field_name == "GAConfig"
    assert config.num_confs == 12
    assert config.generations == 4
    assert config.mode == "ga-min"
    assert config.optimization_method is GAOptimizationMethod.XTB
    assert config.parallel_children == 3


@pytest.mark.asyncio
async def test_crest_parameters_are_used_only_when_crest_is_selected(monkeypatch):
    molecule = _carbon()
    crest_calls = []

    async def fake_crest(crest_input, **kwargs):
        crest_calls.append(crest_input)
        return SimpleNamespace(molecule_list=MoleculeList())

    async def fake_ga(config, **kwargs):
        assert config.initial_molecule is molecule
        return SimpleNamespace(molecules=MoleculeList())

    monkeypatch.setattr("molecular_qm_psi4.nodes.crest.crest", fake_crest)
    monkeypatch.setattr(
        "molecular_qm_search.optimization.ga.run_ga_conformer_gen", fake_ga
    )
    monkeypatch.setattr(
        "molecular_qm_util.obabel_scripts.openbabel_conformers.conformers_openbabel",
        lambda *args, **kwargs: MoleculeList(),
    )

    off = CompareConformers(
        use_ga=True,
        use_babel=False,
        use_crest=False,
        molecule=molecule,
        ga_config=GAConformerParameters(),
    )
    await compare_optimization_methods.__wrapped__(off, node_runner=_runner())
    assert crest_calls == []

    on = CompareConformers(
        use_ga=True,
        use_babel=False,
        use_crest=True,
        crest_input=CrestInput(molecule=molecule),
        ga_config=GAConformerParameters(num_confs=8),
    )
    await compare_optimization_methods.__wrapped__(on, node_runner=_runner())
    assert len(crest_calls) == 1
    assert crest_calls[0].molecule is molecule
