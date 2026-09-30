import asyncio
import sys
import types
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from molecular_qm_models import Atom, Molecule
from molecular_qm_search.optimization.ga_evaluation import (
    CallableEvaluator, DFTBEvaluator, XTBEvaluator, RDKitEvaluator, make_evaluator, validate_results,
)
from molecular_qm_search.optimization.ga_models import GAOptimizationMethod


@pytest.fixture
def dftb_stub(monkeypatch):
    calls = []
    outputs = []
    input_module = types.ModuleType("molecular_qm_dftb.models.dftb_input")
    input_module.DftbInput = lambda **kwargs: SimpleNamespace(**kwargs)
    calculator_module = types.ModuleType("molecular_qm_dftb.nodes.dftb_calculator")

    async def calculator(molecule, opts, **kwargs):
        calls.append((molecule, opts, kwargs, asyncio.get_running_loop()))
        if outputs:
            return outputs.pop(0)
        structure = Molecule.from_molecule(molecule)
        structure.atoms[0].x += 1.0
        return SimpleNamespace(qm_result=SimpleNamespace(
            final_energy=-0.5, final_structure=structure,
            normal_termination=True, scf_converged=True, optimization_converged=True,
        ))

    calculator_module.dftb_calculator = calculator
    monkeypatch.setitem(sys.modules, input_module.__name__, input_module)
    monkeypatch.setitem(sys.modules, calculator_module.__name__, calculator_module)
    return calls, outputs


@pytest.fixture
def molecule():
    return Molecule(atoms=[Atom(element="H", x=0, y=0, z=0)],
                    properties={"label": "original"})


def test_dftb_score_and_optimize_use_options_geometry_and_units(dftb_stub, molecule):
    calls, _ = dftb_stub
    evaluator = DFTBEvaluator({"charge": 1, "optimization": True},
                              node_kwargs={"node_runner": object(), "project": "test"})
    scored = evaluator.score([molecule])[0]
    optimized = evaluator.optimize([molecule], max_iters=40)[0]
    assert scored.atoms[0].x == molecule.atoms[0].x == 0
    assert optimized.atoms[0].x == 1.0
    assert optimized.properties["energy"] == pytest.approx(-0.5 * 627.5094740631)
    assert optimized.properties["energy_hartree"] == -0.5
    assert optimized.properties["label"] == "original"
    assert optimized.properties["optimization_converged"] is True
    assert calls[0][1].optimization is False
    assert calls[1][1].optimization is True
    assert calls[1][1].max_optimization_steps == 40
    assert calls[1][1].charge == 1
    assert calls[1][1].tolerate_failure is False
    assert calls[1][2] == {"project": "test"}


@pytest.mark.asyncio
async def test_dftb_submits_from_worker_to_parent_event_loop(dftb_stub, molecule):
    calls, _ = dftb_stub
    loop = asyncio.get_running_loop()
    evaluator = DFTBEvaluator(loop=loop)
    results = await asyncio.to_thread(evaluator.score, [molecule, molecule])
    assert len(results) == 2
    assert all(call[3] is loop for call in calls)
    with pytest.raises(RuntimeError, match="asyncio.to_thread"):
        evaluator.score([molecule])


@pytest.mark.parametrize("failure", ["missing", "scf", "structure"])
def test_dftb_failures_are_not_silently_scored(dftb_stub, molecule, failure):
    _, outputs = dftb_stub
    result = SimpleNamespace(final_energy=-1, final_structure=molecule,
                             normal_termination=True, scf_converged=True)
    if failure == "missing":
        outputs.append(SimpleNamespace(error_message="backend failed"))
    else:
        if failure == "scf":
            result.scf_converged = False
        else:
            result.final_structure = None
        outputs.append(SimpleNamespace(qm_result=result))
    with pytest.raises(RuntimeError):
        DFTBEvaluator().optimize([molecule], max_iters=5)


def test_factory_explicit_method_overrides_legacy_forcefield():
    evaluator = make_evaluator(GAOptimizationMethod.RDKIT_UFF, forcefield="mmff")
    assert isinstance(evaluator, RDKitEvaluator)
    assert evaluator.forcefield == "rdkit_uff"
    assert isinstance(make_evaluator("dftb"), DFTBEvaluator)
    assert isinstance(make_evaluator("xtb"), XTBEvaluator)
    with pytest.raises(ValueError, match="Unsupported GA"):
        make_evaluator("orca")


def test_external_batch_adapter(molecule):
    calls = []
    def optimize(molecules, *, max_iters):
        calls.append(max_iters)
        return molecules
    evaluator = CallableEvaluator(score_batch=lambda molecules: molecules, optimize_batch=optimize)
    assert evaluator.score([molecule]) == [molecule]
    assert evaluator.optimize([molecule], max_iters=12) == [molecule]
    assert calls == [12]


def test_wrong_units_and_atom_sequence_are_rejected(molecule):
    result = Molecule.from_molecule(molecule)
    result.properties.update(energy=-1.0, energy_unit="hartree")
    with pytest.raises(ValueError, match="kcal/mol"):
        validate_results([molecule], [result])
    result.properties["energy_unit"] = "kcal/mol"
    result.atoms[0].element = "C"
    with pytest.raises(ValueError, match="atom sequence"):
        validate_results([molecule], [result])


@pytest.fixture
def xtb_stub(monkeypatch):
    calls = []
    input_module = types.ModuleType("molecular_qm_psi4.models.crest_input")
    input_module.XTBInput = lambda **kwargs: SimpleNamespace(**kwargs)
    calculator_module = types.ModuleType("molecular_qm_psi4.nodes.crest")

    async def calculate(opts, *, optimize):
        assert opts.optimize is optimize
        calls.append(opts)
        for molecule in opts.molecules:
            molecule.properties["energy"] = -0.25
            if opts.optimize:
                molecule.atoms[0].x += 2
                molecule.properties["optimization_converged"] = True
        return SimpleNamespace(molecule_list=opts.molecules)

    async def score(opts, **kwargs):
        return await calculate(opts, optimize=False)

    async def optimize(opts, **kwargs):
        return await calculate(opts, optimize=True)

    calculator_module.xtb_molecule_list = score
    calculator_module.xtb_optimize_molecule_list = optimize
    monkeypatch.setitem(sys.modules, input_module.__name__, input_module)
    monkeypatch.setitem(sys.modules, calculator_module.__name__, calculator_module)
    return calls, calculator_module


@pytest.mark.parametrize("method", ["xtb", GAOptimizationMethod.XTB])
def test_xtb_batch_dispatch_units_options_and_input_isolation(xtb_stub, molecule, method):
    calls, _ = xtb_stub
    evaluator = make_evaluator(
        method, backend_options={"level_of_theory": {"method": "gfn1", "charge": 1}}
    )
    scored = evaluator.score([molecule, molecule])
    optimized = evaluator.optimize([molecule], max_iters=17)
    assert len(scored) == 2
    assert calls[0].optimize is False
    assert calls[1].optimize is True
    assert calls[1].max_iters == 17
    assert calls[1].level_of_theory == {"method": "gfn1", "charge": 1}
    assert molecule.atoms[0].x == scored[0].atoms[0].x == 0
    assert "energy" not in molecule.properties
    assert optimized[0].atoms[0].x == 2
    assert optimized[0].properties["energy"] == pytest.approx(-0.25 * 627.5094740631)
    assert optimized[0].properties["label"] == "original"
    assert optimized[0].properties["optimization_converged"] is True


def test_xtb_failed_batch_is_rejected(xtb_stub, molecule):
    _, module = xtb_stub
    async def failed(opts, **kwargs):
        return SimpleNamespace(molecule_list=opts.molecules, error_message="SCC failed")
    module.xtb_molecule_list = failed
    with pytest.raises(RuntimeError, match="SCC failed"):
        XTBEvaluator().score([molecule])


def test_evaluator_configuration_uses_pydantic_validation():
    evaluator = RDKitEvaluator.model_validate({"forcefield": "uff", "threads": 2})
    assert isinstance(evaluator, BaseModel)
    assert evaluator.model_dump() == {"forcefield": "uff", "threads": 2}
    with pytest.raises(ValidationError):
        RDKitEvaluator(threads=-1)
    with pytest.raises(ValidationError):
        CallableEvaluator(score_batch="not callable", optimize_batch=lambda mols, **kw: mols)
    assert isinstance(
        CallableEvaluator(score_batch=lambda mols: mols, optimize_batch=lambda mols, **kw: mols),
        BaseModel,
    )
