import asyncio
import importlib
import inspect
from types import SimpleNamespace

import pytest
from simstack.tables import node_children
from simstack.util.docstring_parser import DocstringParser

from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_search.optimization.models.voronoi_models import VoronoiConfig

CALCULATOR_NODES = {
    "dftb_list_calculator": "molecular_qm_dftb.nodes.dftb_list_calculator.dftb_list_calculator",
    "xtb_molecule_list": "molecular_qm_psi4.nodes.crest.xtb_molecule_list",
    "xtb_optimize_molecule_list": "molecular_qm_psi4.nodes.crest.xtb_optimize_molecule_list",
    "score_molecules_rdkit": "molecular_qm_util.rdkit_scripts.rdkit_optimize.score_molecules_rdkit",
    "optimize_molecules_rdkit": "molecular_qm_util.rdkit_scripts.rdkit_optimize.optimize_molecules_rdkit",
}


def test_node_documents_molecules_and_neighbor_graph():
    module = importlib.import_module("molecular_qm_search.optimization.voronoi")
    parsed = DocstringParser(inspect.getdoc(module.run_voronoi_conformer_gen)).simstack_results()
    assert parsed["molecules"]["type"] == "MoleculeList"
    assert parsed["neighbor_graph"]["type"] == "SimpleTable"
    assert parsed["energy_chart"]["type"] == "ChartArtifactModel"


@pytest.mark.asyncio
async def test_node_exposes_calculator_children_as_called_nodes(tmp_path, monkeypatch):
    module = importlib.import_module("molecular_qm_search.optimization.voronoi")
    parsed = DocstringParser(inspect.getdoc(module.run_voronoi_conformer_gen)).called_nodes()
    assert parsed == list(CALCULATOR_NODES)

    models = [
        SimpleNamespace(
            id=name, name=name, function_mapping=mapping,
            description="", called_nodes=[],
        )
        for name, mapping in CALCULATOR_NODES.items()
    ]
    parent = SimpleNamespace(
        id="voronoi",
        name="run_voronoi_conformer_gen",
        function_mapping="molecular_qm_search.optimization.voronoi.run_voronoi_conformer_gen",
        description="",
        called_nodes=[],
    )
    models.append(parent)
    updates = []

    class FakeCollection:
        async def update_one(self, query, update):
            updates.append((query, update))

    class FakeDatabase:
        async def find(self, model):
            return models

        def get_collection(self, model):
            return FakeCollection()

    async def import_function(function_mapping, database):
        if function_mapping == parent.function_mapping:
            return module.run_voronoi_conformer_gen
        return lambda: None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(node_children, "import_function", import_function)
    await node_children.update_node_children(FakeDatabase(), "")

    parent_update = next(update for query, update in updates if query == {"_id": "voronoi"})
    assert parent_update == {"$set": {"called_nodes": sorted(CALCULATOR_NODES.values())}}


@pytest.mark.asyncio
async def test_node_returns_molecules_and_requires_the_neighbor_graph(monkeypatch):
    module = importlib.import_module("molecular_qm_search.optimization.voronoi")
    ranked = MoleculeList()

    def generate(**kwargs):
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        kwargs["node_runner"].energy_chart = object()
        kwargs["node_runner"].energy_histogram = object()
        kwargs["node_runner"].operator_stats = object()
        kwargs["node_runner"].neighbor_graph = object()
        return ranked

    monkeypatch.setattr(module, "generate_voronoi_conformers", generate)
    runner = SimpleNamespace(info=lambda *args: None)
    runner.succeed = lambda: runner
    config = VoronoiConfig(initial_molecule=Molecule(), optimization_method="XTB")
    result = await module.run_voronoi_conformer_gen.__wrapped__(
        config, node_runner=runner, evaluator=object(),
    )
    assert result.molecules is ranked

    def generate_without_graph(**kwargs):
        kwargs["node_runner"].energy_chart = object()
        kwargs["node_runner"].energy_histogram = object()
        kwargs["node_runner"].operator_stats = object()
        return ranked

    monkeypatch.setattr(module, "generate_voronoi_conformers", generate_without_graph)
    with pytest.raises(ValueError, match="neighbor graph"):
        await module.run_voronoi_conformer_gen.__wrapped__(
            config, node_runner=SimpleNamespace(info=lambda *args: None), evaluator=object(),
        )
