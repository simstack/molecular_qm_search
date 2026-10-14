"""Inspection run of the first three GA generations."""
import asyncio
import logging

from molecular_qm_models import Molecule, MoleculeList
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult

from molecular_qm_search.optimization.ga_method import StandardGA
from molecular_qm_search.optimization.models.ga_models import GAOptimizationMethod

GA_INSPECTION_POP_SIZE = 10
GA_INSPECTION_GENERATIONS = 2


class GenerationSnapshotGA(StandardGA):
    """Standard GA that keeps the scored population of each iteration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_molecules: list[MoleculeList] = []

    def selection(self, scored_pop, gen):
        scored_pop.sort(key=lambda item: item[0])
        if not scored_pop:
            raise ValueError(f"Generation {gen} has no conformers")
        molecules = MoleculeList()
        for energy, coords, origin in scored_pop:
            conformer = self.molecule_from_coordinates(coords)
            conformer.properties["energy"] = float(energy)
            conformer.properties["origin"] = origin
            conformer.properties["generation"] = int(gen)
            conformer.properties["label"] = f"gen {gen} {origin} {float(energy):.4g}"
            molecules.add_molecule(conformer)
        self.generation_molecules.append(molecules)
        return super().selection(scored_pop, gen)


def ga_generation_lists(
    molecule: Molecule,
    *,
    node_runner=None,
    evaluator=None,
    coordinates=None,
    artifact_loop=None,
) -> list[MoleculeList]:
    """Run three GA iterations at population size 10 and return each scored generation.

    Iteration 0 is the initial population. The first member is the input geometry.
    Later iterations are the populations kept by selection. Geometries are the
    structures that were scored, before the final force-field optimization.
    """
    if molecule is None or not molecule.atoms:
        raise ValueError("GA generation inspection requires a molecule with atoms")
    if node_runner is None:
        node_runner = logging.getLogger("ga-generations")
    ga = GenerationSnapshotGA(
        molecule,
        node_runner,
        evaluator=evaluator,
        coordinates=coordinates,
        pop_size=GA_INSPECTION_POP_SIZE,
        generations=GA_INSPECTION_GENERATIONS,
        num_confs=GA_INSPECTION_POP_SIZE,
        mutation_rate=0.5,
        dihedral_interval=30.0,
        seed=1,
        prune_rms_thresh=0.1,
        max_iters=500,
        parallel_children=0,
        smart_opt=False,
        db_treatment="180+step",
        match_double_bonds=True,
        rotatable_bond_min=-180.0,
        rotatable_bond_max=180.0,
        optimization_method=GAOptimizationMethod.RDKIT_UFF,
        artifact_loop=artifact_loop,
    )
    ga.run()
    expected = GA_INSPECTION_GENERATIONS + 1
    if len(ga.generation_molecules) != expected:
        raise ValueError(
            f"Expected {expected} GA generations, captured {len(ga.generation_molecules)}"
        )
    for generation, molecule_list in enumerate(ga.generation_molecules):
        if len(list(molecule_list)) < 1:
            raise ValueError(f"Generation {generation} has no conformers")
    return ga.generation_molecules


@node
async def inspect_ga_generations(molecule: Molecule, **kwargs) -> SimstackResult:
    """Save the first three scored GA generations for inspection.

    The search uses the same settings as the conformer GA, with population size
    10 and three iterations. Each list is ordered by energy.

    Parameters:
        molecule (Molecule): Structure that seeds the initial population.

    Raises:
        ValueError: the molecule has no atoms, or a generation was not captured.

    SimstackResult:
        generation_0 (MoleculeList): Initial scored population of 10. The first member is the input geometry when energies tie. generation_1 and generation_2 are the later scored populations.
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("inspect_ga_generations requires node_runner")
    lists = await asyncio.to_thread(
        ga_generation_lists,
        molecule,
        node_runner=node_runner,
        artifact_loop=asyncio.get_running_loop(),
    )
    for index, molecule_list in enumerate(lists):
        setattr(node_runner, f"generation_{index}", molecule_list)
    node_runner.info(
        "GA inspection saved generations "
        + ", ".join(str(len(list(molecule_list))) for molecule_list in lists)
    )
    return node_runner.succeed()
