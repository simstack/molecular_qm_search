import asyncio

from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult

from molecular_qm_models import MoleculeList

from molecular_qm_search.optimization.models.voronoi_models import VoronoiConfig
from molecular_qm_search.optimization.lib.ga_evaluation import make_evaluator
from molecular_qm_search.optimization.voronoi_method import generate_voronoi_conformers


@node
async def run_voronoi_conformer_gen(config: VoronoiConfig, **kwargs) -> SimstackResult:
    """Generate conformers by crossing dihedral neighbors on a weighted torus.

    Each scored generation is a set of points on the periodic dihedral torus.
    Site weights are exp(-(energy - min_energy) / weight_temperature), so a
    lower energy pushes the Voronoi boundary farther from that conformer.
    Crossover uses only pairs that share a face of that diagram. Mutations are
    the same dihedral steps as the genetic algorithm.

    DFTB evaluations run as one dftb_list_calculator child per batch. xTB
    evaluations run as child nodes. RDKit batches are the nodes
    score_molecules_rdkit and optimize_molecules_rdkit. The search runs that
    same force-field implementation in-process.

    Called Nodes:
        dftb_list_calculator
        xtb_molecule_list
        xtb_optimize_molecule_list
        score_molecules_rdkit
        optimize_molecules_rdkit

    Raises:
        ValueError: missing initial molecule, invalid weights, or a crossover with no neighbor edge

    SimstackResult:
        molecules (MoleculeList): Population that survived the last selection, lowest single-point energy first. These geometries are not relaxed.
        energy_chart (ChartArtifactModel): Minimum and maximum population energy versus iteration, updated every iteration
        energy_histogram (ChartArtifactModel): Histogram of the population energies, updated every iteration
        diversity_chart (ChartArtifactModel): PCA of pairwise dihedral RMSD, updated every iteration when the population has at least two conformers
        dihedral_histogram_0, dihedral_histogram_1, ... (ChartArtifactModel): Count of conformers in 30 degree bins for each rotatable dihedral, updated every iteration and again from the returned molecules
        operator_stats (SimpleTable): Elite-survival rate of each population-change type
        conformation_energies (SimpleTable): Last scored population, sorted by energy, one row per conformer. The history column starts with that conformer's creation index and walks back to the input: M,ancestor,dihedral,angle or X,ancestor1,ancestor2,cut, ending at I,0
        neighbor_graph (SimpleTable): Undirected neighbor edges of the latest scored generation. site_a and site_b are indexes into that generation.
    """
    node_runner = kwargs.get("node_runner")

    if config.initial_molecule is None:
        raise ValueError("VoronoiConfig must provide an initial_molecule.")

    start_message = "Starting weighted dihedral neighbor conformer generation from the provided initial_molecule."
    if hasattr(node_runner, "log_string"):
        node_runner.log(start_message)
    node_runner.info(start_message)

    evaluator = kwargs.pop("evaluator", None)
    if evaluator is None:
        evaluator = make_evaluator(
            config.optimization_method,
            parallel_children=config.parallel_children,
            backend_options=config.backend_options,
            loop=asyncio.get_running_loop(), node_kwargs=kwargs,
        )

    ranked = await asyncio.to_thread(
        generate_voronoi_conformers,
        initial_mol=config.initial_molecule,
        num_confs=config.num_confs,
        pop_size=config.pop_size,
        generations=config.generations,
        mutation_rate=config.mutation_rate,
        dihedral_interval=config.dihedral_interval,
        seed=config.seed,
        max_iters=config.max_iters,
        parallel_children=config.parallel_children,
        node_runner=node_runner,
        evaluator=evaluator,
        db_treatment=config.db_treatment,
        match_double_bonds=config.match_double_bonds,
        rotatable_bond_min=config.rotatable_bond_min,
        rotatable_bond_max=config.rotatable_bond_max,
        prune_rms_thresh=config.prune_rms_thresh,
        weight_temperature=config.weight_temperature,
        n_witnesses=config.n_witnesses,
        neighbor_gap=config.neighbor_gap,
        artifact_loop=asyncio.get_running_loop(),
    )

    if not isinstance(ranked, MoleculeList):
        raise ValueError(f"Voronoi search returned {type(ranked).__name__}, expected a MoleculeList")
    if getattr(node_runner, "energy_chart", None) is None:
        raise ValueError("Voronoi search did not produce an energy chart")
    if getattr(node_runner, "energy_histogram", None) is None:
        raise ValueError("Voronoi search did not produce an energy histogram")
    if getattr(node_runner, "operator_stats", None) is None:
        raise ValueError("Voronoi search did not produce operator statistics")
    if getattr(node_runner, "neighbor_graph", None) is None:
        raise ValueError("Voronoi search did not produce a neighbor graph")
    node_runner.molecules = ranked
    return node_runner.succeed()
