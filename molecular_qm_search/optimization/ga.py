import asyncio

from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult

from molecular_qm_models import MoleculeList

from molecular_qm_search.optimization.models.ga_models import GAConfig, GAMode
from molecular_qm_search.optimization.lib.ga_evaluation import make_evaluator
from .ga_method import (
    generate_ga_conformers,
    generate_ga_min_conformers,
    generate_ga_select_conformers,
)


@node
async def run_ga_conformer_gen(config: GAConfig, **kwargs) -> SimstackResult:
    """Generate conformers with a genetic algorithm.

    DFTB evaluations run as one dftb_list_calculator child per batch. xTB
    evaluations run as child nodes. RDKit batches are the nodes
    score_molecules_rdkit and optimize_molecules_rdkit. The genetic algorithm
    runs that same force-field implementation in-process.

    Called Nodes:
        dftb_list_calculator
        xtb_molecule_list
        xtb_optimize_molecule_list
        score_molecules_rdkit
        optimize_molecules_rdkit

    Raises:
        ValueError: missing initial molecule, unknown mode, or invalid GA settings

    SimstackResult:
        molecules (MoleculeList): Population that survived the last selection, lowest single-point energy first. The standard GA does not relax these geometries.
        energy_chart (ChartArtifactModel): Minimum and maximum population energy versus iteration, updated every iteration
        energy_histogram (ChartArtifactModel): Histogram of the population energies, updated every iteration
        diversity_chart (ChartArtifactModel): PCA of pairwise dihedral RMSD, updated every iteration when the population has at least two conformers
        dihedral_histogram_0, dihedral_histogram_1, ... (ChartArtifactModel): Count of conformers in 30 degree bins for each rotatable dihedral, updated every iteration and again from the returned molecules
        operator_stats (SimpleTable): Elite-survival rate of each population-change type
        conformation_energies (SimpleTable): Last scored population, sorted by energy, one row per conformer. The history column starts with that conformer's creation index and walks back to the input: M,ancestor,dihedral,angle or X,ancestor1,ancestor2,cut, ending at I,0
    """
    node_runner = kwargs.get("node_runner")

    # The GA now takes a single simstack Molecule as its only input source.
    if config.initial_molecule is None:
        raise ValueError("GAConfig must provide an initial_molecule.")

    start_message = (
        f"Starting GA ({config.mode}) conformer generation from the provided initial_molecule."
    )
    if hasattr(node_runner, "log_string"):
        node_runner.log(start_message)
    node_runner.info(start_message)

    if config.mode == GAMode.GA:
        gen_func = generate_ga_conformers
    elif config.mode == GAMode.GA_MIN:
        gen_func = generate_ga_min_conformers
    elif config.mode == GAMode.GA_SELECT:
        gen_func = generate_ga_select_conformers
    else:
        raise ValueError(f"Unknown GA mode: {config.mode!r}")

    evaluator = kwargs.pop("evaluator", None)
    if evaluator is None:
        evaluator = make_evaluator(
            config.optimization_method,
            parallel_children=config.parallel_children,
            backend_options=config.backend_options,
            loop=asyncio.get_running_loop(), node_kwargs=kwargs,
        )

    extra_args = {"prune_rms_thresh": config.prune_rms_thresh}
    if config.mode == GAMode.GA_SELECT:
        extra_args["n_prune"] = config.n_prune

    ranked = await asyncio.to_thread(
        gen_func,
        initial_mol=config.initial_molecule,
        num_confs=config.num_confs,
        pop_size=config.pop_size,
        generations=config.generations,
        mutation_rate=config.mutation_rate,
        dihedral_interval=config.dihedral_interval,
        seed=config.seed,
        max_iters=config.max_iters,
        parallel_children=config.parallel_children,
        smart_opt=config.smart_opt,
        node_runner=node_runner,
        evaluator=evaluator,
        db_treatment=config.db_treatment,
        match_double_bonds=config.match_double_bonds,
        rotatable_bond_min=config.rotatable_bond_min,
        rotatable_bond_max=config.rotatable_bond_max,
        artifact_loop=asyncio.get_running_loop(),
        **extra_args,
    )

    if not isinstance(ranked, MoleculeList):
        raise ValueError(f"GA returned {type(ranked).__name__}, expected a MoleculeList")
    if getattr(node_runner, "energy_chart", None) is None:
        raise ValueError("GA did not produce an energy chart")
    if getattr(node_runner, "energy_histogram", None) is None:
        raise ValueError("GA did not produce an energy histogram")
    if getattr(node_runner, "operator_stats", None) is None:
        raise ValueError("GA did not produce operator statistics")
    node_runner.molecules = ranked
    return node_runner.succeed()
