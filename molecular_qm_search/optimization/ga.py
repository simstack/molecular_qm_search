import asyncio

from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult

from molecular_qm_search.optimization.models.ga_models import GAConfig
from molecular_qm_search.optimization.lib.ga_evaluation import make_evaluator
from .ga_method import (
    generate_ga_conformers,
    generate_ga_min_conformers,
    generate_ga_select_conformers,
)


@node
async def run_ga_conformer_gen(config: GAConfig, **kwargs) -> SimstackResult:
    node_runner = kwargs.get("node_runner")

    # The GA now takes a single simstack Molecule as its only input source.
    if config.initial_molecule is None:
        raise ValueError("GAConfig must provide an initial_molecule.")

    node_runner.info(
        f"Starting GA ({config.mode}) conformer generation from the provided initial_molecule."
    )

    if config.mode == "ga":
        gen_func = generate_ga_conformers
    elif config.mode == "ga-min":
        gen_func = generate_ga_min_conformers
    elif config.mode == "ga-select":
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

    extra_args = {}
    if config.mode == "ga-select":
        extra_args["n_prune"] = config.n_prune
        extra_args["prune_rms_thresh"] = config.prune_rms_thresh

    ranked = await asyncio.to_thread(
        gen_func,
        initial_mol=config.initial_molecule,
        num_confs=config.num_confs,
        pop_size=config.pop_size,
        generations=config.generations,
        mutation_rate=config.mutation_rate,
        crossover_rate=config.crossover_rate,
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
        **extra_args,
    )

    node_runner.result = ranked
    return node_runner.succeed()
