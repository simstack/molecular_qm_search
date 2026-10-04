# Molecular QM search

SimStack nodes to search for minima and transition states.

The conformer GA operates on `Molecule`, `MoleculeList`, and
`InternalCoordinatesList` from `molecular_qm_models`.

## Population generation

`optimization.ga_population.PopulationGenerator` discovers torsions through
`molecular_qm_util.get_rotatable_bonds`, or accepts an explicit
`InternalCoordinatesList`. Discovery receives `match_double_bonds`,
`rotatable_bond_min`, and `rotatable_bond_max`. `generate`, `reproduce`,
`molecule_from_coordinates`, and `coordinates_from_molecule` methods never
score or optimize a structure. Passing coordinates avoids RDKit torsion discovery.
Each generator owns its seeded random state; mutation does not modify parents.

## Evaluation and optimization

`optimization.ga_evaluation.MoleculeEvaluator` defines two ordered batch methods:

```python
score(molecules: list[Molecule]) -> list[Molecule]
optimize(molecules: list[Molecule], *, max_iters: int) -> list[Molecule]
```

Both return new molecules in input order, retaining atom order. `score` preserves
geometry. Every result must have a finite `properties["energy"]` in kcal/mol.
Optimizers return the Cartesian structure whose energy was evaluated; final
results are not reconstructed from torsions. Backend failures raise exceptions.

Available adapters:

- `RDKitEvaluator`: calls `score_molecules_rdkit` and `optimize_molecules_rdkit`
  in `molecular_qm_util`. Supports MMFF94, MMFF94s and UFF. RDKit conversion,
  force-field construction and threaded batch evaluation stay in that package.
- `DFTBEvaluator`: calls `molecular_qm_dftb.nodes.dftb_calculator` with `DftbInput`.
- `XTBEvaluator`: calls `xtb_molecule_list` and `xtb_optimize_molecule_list` from
  `molecular_qm_psi4.nodes.crest`, using `XTBInput`. Use a version of that sister
  repository with `XTBInput.max_iters` to honor GA optimization budgets.
- `CallableEvaluator`: adapts other molecule batch functions to the same contract.

DFTB and xTB convert Hartree to kcal/mol and retain the original value in
`properties["energy_hartree"]`. `backend_options` is passed to `DftbInput` or
`XTBInput`; the GA overrides optimization mode and iteration count for each call.
Quantum adapters submit SimStack child nodes and require the normal configured
SimStack context/resources and calculator runtime. They are imported lazily.

```python
from molecular_qm_search.optimization.ga_method import StandardGA

ga = StandardGA(
    initial_mol=molecule,
    node_runner=node_runner,
    optimization_method="XTB",
    backend_options={"level_of_theory": {"method": "gfn2", "charge": 0}},
    pop_size=20,
    generations=10,
)
conformers = ga.run()
```

`optimization_method` is `GAOptimizationMethod`: `RDKIT/mmff` (MMFF94),
`RDKIT/mmff94s`, `RDKIT/uff`, `XTB`, or `DFTB`. `parallel_children` is the
number of evaluations to run together. A supplied `evaluator` takes
precedence. Unsupported methods raise.

`run_ga_conformer_gen(GAConfig(...))` provides the SimStack node entry point and
returns a `MoleculeList` as `node_runner.result`. It runs the synchronous GA in a
worker thread while quantum calculations execute on the parent event loop.
For direct use from an async workflow, construct the quantum evaluator with
`loop=asyncio.get_running_loop()` and run the GA using `await asyncio.to_thread(ga.run)`.

## GA modes

- `ga`: score each generation, then optimize the final population.
- `ga-min`: optimize each generation; optional smart optimization scores the
  starting batch, partially relaxes it, and fully relaxes selected candidates.
- `ga-select`: filter with the selected backend's energy, select torsional
  diversity, and periodically optimize and prune candidates. This replaces the
  previous hard-coded Lennard-Jones prefilter.

Setup performs no initial optimization. Rigid molecules are optimized once by
the selected evaluator. Existing selection, Cartesian pruning and checkpoint
behavior remain; checkpoints also store the generator's random state.

## Tests

Run `python -m pytest tests` in an environment containing the current sister
packages. Tests cover generation, all GA modes with injected and real RDKit
evaluators, node dispatch, quantum adapter contracts and energy conversion.
Quantum adapter tests use stub nodes; they do not launch DFTB+ or xTB binaries.
