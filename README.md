# Molecular QM search

SimStack nodes to search for minima and transition states.

The conformer GA operates on `Molecule`, `MoleculeList`, and
`InternalCoordinatesList` from `molecular_qm_models`.

## Population generation

`optimization.ga_population.PopulationGenerator` discovers torsions through
`molecular_qm_util.get_rotatable_bonds`, or accepts an explicit
`InternalCoordinatesList`. Discovery receives `match_double_bonds`,
`rotatable_bond_min`, and `rotatable_bond_max`. `generate` builds the initial
dihedral population. `reproduce` carries the selected parents once and appends
only new conformers: a mutation with probability `mutation_rate`, otherwise a
crossover of two parents. A child whose dihedral RMSD is below
`prune_rms_thresh` (degrees) is discarded. Neither method scores or optimizes
a structure. Passing coordinates avoids RDKit torsion discovery.
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

- `RDKitEvaluator`: calls `evaluate_molecules_rdkit` in `molecular_qm_util`,
  the implementation behind the `score_molecules_rdkit` and
  `optimize_molecules_rdkit` nodes. Supports MMFF94, MMFF94s and UFF. RDKit
  conversion, force-field construction and threaded batch evaluation stay in
  that package and run in-process. Those nodes are declared called nodes of
  `run_ga_conformer_gen`.
- `DFTBEvaluator`: calls `molecular_qm_dftb.nodes.dftb_list_calculator` once per batch with `DftbInput`.
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
`RDKIT/mmff94s`, `RDKIT/uff`, `XTB`, or `DFTB`. `parallel_children` is the RDKit force-field
thread count. `0` leaves RDKit's own default. DFTB and xTB evaluate each batch
as one SimStack child node, so this count is not forwarded to those calculators.
A supplied `evaluator` takes precedence. Unsupported methods raise.

`run_ga_conformer_gen(GAConfig(...))` provides the SimStack node entry point.
It returns the final population as `node_runner.molecules` (`MoleculeList`),
an energy chart of minimum and maximum energy versus iteration, a histogram of
those energies, and a table of elite-survival success rates for crossover
and mutation by bond type.
It runs the synchronous GA in a
worker thread while quantum calculations execute on the parent event loop.
For direct use from an async workflow, construct the quantum evaluator with
`loop=asyncio.get_running_loop()` and run the GA using `await asyncio.to_thread(ga.run)`.

## GA modes

`GAMode` selects the search. The form shows it as a dropdown.

- `ga` (`StandardGA`): each generation builds Cartesian geometries from the
  dihedral chromosomes and scores them. Parents that already have an energy are
  not scored again. After the last generation the carried population is
  optimized once, and the lowest-energy unique conformers are returned, at most
  `num_confs`.
- `ga-min` (`MinimizingGA`): each new conformer is optimized, and the relaxed
  dihedrals replace the chromosome. Carried conformers are not optimized again.
  Optional smart optimization scores the new batch, partially relaxes it, and
  fully relaxes the candidates it keeps. The returned geometries are those
  relaxed structures.
- `ga-select` (`DiversityGA`): breeding uses the same score-only evaluation as
  `ga`. The next generation keeps the low-energy conformers that are at least
  `prune_rms_thresh` degrees apart. When more than `pop_size` qualify, MaxMin
  keeps a diverse subset. Every `n_prune` generations the current population is
  optimized and merged into an archive (`best_stored`). The archive also
  updates on the first generation, because it starts empty, and once more at
  the end if that generation was not already a prune generation. The archive is
  the returned population: lowest energy first, duplicates removed, at most
  `num_confs`.

`n_prune` has no effect on `ga` or `ga-min`.

In every mode the lowest-energy 20 percent (`ga`, `ga-min`) or the
energy-window survivors (`ga-select`) are carried once. Open slots are filled
with a mutation or a crossover. There is no copy operator and no separate
crossover-rate input: the crossover probability is `1 - mutation_rate`.
`mutation_rate` of 0 is all crossover, and crossover with fewer than two
parents raises. A proposed child closer than `prune_rms_thresh` (default 0.1
degrees) to a conformer already kept is dropped. After `pop_size` such
rejections in a row the population shrinks to the number kept and the node log
records the generation, the requested size, the kept size, and the rejected
RMSD.

`db_treatment` is a dropdown for rotatable double bonds: `ignore` drops them,
`180+step` adds 180 degrees to each double-bond mutation, and `treat-as-single`
mutates them like single bonds.

Setup performs no initial optimization. A molecule with no rotatable bonds is
optimized once by the selected evaluator. Checkpoints store the generator's
random state and the energy of each carried conformer. A checkpoint whose
population has no energy slot raises.

## Tests

Run `python -m pytest tests` in an environment containing the current sister
packages. Tests cover generation, all GA modes with injected and real RDKit
evaluators, node dispatch, quantum adapter contracts and energy conversion.
Quantum adapter tests use stub nodes; they do not launch DFTB+ or xTB binaries.
