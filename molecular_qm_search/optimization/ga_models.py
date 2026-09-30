from odmantic import Model, Reference
from simstack.models import simstack_model
from molecular_qm_models import Molecule
from enum import Enum
from typing import Any, Dict, Optional

class GAOptimizationMethod(str, Enum):
    RDKIT_UFF = "rdkit_uff"
    RDKIT_UFFF = "rdkit_uff"  # Backwards-compatible alias.
    RDKIT_MMFF94 = "rdkit_mmff94"
    RDKIT_MMFF94S = "rdkit_mmff94s"
    DFTB = "dftb"
    XTB = "xtb"
    ORCA = "orca"
    PSI4 = "psi4"


@simstack_model
class GAConfig(Model):
    # GA starts from a SimStack Molecule provided as the initial structure.
    initial_molecule: Molecule = Reference()

    num_confs: int = 50
    seed: int = 1
    threads: int = 0
    mode: str = "ga"  # ga, ga-min, ga-select
    pop_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.5
    dihedral_interval: float = 30.0
    forcefield: str = "mmff"
    # None preserves the legacy forcefield setting. Explicit methods take precedence.
    optimization_method: Optional[GAOptimizationMethod] = None
    backend_options: Dict[str, Any] = {}
    max_iters: int = 500
    smart_opt: bool = False
    n_prune: int = 10
    prune_rms_thresh: float = 0.1
    db_treatment: str = "180+step"
