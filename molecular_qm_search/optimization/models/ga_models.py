from enum import Enum
from typing import Any, Dict

from odmantic import Field, Model, Reference
from simstack.models import simstack_model
from simstack.util.generate_ui_schema import generate_ui_schema

from molecular_qm_models import Molecule


class GAOptimizationMethod(str, Enum):
    """Energy and geometry backend for the conformer GA.

    ``RDKIT/mmff`` is MMFF94. ``RDKIT/mmff94s`` and ``RDKIT/uff`` are the other
    RDKit force fields the evaluator can construct.
    """

    RDKIT_MMFF = "RDKIT/mmff"
    RDKIT_MMFF94S = "RDKIT/mmff94s"
    RDKIT_UFF = "RDKIT/uff"
    XTB = "XTB"
    DFTB = "DFTB"


@simstack_model
class GAConfig(Model):
    # GA starts from a SimStack Molecule provided as the initial structure.
    field_name: str = "GAConfig"
    initial_molecule: Molecule = Reference()

    num_confs: int = 50
    seed: int = 1
    parallel_children: int = Field(
        0,
        title="Parallel children",
        description="Evaluations to run at the same time. 0 keeps the backend default.",
    )
    mode: str = "ga"  # ga, ga-min, ga-select
    pop_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.5
    dihedral_interval: float = 30.0
    optimization_method: GAOptimizationMethod = Field(
        GAOptimizationMethod.RDKIT_MMFF,
        title="Optimization method",
    )
    backend_options: Dict[str, Any] = {}
    max_iters: int = 500
    smart_opt: bool = False
    n_prune: int = 10
    prune_rms_thresh: float = 0.1
    db_treatment: str = "180+step"
    match_double_bonds: bool = Field(
        True,
        title="Match double bonds",
        description="Include double bonds when the initial rotatable-bond list is built.",
    )
    rotatable_bond_min: float = Field(
        -180.0,
        title="Rotatable bond minimum",
        description="Lower dihedral bound in degrees for discovered rotatable bonds.",
    )
    rotatable_bond_max: float = Field(
        180.0,
        title="Rotatable bond maximum",
        description="Upper dihedral bound in degrees for discovered rotatable bonds.",
    )

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        return ui
