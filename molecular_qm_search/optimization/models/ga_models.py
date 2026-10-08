from enum import Enum
from typing import Any, Dict

from odmantic import Field, Model, Reference
from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
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


class GAMode(str, Enum):
    """Which conformer search the GA node runs."""

    GA = "ga"
    GA_MIN = "ga-min"
    GA_SELECT = "ga-select"


class DoubleBondTreatment(str, Enum):
    """How rotatable double bonds enter the GA chromosome."""

    IGNORE = "ignore"
    STEP_180 = "180+step"
    TREAT_AS_SINGLE = "treat-as-single"


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
        description="RDKit force-field threads. 0 keeps RDKit's default. DFTB and xTB ignore it and evaluate each batch as one child node.",
    )
    mode: GAMode = Field(
        GAMode.GA,
        title="GA mode",
        description=(
            "ga scores every generation and optimizes the final population. "
            "ga-min optimizes every generation. "
            "ga-select scores while breeding and optimizes a diverse archive every n_prune generations."
        ),
    )
    pop_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.2
    dihedral_interval: float = 30.0
    optimization_method: GAOptimizationMethod = Field(
        GAOptimizationMethod.RDKIT_MMFF,
        title="Optimization method",
    )
    backend_options: Dict[str, Any] = {}
    max_iters: int = 500
    smart_opt: bool = False
    n_prune: int = Field(
        10,
        title="Prune every",
        description=(
            "ga-select generations between archive optimizations. "
            "The archive also updates on the first generation and at the end of the run."
        ),
    )
    prune_rms_thresh: float = Field(
        0.1,
        title="Prune RMSD",
        description=(
            "Minimum dihedral RMSD in degrees. A conformer closer than this "
            "to one already kept is dropped and the population shrinks."
        ),
    )
    db_treatment: DoubleBondTreatment = Field(
        DoubleBondTreatment.STEP_180,
        title="Double bond treatment",
        description=(
            "ignore drops double bonds. "
            "180+step adds 180 degrees to each double-bond mutation. "
            "treat-as-single mutates double bonds like single bonds."
        ),
    )
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


@simstack_model
class GAConformerParameters(Model):
    """GA conformer-generation settings, without the starting molecule.

    ``compare_optimization_methods`` supplies the molecule. Field names and
    defaults match ``GAConfig`` aside from ``initial_molecule``.
    """

    field_name: str = "GAConformerParameters"
    num_confs: int = 50
    seed: int = 1
    parallel_children: int = Field(
        0,
        title="Parallel children",
        description="RDKit force-field threads. 0 keeps RDKit's default. DFTB and xTB ignore it and evaluate each batch as one child node.",
    )
    mode: GAMode = Field(
        GAMode.GA,
        title="GA mode",
        description=(
            "ga scores every generation and optimizes the final population. "
            "ga-min optimizes every generation. "
            "ga-select scores while breeding and optimizes a diverse archive every n_prune generations."
        ),
    )
    pop_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.2
    dihedral_interval: float = 30.0
    optimization_method: GAOptimizationMethod = Field(
        GAOptimizationMethod.RDKIT_MMFF,
        title="Optimization method",
    )
    backend_options: Dict[str, Any] = {}
    max_iters: int = 500
    smart_opt: bool = False
    n_prune: int = Field(
        10,
        title="Prune every",
        description=(
            "ga-select generations between archive optimizations. "
            "The archive also updates on the first generation and at the end of the run."
        ),
    )
    prune_rms_thresh: float = Field(
        0.1,
        title="Prune RMSD",
        description=(
            "Minimum dihedral RMSD in degrees. A conformer closer than this "
            "to one already kept is dropped and the population shrinks."
        ),
    )
    db_treatment: DoubleBondTreatment = Field(
        DoubleBondTreatment.STEP_180,
        title="Double bond treatment",
        description=(
            "ignore drops double bonds. "
            "180+step adds 180 degrees to each double-bond mutation. "
            "treat-as-single mutates double bonds like single bonds."
        ),
    )
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
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = "GA conformer generation"
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        return ui
