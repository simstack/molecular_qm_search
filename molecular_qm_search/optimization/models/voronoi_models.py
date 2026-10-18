from typing import Any, Dict

from odmantic import Field, Model, Reference
from simstack.models import simstack_model
from simstack.util.generate_ui_schema import generate_ui_schema

from molecular_qm_models import Molecule

from molecular_qm_search.optimization.models.ga_models import (
    DoubleBondTreatment,
    GAOptimizationMethod,
)


@simstack_model
class VoronoiConfig(Model):
    """Dihedral search whose crossover follows weighted Voronoi neighbor edges."""

    field_name: str = "VoronoiConfig"
    initial_molecule: Molecule = Reference()

    num_confs: int = 50
    seed: int = 1
    parallel_children: int = Field(
        0,
        title="Parallel children",
        description="RDKit force-field threads. 0 keeps RDKit's default. DFTB and xTB ignore it and evaluate each batch as one child node.",
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
    prune_rms_thresh: float = Field(
        20.0,
        title="Prune RMSD",
        description=(
            "Minimum dihedral RMSD in degrees between conformers that are both kept. "
            "Each generation scores new children and keeps the lower-energy conformer "
            "when one is closer than this to a parent or another child."
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
    weight_temperature: float = Field(
        1.0,
        title="Weight temperature",
        description=(
            "Positive energy scale for site weights exp(-(energy - min_energy) / temperature). "
            "A lower energy gives a larger weight, which pushes the Voronoi boundary farther "
            "from that conformer."
        ),
    )
    n_witnesses: int = Field(
        4096,
        title="Witness samples",
        description=(
            "Uniform samples on the dihedral box used to detect neighbor edges. "
            "A face thinner than the sample spacing can be missed."
        ),
    )
    neighbor_gap: float = Field(
        0.05,
        title="Neighbor gap",
        description=(
            "Relative tolerance on weighted distance. A witness votes for an edge when "
            "its second-best weighted distance is at most (1 + gap) times the best and "
            "the third-best is strictly worse."
        ),
    )

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        return ui
