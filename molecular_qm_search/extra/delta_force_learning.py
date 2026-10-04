import logging
import math
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pydantic import model_validator
from odmantic import Field, Model, ObjectId, Reference

from molecular_qm_models import (
    Molecule,
    MoleculeList,
    MoleculeSnapshot,
    QMInput,
    geometry_hash_from_molecule,
)
from molecular_qm_models.constants import BOHR_TO_ANGSTROM
from molecular_qm_models.energy_units import MolecularEnergyUnitEnum, convert_energy_unit
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model
from simstack.models.files import FileStack
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType

logger = logging.getLogger(__name__)

HARTREE_TO_EV = convert_energy_unit(
    MolecularEnergyUnitEnum.HARTREE, 1.0, MolecularEnergyUnitEnum.EV
)
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


def _named_value(value: Any) -> Optional[str]:
    if value is None:
        raise ValueError("QM setting value is required")
    inner = getattr(value, "value", value)
    if inner is None:
        raise ValueError("QM setting value is required")
    if hasattr(inner, "value"):
        inner = inner.value
    if inner is None:
        raise ValueError("QM setting value is required")
    return str(inner)


def qm_level_key(qm_input: QMInput) -> Tuple[str, str, str]:
    if qm_input is None:
        raise ValueError("qm_input is required")
    functional = getattr(qm_input, "functional", None)
    functional_name = _named_value(getattr(functional, "functional", functional))
    dispersion = getattr(functional, "dispersion_correction", None)
    dispersion_name = _named_value(getattr(dispersion, "value", dispersion))
    basis = getattr(qm_input, "basis_set", None)
    basis_name = _named_value(getattr(basis, "basis_set", basis))
    return functional_name, dispersion_name, basis_name


def hartree_forces_to_ev_angstrom(forces: np.ndarray) -> np.ndarray:
    if forces is None:
        raise ValueError("forces are required")
    arr = np.asarray(forces, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"forces must have shape (N, 3), got {arr.shape}")
    return arr * HARTREE_BOHR_TO_EV_ANGSTROM


def hartree_energy_to_ev(energy: float) -> float:
    if energy is None:
        raise ValueError("energy is required")
    return convert_energy_unit(
        MolecularEnergyUnitEnum.HARTREE, float(energy), MolecularEnergyUnitEnum.EV
    )


def ev_energy_to_kcal_per_mol(energy: float) -> float:
    if energy is None:
        raise ValueError("energy is required")
    hartree = convert_energy_unit(
        MolecularEnergyUnitEnum.EV, float(energy), MolecularEnergyUnitEnum.HARTREE
    )
    return convert_energy_unit(
        MolecularEnergyUnitEnum.HARTREE, hartree, MolecularEnergyUnitEnum.KCAL_PER_MOL
    )


def pair_force_snapshots(
    snapshots: Sequence[MoleculeSnapshot],
    cheap_key: Tuple[str, str, str],
    expensive_key: Tuple[str, str, str],
) -> List[Dict[str, Any]]:
    if snapshots is None:
        raise ValueError("snapshots are required")
    if cheap_key == expensive_key:
        raise ValueError("cheap and expensive QM levels must differ")
    grouped: Dict[Tuple[str, str], Dict[str, MoleculeSnapshot]] = {}
    for snapshot in snapshots:
        if not getattr(snapshot, "has_forces", False):
            continue
        geometry_hash = getattr(snapshot, "geometry_hash", None)
        smiles = getattr(snapshot, "smiles", None)
        if not geometry_hash or smiles is None:
            continue
        level = qm_level_key(snapshot.qm_input)
        bucket = grouped.setdefault((smiles, geometry_hash), {})
        bucket[level] = snapshot
    pairs = []
    for (smiles, geometry_hash), by_level in grouped.items():
        cheap = by_level.get(cheap_key)
        expensive = by_level.get(expensive_key)
        if cheap is None or expensive is None:
            logger.warning(
                "skipping unpaired geometry smiles=%s geometry_hash=%s",
                smiles,
                geometry_hash,
            )
            continue
        if cheap.energy_hartree is None or expensive.energy_hartree is None:
            raise ValueError(
                f"paired snapshots for {smiles} {geometry_hash} are missing energy_hartree"
            )
        cheap_forces = cheap.forces_hartree_bohr.array
        expensive_forces = expensive.forces_hartree_bohr.array
        if cheap_forces.shape != expensive_forces.shape:
            raise ValueError(
                f"force shape mismatch for {smiles} {geometry_hash}: "
                f"{cheap_forces.shape} vs {expensive_forces.shape}"
            )
        pairs.append(
            {
                "smiles": smiles,
                "formula": cheap.formula,
                "geometry_hash": geometry_hash,
                "molecule": cheap.molecule,
                "cheap": cheap,
                "expensive": expensive,
                "delta_energy_hartree": expensive.energy_hartree - cheap.energy_hartree,
                "delta_forces_hartree_bohr": expensive_forces - cheap_forces,
            }
        )
    return pairs


def assign_splits(
    pairs: Sequence[Dict[str, Any]],
    held_out_smiles: str,
    holdout_fraction: float,
    random_seed: int,
) -> List[Dict[str, Any]]:
    if not held_out_smiles:
        raise ValueError("held_out_smiles is required")
    if holdout_fraction is None:
        raise ValueError("holdout_fraction is required")
    if not 0.0 <= float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    assigned = []
    train_pool = []
    for pair in pairs:
        row = dict(pair)
        if pair["smiles"] == held_out_smiles:
            row["split"] = "test"
            assigned.append(row)
        else:
            train_pool.append(row)
    if not assigned:
        raise ValueError(f"no pairs found for held_out_smiles={held_out_smiles!r}")
    if not train_pool:
        raise ValueError("training split is empty; held_out_smiles cannot cover every pair")
    rng = random.Random(random_seed)
    rng.shuffle(train_pool)
    n_valid = int(math.floor(len(train_pool) * float(holdout_fraction)))
    for i, row in enumerate(train_pool):
        row["split"] = "valid" if i < n_valid else "train"
        assigned.append(row)
    return assigned


def write_delta_extxyz(path: Path, pairs: Sequence[Dict[str, Any]]) -> None:
    if path is None:
        raise ValueError("path is required")
    lines = []
    for pair in pairs:
        molecule = pair["molecule"]
        atoms = list(molecule.atoms)
        if not atoms:
            raise ValueError("molecule with atoms is required")
        cheap_f = hartree_forces_to_ev_angstrom(pair["cheap"].forces_hartree_bohr.array)
        exp_f = hartree_forces_to_ev_angstrom(pair["expensive"].forces_hartree_bohr.array)
        delta_f = hartree_forces_to_ev_angstrom(pair["delta_forces_hartree_bohr"])
        if not (cheap_f.shape == exp_f.shape == delta_f.shape == (len(atoms), 3)):
            raise ValueError("force arrays do not match atom count")
        cheap_e = hartree_energy_to_ev(pair["cheap"].energy_hartree)
        exp_e = hartree_energy_to_ev(pair["expensive"].energy_hartree)
        delta_e = hartree_energy_to_ev(pair["delta_energy_hartree"])
        lines.append(str(len(atoms)))
        lines.append(
            'Properties=species:S:1:pos:R:3:forces:R:3:cheap_forces:R:3:expensive_forces:R:3 '
            f"energy={delta_e:.12f} cheap_energy={cheap_e:.12f} expensive_energy={exp_e:.12f} "
            f'smiles={pair["smiles"]} geometry_hash={pair["geometry_hash"]} pbc="F F F"'
        )
        for atom, df, cf, ef in zip(atoms, delta_f, cheap_f, exp_f):
            lines.append(
                f"{atom.element} {atom.x:.12f} {atom.y:.12f} {atom.z:.12f} "
                f"{df[0]:.12f} {df[1]:.12f} {df[2]:.12f} "
                f"{cf[0]:.12f} {cf[1]:.12f} {cf[2]:.12f} "
                f"{ef[0]:.12f} {ef[1]:.12f} {ef[2]:.12f}"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_delta_extxyz(path: Path) -> List[Dict[str, Any]]:
    if path is None or not Path(path).is_file():
        raise ValueError("extxyz path is required")
    text = Path(path).read_text(encoding="utf-8")
    raw_lines = text.splitlines()
    frames = []
    i = 0
    while i < len(raw_lines):
        n_atoms = int(raw_lines[i])
        header = raw_lines[i + 1]
        fields = {}
        for token in header.split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value.strip('"')
        species = []
        positions = []
        delta_forces = []
        cheap_forces = []
        expensive_forces = []
        for row in raw_lines[i + 2 : i + 2 + n_atoms]:
            parts = row.split()
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            delta_forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
            cheap_forces.append([float(parts[7]), float(parts[8]), float(parts[9])])
            expensive_forces.append([float(parts[10]), float(parts[11]), float(parts[12])])
        frames.append(
            {
                "n_atoms": n_atoms,
                "energy": float(fields["energy"]),
                "cheap_energy": float(fields["cheap_energy"]),
                "expensive_energy": float(fields["expensive_energy"]),
                "smiles": fields.get("smiles"),
                "geometry_hash": fields.get("geometry_hash"),
                "species": species,
                "positions": np.asarray(positions, dtype=float),
                "delta_forces": np.asarray(delta_forces, dtype=float),
                "cheap_forces": np.asarray(cheap_forces, dtype=float),
                "expensive_forces": np.asarray(expensive_forces, dtype=float),
            }
        )
        i += 2 + n_atoms
    return frames


def delta_force_metrics(
    frames: Sequence[Dict[str, Any]],
    predicted_delta_energy: Sequence[float],
    predicted_delta_forces: Sequence[np.ndarray],
) -> Dict[str, float]:
    if not frames:
        raise ValueError("frames are required")
    if len(predicted_delta_energy) != len(frames) or len(predicted_delta_forces) != len(frames):
        raise ValueError("predictions must match the number of frames")
    e_ml = []
    e_base = []
    f_ml = []
    f_base = []
    for frame, dE, dF in zip(frames, predicted_delta_energy, predicted_delta_forces):
        if dE is None or dF is None:
            raise ValueError("predicted delta energy and forces are required")
        if frame.get("cheap_energy") is None or frame.get("expensive_energy") is None:
            raise ValueError("cheap and expensive energies are required for baseline metrics")
        if frame.get("cheap_forces") is None or frame.get("expensive_forces") is None:
            raise ValueError("cheap and expensive forces are required for baseline metrics")
        dF = np.asarray(dF, dtype=float)
        if dF.shape != frame["delta_forces"].shape:
            raise ValueError("predicted force shape does not match labels")
        e_hat = frame["cheap_energy"] + float(dE)
        f_hat = frame["cheap_forces"] + dF
        e_ml.append(e_hat - frame["expensive_energy"])
        e_base.append(frame["cheap_energy"] - frame["expensive_energy"])
        f_ml.append(f_hat - frame["expensive_forces"])
        f_base.append(frame["cheap_forces"] - frame["expensive_forces"])
    e_ml = np.asarray(e_ml, dtype=float)
    e_base = np.asarray(e_base, dtype=float)
    f_ml = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in f_ml])
    f_base = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in f_base])
    return {
        "energy_mae_kcal_ml": float(np.mean(np.abs(e_ml)) * ev_energy_to_kcal_per_mol(1.0)),
        "energy_rmse_kcal_ml": float(np.sqrt(np.mean(e_ml ** 2)) * ev_energy_to_kcal_per_mol(1.0)),
        "energy_mae_kcal_baseline": float(np.mean(np.abs(e_base)) * ev_energy_to_kcal_per_mol(1.0)),
        "energy_rmse_kcal_baseline": float(
            np.sqrt(np.mean(e_base ** 2)) * ev_energy_to_kcal_per_mol(1.0)
        ),
        "force_mae_ev_a_ml": float(np.mean(np.abs(f_ml))),
        "force_rmse_ev_a_ml": float(np.sqrt(np.mean(f_ml ** 2))),
        "force_max_ev_a_ml": float(np.max(np.abs(f_ml))),
        "force_mae_ev_a_baseline": float(np.mean(np.abs(f_base))),
        "force_rmse_ev_a_baseline": float(np.sqrt(np.mean(f_base ** 2))),
        "force_max_ev_a_baseline": float(np.max(np.abs(f_base))),
    }


def _filestack_from_path(path: Path) -> FileStack:
    return FileStack.from_local_file(path, in_memory=True, is_hashable=True)


def _validate_label_inputs(cheap_qm_input: QMInput, expensive_qm_input: QMInput) -> None:
    if cheap_qm_input is None or expensive_qm_input is None:
        raise ValueError("cheap_qm_input and expensive_qm_input are required")
    if not cheap_qm_input.gradients or not expensive_qm_input.gradients:
        raise ValueError("both QMInput templates must have gradients=True")
    if cheap_qm_input.optimization or expensive_qm_input.optimization:
        raise ValueError("labeling jobs must be single-point (optimization=False)")


def _copy_qm_input(template: QMInput, molecule: Molecule) -> QMInput:
    copied = template.model_copy()
    copied.id = ObjectId()
    copied.molecule = molecule
    copied.gradients = True
    copied.optimization = False
    return copied


@node
async def delta_force_label(
    molecules: MoleculeList,
    cheap_qm_input: QMInput,
    expensive_qm_input: QMInput,
    **kwargs,
) -> SimstackResult:
    """
    Run paired cheap/expensive Psi4 force jobs on each geometry.

    SimstackResult:
        table (SimpleTable): Per-geometry task success for cheap and expensive levels
    """
    from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
    from simstack.methods.mass_runner import MassRunner

    node_runner = kwargs.get("node_runner")
    _validate_label_inputs(cheap_qm_input, expensive_qm_input)
    if molecules is None:
        raise ValueError("molecules are required")
    mols = list(molecules)
    if not mols:
        raise ValueError("molecules are required")
    table = SimpleTable(name="Delta force labels")
    table.add_column("smiles", SimpleTableColumnType.STRING)
    table.add_column("geometry_hash", SimpleTableColumnType.STRING)
    table.add_column("cheap_success", SimpleTableColumnType.STRING)
    table.add_column("expensive_success", SimpleTableColumnType.STRING)
    async with MassRunner(psi4_calculator, **kwargs) as mass_result:
        for molecule in mols:
            if molecule is None:
                raise ValueError("molecule is required")
            geometry_hash = geometry_hash_from_molecule(molecule)
            cheap_mol = Molecule.from_molecule(molecule)
            expensive_mol = Molecule.from_molecule(molecule)
            mass_result.create_tasks(_copy_qm_input(cheap_qm_input, cheap_mol))
            mass_result.create_tasks(_copy_qm_input(expensive_qm_input, expensive_mol))
            table.add_row(
                {
                    "smiles": molecule.smiles,
                    "geometry_hash": geometry_hash,
                    "cheap_success": "submitted",
                    "expensive_success": "submitted",
                }
            )
    node_runner.table = table
    node_runner.result = mass_result.dataset
    return node_runner.succeed()


@simstack_model
class DeltaForceDatasetInput(Model):
    field_name: str = "DeltaForceDatasetInput"
    cheap_qm_input: QMInput = Reference()
    expensive_qm_input: QMInput = Reference()
    held_out_smiles: str
    holdout_fraction: float = 0.2
    random_seed: int = 0


@node
async def delta_force_dataset(opts: DeltaForceDatasetInput, **kwargs) -> SimstackResult:
    """
    Pair snapshots and write MACE extxyz splits.

    SimstackResult:
        train_xyz (FileStack): Training delta extxyz
        valid_xyz (FileStack): Validation delta extxyz
        test_xyz (FileStack): Leave-one-molecule-out test extxyz
        split_table (SimpleTable): Split membership for each pair
    """
    node_runner = kwargs.get("node_runner")
    if opts is None:
        raise ValueError("DeltaForceDatasetInput is required")
    cheap_key = qm_level_key(opts.cheap_qm_input)
    expensive_key = qm_level_key(opts.expensive_qm_input)
    snapshots = await context.db.find(MoleculeSnapshot, MoleculeSnapshot.has_forces == True)
    pairs = pair_force_snapshots(snapshots, cheap_key, expensive_key)
    if not pairs:
        raise ValueError("no paired cheap/expensive force snapshots were found")
    assigned = assign_splits(
        pairs, opts.held_out_smiles, opts.holdout_fraction, opts.random_seed
    )
    by_split = {"train": [], "valid": [], "test": []}
    split_table = SimpleTable(name="Delta force splits")
    for name, col in (
        ("smiles", SimpleTableColumnType.STRING),
        ("geometry_hash", SimpleTableColumnType.STRING),
        ("split", SimpleTableColumnType.STRING),
        ("delta_energy_hartree", SimpleTableColumnType.NUMBER),
    ):
        split_table.add_column(name, col)
    for row in assigned:
        by_split[row["split"]].append(row)
        split_table.add_row(
            {
                "smiles": row["smiles"],
                "geometry_hash": row["geometry_hash"],
                "split": row["split"],
                "delta_energy_hartree": float(row["delta_energy_hartree"]),
            }
        )
    if not by_split["train"]:
        raise ValueError("training split is empty")
    if not by_split["valid"]:
        raise ValueError(
            "validation split is empty; increase holdout_fraction or add more non-held-out geometries"
        )
    tmpdir = Path(tempfile.mkdtemp())
    train_path = tmpdir / "delta_train.xyz"
    valid_path = tmpdir / "delta_valid.xyz"
    test_path = tmpdir / "delta_test.xyz"
    write_delta_extxyz(train_path, by_split["train"])
    write_delta_extxyz(valid_path, by_split["valid"])
    write_delta_extxyz(test_path, by_split["test"])
    node_runner.train_xyz = _filestack_from_path(train_path)
    node_runner.valid_xyz = _filestack_from_path(valid_path)
    node_runner.test_xyz = _filestack_from_path(test_path)
    node_runner.split_table = split_table
    return node_runner.succeed()


@simstack_model
class MaceTrainInput(Model):
    field_name: str = "MaceTrainInput"
    train_xyz: FileStack = Reference()
    valid_xyz: FileStack = Reference()
    r_max: float = 5.0
    energy_weight: float = 1.0
    forces_weight: float = 100.0
    max_num_epochs: int = 100
    batch_size: int = 4
    hidden_irreps: str = "128x0e + 128x1o"
    seed: int = 42
    use_e0s: bool = Field(False, json_schema_extra={"title": "Pass isolated-atom E0s"})
    e0s: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_e0s(cls, data):
        if not isinstance(data, dict):
            return data
        if "use_e0s" not in data:
            data["use_e0s"] = data.get("e0s") is not None
        if not data.get("use_e0s"):
            if data.get("e0s"):
                raise ValueError("e0s must be omitted when use_e0s is false")
            data["e0s"] = None
        return data


@node
async def delta_force_train(opts: MaceTrainInput, **kwargs) -> SimstackResult:
    """
    Train MACE on delta energy and force labels.

    SimstackResult:
        model (FileStack): Trained MACE checkpoint
        log (FileStack): Trainer stdout/stderr
    """
    node_runner = kwargs.get("node_runner")
    if opts is None:
        raise ValueError("MaceTrainInput is required")
    if opts.use_e0s and not opts.e0s:
        raise ValueError("e0s is required when use_e0s is true")
    train_path = Path(opts.train_xyz.get(local_dir=Path("../../../simstack-mariana/nodes")))
    valid_path = Path(opts.valid_xyz.get(local_dir=Path("../../../simstack-mariana/nodes")))
    cmd = [
        "mace_run_train",
        "--name",
        "delta_forces",
        "--train_file",
        str(train_path),
        "--valid_file",
        str(valid_path),
        "--energy_key",
        "energy",
        "--forces_key",
        "forces",
        "--r_max",
        str(opts.r_max),
        "--energy_weight",
        str(opts.energy_weight),
        "--forces_weight",
        str(opts.forces_weight),
        "--max_num_epochs",
        str(opts.max_num_epochs),
        "--batch_size",
        str(opts.batch_size),
        "--hidden_irreps",
        opts.hidden_irreps,
        "--seed",
        str(opts.seed),
        "--device",
        "cpu",
    ]
    if opts.use_e0s:
        cmd.extend(["--E0s", opts.e0s])
    completed = subprocess.run(cmd, capture_output=True, text=True)
    log_path = Path("mace_train.log")
    log_path.write_text(
        (completed.stdout or "") + "\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    node_runner.log = _filestack_from_path(log_path)
    if completed.returncode != 0:
        raise ValueError(f"mace_run_train failed with code {completed.returncode}: {completed.stderr}")
    model_path = Path("delta_forces.model")
    if not model_path.is_file():
        staged = list(Path("../../../simstack-mariana/nodes").glob("*.model"))
        if not staged:
            raise ValueError("mace_run_train produced no .model checkpoint")
        model_path = staged[0]
    node_runner.model = _filestack_from_path(model_path)
    return node_runner.succeed()


@simstack_model
class MaceEvalInput(Model):
    field_name: str = "MaceEvalInput"
    model: FileStack = Reference()
    test_xyz: FileStack = Reference()


def _parse_mace_pred_xyz(path: Path) -> Tuple[List[float], List[np.ndarray]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = 0
    energies = []
    forces = []
    while i < len(lines):
        n_atoms = int(lines[i])
        header = lines[i + 1]
        energy = None
        for token in header.split():
            if token.startswith("energy="):
                energy = float(token.split("=", 1)[1])
        if energy is None:
            raise ValueError(f"predicted extxyz frame is missing energy at line {i + 2}")
        frame_forces = []
        for row in lines[i + 2 : i + 2 + n_atoms]:
            parts = row.split()
            frame_forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        energies.append(energy)
        forces.append(np.asarray(frame_forces, dtype=float))
        i += 2 + n_atoms
    return energies, forces


@node
async def delta_force_eval(opts: MaceEvalInput, **kwargs) -> SimstackResult:
    """
    Evaluate cheap+MACE forces against expensive labels and the cheap-only baseline.

    SimstackResult:
        metrics (SimpleTable): MAE/RMSE for energy and forces
        plot (FileStack): Force MAE comparison plot
    """
    node_runner = kwargs.get("node_runner")
    if opts is None:
        raise ValueError("MaceEvalInput is required")
    test_path = Path(opts.test_xyz.get(local_dir=Path("../../../simstack-mariana/nodes")))
    model_path = Path(opts.model.get(local_dir=Path("../../../simstack-mariana/nodes")))
    pred_path = Path("mace_pred.xyz")
    cmd = [
        "mace_eval_configs",
        "--configs",
        str(test_path),
        "--model",
        str(model_path),
        "--output",
        str(pred_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ValueError(f"mace_eval_configs failed with code {completed.returncode}: {completed.stderr}")
    frames = read_delta_extxyz(test_path)
    pred_e, pred_f = _parse_mace_pred_xyz(pred_path)
    metrics = delta_force_metrics(frames, pred_e, pred_f)
    table = SimpleTable(name="Delta force metrics")
    table.add_column("metric", SimpleTableColumnType.STRING)
    table.add_column("ml", SimpleTableColumnType.NUMBER)
    table.add_column("baseline", SimpleTableColumnType.NUMBER)
    table.add_row(
        {
            "metric": "energy_mae_kcal",
            "ml": metrics["energy_mae_kcal_ml"],
            "baseline": metrics["energy_mae_kcal_baseline"],
        }
    )
    table.add_row(
        {
            "metric": "energy_rmse_kcal",
            "ml": metrics["energy_rmse_kcal_ml"],
            "baseline": metrics["energy_rmse_kcal_baseline"],
        }
    )
    table.add_row(
        {
            "metric": "force_mae_ev_a",
            "ml": metrics["force_mae_ev_a_ml"],
            "baseline": metrics["force_mae_ev_a_baseline"],
        }
    )
    table.add_row(
        {
            "metric": "force_rmse_ev_a",
            "ml": metrics["force_rmse_ev_a_ml"],
            "baseline": metrics["force_rmse_ev_a_baseline"],
        }
    )
    table.add_row(
        {
            "metric": "force_max_ev_a",
            "ml": metrics["force_max_ev_a_ml"],
            "baseline": metrics["force_max_ev_a_baseline"],
        }
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["cheap only", "cheap+MACE"],
        [metrics["force_mae_ev_a_baseline"], metrics["force_mae_ev_a_ml"]],
    )
    ax.set_ylabel("Force MAE (eV/Å)")
    ax.set_title("Delta-learning force error")
    fig.tight_layout()
    plot_path = Path("delta_force_mae.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    node_runner.metrics = table
    node_runner.plot = _filestack_from_path(plot_path)
    return node_runner.succeed()
