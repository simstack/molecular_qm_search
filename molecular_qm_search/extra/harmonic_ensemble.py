import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from odmantic import Model, ObjectId, Reference

from molecular_qm_models import Molecule, MoleculeList, QMInput
from molecular_qm_models.constants import BOHR_TO_ANGSTROM
from molecular_qm_psi4.util.frequency_table import FREQ_ZERO_CM1, signed_wavenumber_cm1
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model
from simstack.models.simple_table import SimpleTable

logger = logging.getLogger(__name__)

# CODATA 2018. Masses in m_e; energy in hartree; ħ = 1.
HARTREE_TO_CM1 = 219474.63136320
K_B_HARTREE_PER_K = 3.166811563455561e-6
AMU_TO_AU = 1822.888486209


def _datum_array(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("frequency-analysis quantity is required")
    if isinstance(value, dict) and value.get("__vibdatum__"):
        data = value.get("data")
    else:
        data = getattr(value, "data", value)
    if data is None:
        raise ValueError("frequency-analysis quantity is missing data")
    return np.asarray(data, dtype=float)


def harmonic_sigma_au(omega_cm1: float, temperature_K: float) -> float:
    """Mass-weighted HO width (bohr * sqrt(m_e)) at temperature T."""
    if omega_cm1 is None or temperature_K is None:
        raise ValueError("frequency and temperature are required")
    if float(temperature_K) <= 0.0:
        raise ValueError("temperature_K must be > 0")
    omega = float(omega_cm1) / HARTREE_TO_CM1
    if omega <= 0.0:
        raise ValueError(f"vibrational frequency must be positive, got {omega_cm1} cm^-1")
    half_beta_hw = omega / (2.0 * K_B_HARTREE_PER_K * float(temperature_K))
    if half_beta_hw == 0.0:
        raise ValueError("coth argument must be nonzero")
    return float(np.sqrt((1.0 / (2.0 * omega)) * (np.cosh(half_beta_hw) / np.sinh(half_beta_hw))))


def cartesian_modes_mass_normalized(
    cartesian_modes: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Scale Cartesian modes so each column satisfies L^T M L = 1 (M in m_e)."""
    if cartesian_modes is None or masses_amu is None:
        raise ValueError("cartesian modes and masses are required")
    modes = np.asarray(cartesian_modes, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    if modes.ndim != 2:
        raise ValueError(f"cartesian modes must be 2-D, got shape {modes.shape}")
    n_cart, n_modes = modes.shape
    if n_cart % 3 != 0:
        raise ValueError(f"cartesian mode length must be 3N, got {n_cart}")
    n_atoms = n_cart // 3
    if masses.shape != (n_atoms,):
        raise ValueError(f"masses must have shape ({n_atoms},), got {masses.shape}")
    if np.any(masses <= 0.0):
        raise ValueError("atomic masses must be positive")
    mass_au = np.repeat(masses * AMU_TO_AU, 3)
    normalized = np.empty_like(modes)
    for k in range(n_modes):
        vec = modes[:, k]
        mw = float(np.dot(vec * mass_au, vec))
        if mw <= 0.0:
            raise ValueError(f"mode {k} has zero mass-weighted norm")
        normalized[:, k] = vec / np.sqrt(mw)
    return normalized


def vibrational_mode_indices(wavenumbers_cm1: Sequence[float]) -> List[int]:
    if wavenumbers_cm1 is None:
        raise ValueError("wavenumbers are required")
    values = [signed_wavenumber_cm1(freq) for freq in wavenumbers_cm1]
    if not values:
        raise ValueError("wavenumbers are empty")
    imaginary = [i for i, freq in enumerate(values) if freq < -FREQ_ZERO_CM1]
    if imaginary:
        details = ", ".join(f"mode {i + 1}={values[i]:.2f} cm^-1" for i in imaginary)
        raise ValueError(f"imaginary frequencies; geometry is not a minimum: {details}")
    kept = [i for i, freq in enumerate(values) if freq > FREQ_ZERO_CM1]
    if not kept:
        raise ValueError("no real vibrational modes above the trans/rot cutoff")
    return kept


def sample_harmonic_displacements(
    modes_mass_normalized: np.ndarray,
    wavenumbers_cm1: Sequence[float],
    mode_indices: Sequence[int],
    temperature_K: float,
    n_conformers: int,
    random_seed: int,
) -> np.ndarray:
    """Return Cartesian displacements in Å, shape (n_conformers, n_atoms, 3)."""
    if n_conformers is None or int(n_conformers) < 1:
        raise ValueError("n_conformers must be >= 1")
    if random_seed is None:
        raise ValueError("random_seed is required")
    modes = np.asarray(modes_mass_normalized, dtype=float)
    n_cart, n_modes = modes.shape
    values = [signed_wavenumber_cm1(freq) for freq in wavenumbers_cm1]
    if len(values) != n_modes:
        raise ValueError(
            f"mode count {n_modes} does not match {len(values)} frequencies"
        )
    n_atoms = n_cart // 3
    rng = np.random.default_rng(int(random_seed))
    displacements_bohr = np.zeros((int(n_conformers), n_cart), dtype=float)
    for index in mode_indices:
        sigma = harmonic_sigma_au(values[index], temperature_K)
        amplitudes = rng.normal(0.0, sigma, size=int(n_conformers))
        displacements_bohr += amplitudes[:, None] * modes[:, index]
    return displacements_bohr.reshape(int(n_conformers), n_atoms, 3) * BOHR_TO_ANGSTROM


def _modes_from_frequency_analysis(vibinfo: Dict[str, Any], n_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    if not vibinfo:
        raise ValueError("frequency_analysis is required")
    omega = _datum_array(vibinfo.get("omega"))
    wavenumbers = np.asarray([signed_wavenumber_cm1(freq) for freq in omega.reshape(-1)], dtype=float)
    cart = None
    for key in ("x", "q"):
        if key in vibinfo and vibinfo[key] is not None:
            cart = _datum_array(vibinfo[key])
            break
    if cart is None:
        raise ValueError("frequency_analysis is missing Cartesian modes ('x' or 'q')")
    cart = np.asarray(cart, dtype=float)
    n_cart = 3 * n_atoms
    if cart.ndim == 1:
        raise ValueError(f"Cartesian modes must be 2-D, got shape {cart.shape}")
    if cart.shape[0] == n_cart:
        modes = cart
    elif cart.shape[1] == n_cart:
        modes = cart.T
    else:
        raise ValueError(
            f"Cartesian modes shape {cart.shape} is incompatible with 3N={n_cart}"
        )
    if modes.shape[1] != len(wavenumbers):
        if modes.shape[1] > len(wavenumbers):
            modes = modes[:, : len(wavenumbers)]
        else:
            raise ValueError(
                f"mode columns {modes.shape[1]} vs frequencies {len(wavenumbers)}"
            )
    return wavenumbers, modes


def frequency_analysis_from_wfn_npy(path: Path) -> Dict[str, Any]:
    if path is None or not Path(path).is_file():
        raise ValueError("wavefunction npy path is required")
    payload = np.load(str(path), allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError("wavefunction payload must be a dict")
    vibinfo = payload.get("frequency_analysis")
    if not vibinfo:
        raise ValueError("wavefunction has no frequency_analysis")
    return vibinfo


def _masses_amu(molecule: Molecule) -> np.ndarray:
    try:
        from rdkit.Chem import GetPeriodicTable
    except Exception as exc:
        raise ValueError("RDKit is required to look up atomic masses") from exc
    table = GetPeriodicTable()
    masses = []
    for atom in molecule.atoms:
        element = getattr(atom, "element", None)
        if not element:
            raise ValueError("atom element is required")
        mass = float(table.GetAtomicWeight(str(element)))
        if mass <= 0.0:
            raise ValueError(f"atomic mass for {element!r} is missing")
        masses.append(mass)
    return np.asarray(masses, dtype=float)


def _coords_angstrom(molecule: Molecule) -> np.ndarray:
    atoms = list(molecule.atoms)
    if not atoms:
        raise ValueError("molecule with atoms is required")
    coords = []
    for atom in atoms:
        if atom.x is None or atom.y is None or atom.z is None:
            raise ValueError("atom x, y, z coordinates are required")
        coords.append([float(atom.x), float(atom.y), float(atom.z)])
    return np.asarray(coords, dtype=float)


def molecules_from_displacements(
    source: Molecule,
    displacements_angstrom: np.ndarray,
) -> List[Molecule]:
    coords = _coords_angstrom(source)
    disp = np.asarray(displacements_angstrom, dtype=float)
    if disp.ndim != 3 or disp.shape[1:] != coords.shape:
        raise ValueError(
            f"displacements must have shape (N, {coords.shape[0]}, 3), got {disp.shape}"
        )
    molecules = []
    for i, shift in enumerate(disp):
        mol = Molecule.from_molecule(source)
        mol.smiles = source.smiles
        mol.formula = source.formula
        new_coords = coords + shift
        for atom, xyz in zip(mol.atoms, new_coords):
            atom.x, atom.y, atom.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        mol.properties["ensemble_index"] = i
        molecules.append(mol)
    return molecules


def _find_wfn_file(result) -> Path:
    candidates = []
    for container in (
        getattr(result, "files", None),
        getattr(getattr(result, "qm_result", None), "files", None),
    ):
        if container is None:
            continue
        for fs in container:
            name = getattr(fs, "name", "") or ""
            if name.endswith(".wfn.npy") or name.endswith(".wfn") or name == "result.wfn.npy":
                candidates.append(fs)
    if not candidates:
        raise ValueError("frequency job produced no wavefunction file")
    return Path(candidates[0].get(local_dir=Path("../../../simstack-mariana/nodes")))


@simstack_model
class HarmonicEnsembleInput(Model):
    field_name: str = "HarmonicEnsembleInput"
    smiles: str
    qm_input: QMInput = Reference()
    n_conformers: int
    temperature_K: float
    random_seed: int


@node
async def harmonic_ensemble(opts: HarmonicEnsembleInput, **kwargs) -> SimstackResult:
    """
    Optimize a SMILES geometry, compute harmonic frequencies, and sample a
    near-equilibrium Boltzmann ensemble.

    SimstackResult:
        molecules (MoleculeList): N geometries sampled from the thermal harmonic Wigner distribution
        vibrational_frequencies (SimpleTable): Harmonic frequencies (cm^-1) from the frequency job
    """
    from molecular_qm_psi4.nodes.psi4_calculator import psi4_calculator
    from molecular_qm_util.rdkit_scripts.smiles_to_molecule import smiles_to_molecule

    node_runner = kwargs.get("node_runner")
    if opts is None:
        raise ValueError("HarmonicEnsembleInput is required")
    if not opts.smiles:
        raise ValueError("smiles is required")
    if opts.qm_input is None:
        raise ValueError("qm_input is required")
    if opts.n_conformers is None or int(opts.n_conformers) < 1:
        raise ValueError("n_conformers must be >= 1")
    if opts.temperature_K is None or float(opts.temperature_K) <= 0.0:
        raise ValueError("temperature_K must be > 0")
    if opts.random_seed is None:
        raise ValueError("random_seed is required")

    molecule = smiles_to_molecule(opts.smiles)
    molecule.smiles = opts.smiles
    copied = opts.qm_input.model_copy()
    copied.id = ObjectId()
    copied.molecule = molecule
    copied.optimization = True
    copied.frequencies = True
    freq_result = await psi4_calculator(copied, **kwargs)
    error_message = getattr(freq_result, "error_message", None)
    if error_message:
        raise ValueError(error_message)
    qm_result = getattr(freq_result, "qm_result", None)
    optimized = None if qm_result is None else getattr(qm_result, "final_structure", None)
    if optimized is None or not list(optimized.atoms):
        raise ValueError("frequency job did not return an optimized structure")
    optimized.smiles = opts.smiles
    if optimized.formula is None:
        optimized.formula = molecule.formula

    wfn_path = _find_wfn_file(freq_result)
    vibinfo = frequency_analysis_from_wfn_npy(wfn_path)
    wavenumbers, raw_modes = _modes_from_frequency_analysis(vibinfo, len(list(optimized.atoms)))
    kept = vibrational_mode_indices(wavenumbers)
    masses = _masses_amu(optimized)
    modes = cartesian_modes_mass_normalized(raw_modes, masses)
    displacements = sample_harmonic_displacements(
        modes,
        wavenumbers,
        kept,
        opts.temperature_K,
        opts.n_conformers,
        opts.random_seed,
    )
    sampled = molecules_from_displacements(optimized, displacements)
    mol_list = MoleculeList()
    for mol in sampled:
        mol.properties["temperature_K"] = float(opts.temperature_K)
        mol_list.add_molecule(mol)
    node_runner.molecules = mol_list
    freq_table = getattr(freq_result, "vibrational_frequencies", None)
    if freq_table is None and qm_result is not None:
        freq_table = getattr(qm_result, "vibrational_frequencies", None)
    if freq_table is not None:
        node_runner.vibrational_frequencies = freq_table
    return node_runner.succeed()
