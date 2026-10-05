import inspect
import logging
from typing import Dict, List, Optional, Tuple

from odmantic import Field, Model
from pydantic import model_validator

from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_psi4.models.crest_input import CrestInput
from molecular_qm_search.optimization.models.ga_models import GAConformerParameters
from simstack.core.node import node
from simstack.models import simstack_model
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

logger = logging.getLogger(__name__)


@simstack_model
class CompareConformers(Model):
    field_name: str = "CompareConformers"
    use_ga: bool = Field(
        True,
        description="Use genetic algorithm conformer generation",
        json_schema_extra={"title": "Use genetic algorithm"},
    )
    ga_config: Optional[GAConformerParameters] = Field(
        None,
        json_schema_extra={
            "title": "GA conformer generation",
            "description": "Parameters for ga_conformer_gen. The starting structure is the comparison molecule.",
        },
    )
    use_babel: bool = Field(
        True,
        description="Use OpenBabel CLI conformer search",
        json_schema_extra={"title": "Use OpenBabel"},
    )
    use_rdkit: bool = Field(
        False,
        description="Use RDKit conformer generator",
        json_schema_extra={"title": "Use RDKit"},
    )
    use_crest: bool = Field(
        False,
        description="Use CREST conformer generator",
        json_schema_extra={"title": "Use CREST"},
    )
    molecule: Optional[Molecule] = Field(
        None,
        json_schema_extra={
            "title": "Molecule",
            "description": "Starting structure for the selected generators",
        },
    )
    crest_input: Optional[CrestInput] = Field(
        None,
        json_schema_extra={
            "title": "CREST",
            "description": "CREST conformer search settings, including the starting molecule",
        },
    )
    comparison_count: int = Field(
        default=3, description="Number of relaxed conformers to compare"
    )
    rmsd_threshold: float = Field(
        default=0.5, description="RMSD threshold for matching conformers (Angstrom)"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_method_parameters(cls, data):
        if not isinstance(data, dict):
            return data
        if "use_crest" not in data:
            data["use_crest"] = data.get("crest_input") is not None
        if "use_ga" not in data and data.get("ga_config") is not None:
            data["use_ga"] = True
        if data.get("use_ga") is False:
            data["ga_config"] = None
        if data.get("use_crest"):
            data["molecule"] = None
        else:
            data["crest_input"] = None
        return data

    @model_validator(mode="after")
    def require_selected_method_parameters(self):
        if self.use_ga and self.ga_config is None:
            raise ValueError(
                "GA parameters are required when the genetic algorithm is selected"
            )
        if self.use_crest:
            if self.crest_input is None:
                raise ValueError("CREST parameters are required when CREST is selected")
        elif self.molecule is None:
            raise ValueError("A molecule is required when CREST is not selected")
        return self

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        props = schema["properties"]
        molecule_schema = props.pop("molecule", None)
        if isinstance(molecule_schema, dict) and "anyOf" in molecule_schema:
            concrete = [
                option
                for option in molecule_schema["anyOf"]
                if option.get("type") != "null"
            ]
            if len(concrete) != 1:
                raise ValueError("molecule schema must be a Molecule or null")
            molecule_schema = concrete[0]
        props.pop("ga_config", None)
        props.pop("crest_input", None)
        schema["required"] = [
            name
            for name in schema.get("required", [])
            if name not in {"molecule", "ga_config", "crest_input"}
        ]

        ga_schema = GAConformerParameters.json_schema()
        for name, definition in (ga_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        ga_schema["default"] = GAConformerParameters().model_dump(
            mode="json", exclude={"id"}
        )
        crest_schema = CrestInput.json_schema()
        for name, definition in (crest_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        crest_schema["title"] = "CREST"

        schema.setdefault("dependencies", {})["use_ga"] = {
            "oneOf": [
                {"properties": {"use_ga": {"const": False}}},
                {
                    "properties": {
                        "use_ga": {"const": True},
                        "ga_config": ga_schema,
                    },
                    "required": ["ga_config"],
                },
            ]
        }
        schema["dependencies"]["use_crest"] = {
            "oneOf": [
                {
                    "properties": {
                        "use_crest": {"const": False},
                        "molecule": molecule_schema,
                    },
                    "required": ["molecule"],
                },
                {
                    "properties": {
                        "use_crest": {"const": True},
                        "crest_input": crest_schema,
                    },
                    "required": ["crest_input"],
                },
            ]
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        for name, title in (
            ("use_ga", "Use genetic algorithm"),
            ("use_babel", "Use OpenBabel"),
            ("use_rdkit", "Use RDKit"),
            ("use_crest", "Use CREST"),
        ):
            ui[name] = {"ui:widget": "checkbox", "ui:title": title}
        ui.setdefault("ga_config", {})["ui:condition"] = {"use_ga": True}
        ui.setdefault("molecule", {})["ui:condition"] = {"use_crest": False}
        ui.setdefault("crest_input", {})["ui:condition"] = {"use_crest": True}
        return ui


async def _relax_conformers_dftb(
    conformers: MoleculeList, **kwargs
) -> List[Tuple[Molecule, float]]:
    """
    Relax each conformer using DFTB with max 1000 iterations.
    Returns a list of (relaxed_molecule, energy_hartree) sorted by energy ascending.
    """
    from molecular_qm_dftb.models.dftb_input import DftbInput
    from molecular_qm_dftb.nodes.dftb_list_calculator import dftb_list_calculator
    from simstack.core.context import context

    dftb_opts = DftbInput(optimization=True, max_optimization_steps=1000)
    calc_result = await dftb_list_calculator(conformers, dftb_opts, **kwargs)

    dataset = getattr(calc_result, "dataset", None)
    if dataset is None and hasattr(calc_result, "sections"):
        dataset = calc_result
    if dataset is None:
        node_runner = kwargs.get("node_runner")
        dataset = None if node_runner is None else getattr(node_runner, "dataset", None)
    if dataset is None:
        raise ValueError("dftb_list_calculator did not return a dataset")
    if "results" not in dataset:
        raise ValueError("dftb_list_calculator dataset has no results section")

    results = dataset["results"]
    if (
        hasattr(results, "data")
        and results.data
        and hasattr(results, "load_to_cache")
        and len(results) == 0
    ):
        await results.load_to_cache(context.db)

    relaxed_list: List[Tuple[Molecule, float]] = []
    for _row_name, row in results.items():
        success = row.get("success")
        if success is None:
            raise ValueError("DFTB result row is missing success")
        success_value = success.value if hasattr(success, "value") else success
        if not success_value:
            error = row.get("error")
            if error is None:
                logger.warning("DFTB relaxation failed for a conformer")
            else:
                error_msg = error.value if hasattr(error, "value") else error
                logger.warning(f"DFTB relaxation failed for a conformer: {error_msg}")
            continue

        qm_res = row.get("result_qm_result")
        opt_mol = None if qm_res is None else getattr(qm_res, "final_structure", None)
        if opt_mol is None:
            raise ValueError("DFTB result row is missing final_structure")

        energy = None if qm_res is None else getattr(qm_res, "final_energy", None)
        if energy is None:
            energy_hartree = row.get("result_energy_hartree")
            if energy_hartree is None:
                raise ValueError("DFTB result row is missing energy")
            energy = getattr(energy_hartree, "value", energy_hartree)
        if energy is None:
            raise ValueError("DFTB result row is missing energy")

        energy = float(energy)
        opt_mol.properties["energy"] = energy
        relaxed_list.append((opt_mol, energy))

    relaxed_list.sort(key=lambda item: item[1])
    return relaxed_list


@node
async def compare_optimization_methods(
    opts: CompareConformers,
    **kwargs,
) -> MoleculeList:
    """
    Compare up to 4 conformer generators (GA, OpenBabel, RDKit, CREST),
    optimize conformers using DFTB (max 1000 iterations), compute an NxN
    aligned RMSD matrix between lowest relaxed conformers, and prepare a SimpleTable
    of conformer ranks.

    The genetic algorithm runs ``run_ga_conformer_gen``. Its parameters are
    ``ga_config`` and are used only when the genetic algorithm is selected.
    CREST parameters are used only when CREST is selected. Otherwise every
    generator starts from ``molecule``.

    Called Nodes:
        run_ga_conformer_gen
        conformers_openbabel
        conformers_rdkit
        crest
        dftb_list_calculator
    """
    node_runner = kwargs.get("node_runner")

    if opts.use_crest:
        if opts.crest_input is None:
            raise ValueError("CREST parameters are required when CREST is selected")
        input_molecule = opts.crest_input.molecule
    else:
        input_molecule = opts.molecule
    if input_molecule is None or len(input_molecule.atoms) == 0:
        if node_runner is not None:
            node_runner.fail("No input molecule provided for compare_conformers.")
        raise ValueError("No input molecule provided for compare_conformers.")

    methods_to_run: List[Tuple[str, str]] = []
    if opts.use_ga:
        methods_to_run.append(("GA", "ga"))
    if opts.use_babel:
        methods_to_run.append(("Babel", "babel"))
    if opts.use_rdkit:
        methods_to_run.append(("RDKit", "rdkit"))
    if opts.use_crest:
        methods_to_run.append(("CREST", "crest"))

    if not methods_to_run:
        if node_runner is not None:
            node_runner.warning("No conformer generation method selected.")
        return MoleculeList()

    if node_runner is not None:
        node_runner.info(
            f"Running conformer generators: {[m[0] for m in methods_to_run]} "
            f"with comparison_count={opts.comparison_count}"
        )

    from molecular_qm_models.align_molecules import align_molecules
    from molecular_qm_psi4.nodes.crest import crest
    from molecular_qm_search.optimization.ga import run_ga_conformer_gen
    from molecular_qm_search.optimization.models.ga_models import GAConfig
    from molecular_qm_util.obabel_scripts.openbabel_conformers import (
        ConformerGenerationInput,
        conformers_openbabel,
    )
    from molecular_qm_util.rdkit_scripts.rdkit_conformers import conformers_rdkit

    # 1. Run each conformer generation method
    generated_by_method: Dict[str, MoleculeList] = {}

    for method_label, method_key in methods_to_run:
        try:
            if method_key == "ga":
                if opts.ga_config is None:
                    raise ValueError(
                        "GA parameters are required when the genetic algorithm is selected"
                    )
                ga_parameters = {
                    name: getattr(opts.ga_config, name)
                    for name in type(opts.ga_config).model_fields
                    if name not in {"field_name", "id"}
                }
                ga_config = GAConfig(initial_molecule=input_molecule, **ga_parameters)
                ga_res = await run_ga_conformer_gen(ga_config, **dict(kwargs))
                if isinstance(ga_res, MoleculeList):
                    confs = ga_res
                else:
                    confs = None if ga_res is None else getattr(ga_res, "molecules", None)
                if confs is None:
                    raise ValueError("run_ga_conformer_gen did not return molecules")
                generated_by_method[method_label] = confs

            elif method_key == "babel":
                cg_input = ConformerGenerationInput(num_confs=50)
                confs = conformers_openbabel(input_molecule, cg_input, **dict(kwargs))
                if inspect.isawaitable(confs):
                    confs = await confs
                generated_by_method[method_label] = confs

            elif method_key == "rdkit":
                cg_input = ConformerGenerationInput(num_confs=50)
                confs = conformers_rdkit(input_molecule, cg_input, **dict(kwargs))
                if inspect.isawaitable(confs):
                    confs = await confs
                generated_by_method[method_label] = confs

            elif method_key == "crest":
                crest_res = await crest(opts.crest_input, **dict(kwargs))
                confs = getattr(crest_res, "molecule_list", None)
                if confs is None and node_runner is not None:
                    confs = getattr(node_runner, "crest_result", None)
                if confs is None:
                    confs = MoleculeList()
                generated_by_method[method_label] = confs

        except Exception as exc:
            logger.error(f"Failed generating conformers for method {method_label}: {exc}")
            if node_runner is not None:
                node_runner.warning(f"Failed generating conformers for {method_label}: {exc}")
            generated_by_method[method_label] = MoleculeList()

    # 2. Relax conformers with DFTB (1000 max iterations)
    relaxed_by_method: Dict[str, List[Tuple[Molecule, float]]] = {}
    for method_label, confs in generated_by_method.items():
        if len(confs) > 0:
            if node_runner is not None:
                node_runner.info(f"Relaxing {len(confs)} conformers for {method_label} with DFTB...")
            relaxed_sorted = await _relax_conformers_dftb(confs, **kwargs)
            relaxed_by_method[method_label] = relaxed_sorted
        else:
            relaxed_by_method[method_label] = []

    active_methods = [m[0] for m in methods_to_run if len(relaxed_by_method.get(m[0], [])) > 0]
    if not active_methods:
        if node_runner is not None:
            node_runner.warning("No conformers generated or relaxed successfully.")
        return MoleculeList()

    # 3. Sort all conformers by energy and consider the list up to 2xcomparison_count
    all_conformers: List[Dict[str, object]] = []
    for method_label in active_methods:
        for rank_in_method, (mol, energy) in enumerate(relaxed_by_method[method_label], start=1):
            all_conformers.append(
                {
                    "molecule": mol,
                    "method": method_label,
                    "rank_in_method": rank_in_method,
                    "energy": energy,
                }
            )

    all_conformers.sort(key=lambda x: x["energy"])
    top_limit = 2 * opts.comparison_count
    top_conformers = all_conformers[:top_limit]

    # 4. Prepare SimpleTable comparing each conformer with conformers from each method
    comparison_table = SimpleTable(name="Conformer Comparison Table")
    comparison_table.add_column("Conformer", SimpleTableColumnType.NUMBER)
    for m in active_methods:
        comparison_table.add_column(m, SimpleTableColumnType.STRING)

    for conf_idx, item in enumerate(top_conformers, start=1):
        target_mol = item["molecule"]
        target_method = item["method"]
        target_rank = item["rank_in_method"]

        row_data: Dict[str, object] = {"Conformer": conf_idx}

        for m in active_methods:
            if m == target_method:
                row_data[m] = f"{target_rank}/0.0"
            else:
                method_conformers = relaxed_by_method.get(m, [])
                best_rmsd = float("inf")
                best_rank = None

                for k, (cand_mol, _) in enumerate(method_conformers, start=1):
                    try:
                        _, _, rmsd = align_molecules(target_mol, cand_mol)
                        if rmsd < best_rmsd:
                            best_rmsd = rmsd
                            best_rank = k
                    except Exception as exc:
                        logger.warning(f"Alignment failed between conformers: {exc}")

                if best_rank is not None and best_rmsd <= opts.rmsd_threshold:
                    row_data[m] = f"{best_rank}/{round(best_rmsd, 4)}"
                else:
                    row_data[m] = "NA/NA"

        comparison_table.add_row(row_data)

    best_conformers = MoleculeList()
    for global_rank, item in enumerate(top_conformers, start=1):
        mol = item["molecule"]
        mol.properties["rank_id"] = global_rank
        mol.properties["method"] = item["method"]
        best_conformers.append(mol)

    if node_runner is not None:
        node_runner.table = comparison_table
        node_runner.comparison_table = comparison_table
        node_runner.molecule_list = best_conformers
        node_runner.info(f"Compare conformers finished with {len(best_conformers)} best conformers.")

    return best_conformers