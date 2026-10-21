import inspect
import logging
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

from odmantic import Field, Model
from pydantic import model_validator

from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_models import Molecule, MoleculeList
from molecular_qm_psi4.models.crest_input import CrestInput, CrestLevelOfTheory
from molecular_qm_search.optimization.models.ga_models import GAConformerParameters
from simstack.core.node import node
from simstack.models import simstack_model
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

logger = logging.getLogger(__name__)


class ConformerRelaxationMethod(str, Enum):
    """List calculator used to relax the generated conformers."""

    OFF = "off"
    XTB = "xtb"
    DFTB = "dftb"


@simstack_model
class XTBListCalculatorParameters(Model):
    """xTB list-calculator settings, without the molecule list.

    ``compare_optimization_methods`` supplies the generated conformers.
    Field names match ``XTBInput`` aside from ``molecules``.
    """

    field_name: str = "XTBListCalculatorParameters"
    level_of_theory: CrestLevelOfTheory = Field(default_factory=CrestLevelOfTheory)
    compute_gradients: bool = Field(False, description="Whether to compute gradients")
    optimize: bool = Field(True, description="Whether to perform geometry optimization")
    additional_keywords: Optional[str] = Field(
        None, description="Any additional CREST command line arguments"
    )

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = "xTB list calculator"
        keywords = schema["properties"].get("additional_keywords")
        if isinstance(keywords, dict) and "anyOf" in keywords:
            for option in keywords["anyOf"]:
                if option.get("type") == "string":
                    option["title"] = "Additional Keywords"
                elif option.get("type") == "null":
                    option["title"] = "No Additional Keywords"
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        ui["ui:order"] = [
            "level_of_theory",
            "compute_gradients",
            "optimize",
            "additional_keywords",
        ]
        return ui


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
        default=3, description="Number of conformers to compare"
    )
    rmsd_threshold: float = Field(
        default=0.5, description="RMSD threshold for matching conformers (Angstrom)"
    )
    relaxation_method: ConformerRelaxationMethod = Field(
        ConformerRelaxationMethod.DFTB,
        json_schema_extra={
            "title": "Final relaxation",
            "description": "Relax generated conformers with xTB or DFTB, or select off to use existing energies",
        },
    )
    xtb_parameters: Optional[XTBListCalculatorParameters] = Field(
        None,
        json_schema_extra={
            "title": "xTB list calculator",
            "description": "Parameters for xtb_optimize_molecule_list. The molecules are the generated conformers.",
        },
    )
    dftb_parameters: Optional[DftbInput] = Field(
        None,
        json_schema_extra={
            "title": "DFTB list calculator",
            "description": "Parameters for dftb_list_calculator. The molecules are the generated conformers.",
        },
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
        if "relaxation_method" not in data:
            method = ConformerRelaxationMethod.DFTB
        else:
            method = data["relaxation_method"]
            if method is None:
                raise ValueError("relaxation_method is required")
            if isinstance(method, str):
                try:
                    method = ConformerRelaxationMethod(method)
                except ValueError as exc:
                    raise ValueError("relaxation_method must be off, xtb or dftb") from exc
            if method not in tuple(ConformerRelaxationMethod):
                raise ValueError("relaxation_method must be off, xtb or dftb")
        if method == ConformerRelaxationMethod.OFF:
            data["xtb_parameters"] = None
            data["dftb_parameters"] = None
        elif method == ConformerRelaxationMethod.XTB:
            data["dftb_parameters"] = None
        else:
            data["xtb_parameters"] = None
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
        if self.relaxation_method == ConformerRelaxationMethod.XTB:
            if self.xtb_parameters is None:
                raise ValueError(
                    "xTB list calculator parameters are required when xTB relaxation is selected"
                )
            if not self.xtb_parameters.optimize:
                raise ValueError("xTB optimization must be enabled to relax conformers")
        elif self.relaxation_method == ConformerRelaxationMethod.DFTB:
            if self.dftb_parameters is None:
                raise ValueError(
                    "DFTB list calculator parameters are required when DFTB relaxation is selected"
                )
            if not self.dftb_parameters.optimization:
                raise ValueError("DFTB optimization must be enabled to relax conformers")
        elif self.relaxation_method != ConformerRelaxationMethod.OFF:
            raise ValueError("relaxation_method must be off, xtb or dftb")
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
        props.pop("xtb_parameters", None)
        props.pop("dftb_parameters", None)
        schema["required"] = [
            name
            for name in schema.get("required", [])
            if name
            not in {
                "molecule",
                "ga_config",
                "crest_input",
                "xtb_parameters",
                "dftb_parameters",
            }
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

        xtb_schema = XTBListCalculatorParameters.json_schema()
        for name, definition in (xtb_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        xtb_schema["default"] = XTBListCalculatorParameters().model_dump(
            mode="json", exclude={"id"}
        )
        dftb_schema = DftbInput.json_schema()
        for name, definition in (dftb_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        dftb_schema["title"] = "DFTB list calculator"
        dftb_schema["default"] = DftbInput(
            optimization=True, max_optimization_steps=1000
        ).model_dump(mode="json", exclude={"id"})

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
        schema["dependencies"]["relaxation_method"] = {
            "oneOf": [
                {
                    "properties": {
                        "relaxation_method": {"const": ConformerRelaxationMethod.OFF.value},
                    },
                },
                {
                    "properties": {
                        "relaxation_method": {"const": ConformerRelaxationMethod.XTB.value},
                        "xtb_parameters": xtb_schema,
                    },
                    "required": ["xtb_parameters"],
                },
                {
                    "properties": {
                        "relaxation_method": {"const": ConformerRelaxationMethod.DFTB.value},
                        "dftb_parameters": dftb_schema,
                    },
                    "required": ["dftb_parameters"],
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
        ui.setdefault("xtb_parameters", {})["ui:condition"] = {
            "relaxation_method": ConformerRelaxationMethod.XTB.value
        }
        ui.setdefault("dftb_parameters", {})["ui:condition"] = {
            "relaxation_method": ConformerRelaxationMethod.DFTB.value
        }
        return ui


async def _relax_conformers(
    conformers: MoleculeList, opts: CompareConformers, **kwargs
) -> List[Tuple[Molecule, float]]:
    """
    Relax each conformer with the selected list calculator.
    With off, use existing energies without changing conformer geometries.
    Returns (molecule, energy) pairs sorted by energy ascending.
    """
    if opts.relaxation_method == ConformerRelaxationMethod.OFF:
        ranked = []
        for index, molecule in enumerate(conformers, start=1):
            raw_energy = molecule.properties.get("energy")
            try:
                energy = float(raw_energy)
                if isinstance(raw_energy, bool) or not math.isfinite(energy):
                    raise ValueError("energy must be finite and numeric")
            except (TypeError, ValueError, OverflowError):
                message = (
                    f"Skipping conformer {index}: missing or invalid existing energy "
                    "with final relaxation off."
                )
                logger.warning(message)
                node_runner = kwargs.get("node_runner")
                if node_runner is not None:
                    node_runner.warning(message)
                continue
            ranked.append((molecule, energy))
        ranked.sort(key=lambda item: item[1])
        return ranked

    if opts.relaxation_method == ConformerRelaxationMethod.XTB:
        from molecular_qm_psi4.models.crest_input import XTBInput
        from molecular_qm_psi4.nodes.crest import xtb_optimize_molecule_list

        if opts.xtb_parameters is None:
            raise ValueError(
                "xTB list calculator parameters are required when xTB relaxation is selected"
            )
        xtb_input = XTBInput(
            molecules=conformers,
            level_of_theory=opts.xtb_parameters.level_of_theory,
            compute_gradients=opts.xtb_parameters.compute_gradients,
            optimize=opts.xtb_parameters.optimize,
            additional_keywords=opts.xtb_parameters.additional_keywords,
        )
        calc_result = await xtb_optimize_molecule_list(xtb_input, **kwargs)
        if getattr(calc_result, "error_message", None):
            raise ValueError(
                f"xtb_optimize_molecule_list failed: {calc_result.error_message}"
            )
        molecules_out = getattr(calc_result, "molecule_list", None)
        if molecules_out is None:
            node_runner = kwargs.get("node_runner")
            molecules_out = (
                None if node_runner is None else getattr(node_runner, "molecule_list", None)
            )
        if molecules_out is None:
            raise ValueError("xtb_optimize_molecule_list did not return molecules")

        relaxed_list: List[Tuple[Molecule, float]] = []
        for molecule in molecules_out:
            energy = molecule.properties.get("energy")
            if energy is None:
                logger.warning("xTB relaxation failed for a conformer")
                continue
            energy = float(energy)
            molecule.properties["energy"] = energy
            relaxed_list.append((molecule, energy))
        relaxed_list.sort(key=lambda item: item[1])
        return relaxed_list

    if opts.relaxation_method != ConformerRelaxationMethod.DFTB:
        raise ValueError("relaxation_method must be off, xtb or dftb")
    if opts.dftb_parameters is None:
        raise ValueError(
            "DFTB list calculator parameters are required when DFTB relaxation is selected"
        )

    from molecular_qm_dftb.nodes.dftb_list_calculator import dftb_list_calculator
    from simstack.core.context import context

    calc_result = await dftb_list_calculator(conformers, opts.dftb_parameters, **kwargs)

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
    optionally relax generated conformers with the xTB or DFTB list calculator,
    compare the lowest-energy conformers by aligned RMSD, and
    prepare a SimpleTable of conformer ranks.

    The genetic algorithm runs ``run_ga_conformer_gen``. Its parameters are
    ``ga_config`` and are used only when the genetic algorithm is selected.
    CREST parameters are used only when CREST is selected. Otherwise every
    generator starts from ``molecule``.

    ``relaxation_method`` selects the final relaxation. xTB exposes
    ``xtb_parameters`` for ``xtb_optimize_molecule_list``. DFTB exposes
    ``dftb_parameters`` for ``dftb_list_calculator``. Off skips both calculators
    and ranks using existing energies as supplied, without unit conversion.

    Called Nodes:
        run_ga_conformer_gen
        conformers_openbabel
        conformers_rdkit
        crest
        dftb_list_calculator
        xtb_optimize_molecule_list
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

    # 2. Relax conformers with the selected list calculator
    relaxed_by_method: Dict[str, List[Tuple[Molecule, float]]] = {}
    for method_label, confs in generated_by_method.items():
        if len(confs) > 0:
            if node_runner is not None and opts.relaxation_method == ConformerRelaxationMethod.OFF:
                node_runner.info(
                    f"Skipping final relaxation for {method_label}; using existing energies."
                )
            elif node_runner is not None:
                node_runner.info(
                    f"Relaxing {len(confs)} conformers for {method_label} "
                    f"with {opts.relaxation_method.value}..."
                )
            relaxed_sorted = await _relax_conformers(confs, opts, **kwargs)
            relaxed_by_method[method_label] = relaxed_sorted
        else:
            relaxed_by_method[method_label] = []

    active_methods = [m[0] for m in methods_to_run if len(relaxed_by_method.get(m[0], [])) > 0]
    if not active_methods:
        if node_runner is not None:
            node_runner.warning("No conformers with usable energies available after final relaxation selection.")
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