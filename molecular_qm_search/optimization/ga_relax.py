from enum import Enum
from typing import Optional

from odmantic import Field, Model
from pydantic import model_validator

from molecular_qm_dftb.models.dftb_input import DftbInput
from molecular_qm_models import MoleculeList
from molecular_qm_psi4.models.crest_input import CrestLevelOfTheory
from molecular_qm_search.optimization.models.ga_models import GAConfig
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


class GARelaxOptimizer(str, Enum):
    """List calculator that relaxes the genetic-algorithm population."""

    XTB = "xtb"
    DFTB = "dftb"


@simstack_model
class XTBOptimizerParameters(Model):
    """xTB list-optimizer settings, without the molecule list.

    ``ga_relax`` supplies the genetic-algorithm population and always optimizes it.
    Field names match ``XTBInput`` aside from ``molecules`` and ``optimize``.
    """

    field_name: str = "XTBOptimizerParameters"
    level_of_theory: CrestLevelOfTheory = Field(default_factory=CrestLevelOfTheory)
    compute_gradients: bool = Field(False, description="Whether to compute gradients")
    additional_keywords: Optional[str] = Field(
        None, description="Any additional CREST command line arguments"
    )

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = "xTB optimizer"
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
            "additional_keywords",
        ]
        return ui


@simstack_model
class GARelaxInput(Model):
    """Genetic algorithm followed by an xTB or DFTB list optimization."""

    field_name: str = "GARelaxInput"
    ga_config: Optional[GAConfig] = Field(
        None,
        json_schema_extra={
            "title": "Genetic algorithm",
            "description": "Settings for run_ga_conformer_gen, including the starting molecule.",
        },
    )
    optimizer: GARelaxOptimizer = Field(
        GARelaxOptimizer.DFTB,
        json_schema_extra={
            "title": "Final relaxation",
            "description": "Relax the genetic-algorithm population with this list optimizer.",
        },
    )
    xtb_parameters: Optional[XTBOptimizerParameters] = Field(
        None,
        json_schema_extra={
            "title": "xTB optimizer",
            "description": "Parameters for xtb_optimize_molecule_list. The molecules are the GA population.",
        },
    )
    dftb_parameters: Optional[DftbInput] = Field(
        None,
        json_schema_extra={
            "title": "DFTB optimizer",
            "description": "Parameters for dftb_list_calculator. Geometry optimization must be enabled.",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def sync_optimizer_parameters(cls, data):
        if not isinstance(data, dict):
            return data
        if "optimizer" not in data:
            method = GARelaxOptimizer.DFTB
        else:
            method = data["optimizer"]
            if method is None:
                raise ValueError("optimizer is required")
            if isinstance(method, str):
                try:
                    method = GARelaxOptimizer(method)
                except ValueError as exc:
                    raise ValueError("optimizer must be xtb or dftb") from exc
            if method not in (GARelaxOptimizer.XTB, GARelaxOptimizer.DFTB):
                raise ValueError("optimizer must be xtb or dftb")
        if method == GARelaxOptimizer.XTB:
            data["dftb_parameters"] = None
        else:
            data["xtb_parameters"] = None
        return data

    @model_validator(mode="after")
    def require_selected_optimizer_parameters(self):
        if self.ga_config is None:
            raise ValueError("GA settings are required")
        if self.optimizer == GARelaxOptimizer.XTB:
            if self.xtb_parameters is None:
                raise ValueError(
                    "xTB optimizer parameters are required when xTB relaxation is selected"
                )
        elif self.optimizer == GARelaxOptimizer.DFTB:
            if self.dftb_parameters is None:
                raise ValueError(
                    "DFTB optimizer parameters are required when DFTB relaxation is selected"
                )
            if not self.dftb_parameters.optimization:
                raise ValueError("DFTB optimization must be enabled to relax conformers")
        else:
            raise ValueError("optimizer must be xtb or dftb")
        return self

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        props = schema["properties"]
        props.pop("ga_config", None)
        props.pop("xtb_parameters", None)
        props.pop("dftb_parameters", None)
        schema["required"] = [
            name
            for name in schema.get("required", [])
            if name not in {"ga_config", "xtb_parameters", "dftb_parameters"}
        ]
        schema["required"].append("ga_config")

        ga_schema = cleaned_json_schema(GAConfig)
        ga_schema["title"] = "Genetic algorithm"
        for name, definition in (ga_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        xtb_schema = XTBOptimizerParameters.json_schema()
        for name, definition in (xtb_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        xtb_schema["default"] = XTBOptimizerParameters().model_dump(mode="json", exclude={"id"})
        dftb_schema = DftbInput.json_schema()
        for name, definition in (dftb_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        dftb_schema["title"] = "DFTB optimizer"
        dftb_schema["default"] = DftbInput(
            optimization=True, max_optimization_steps=1000
        ).model_dump(mode="json", exclude={"id"})

        props["ga_config"] = ga_schema
        schema.setdefault("dependencies", {})["optimizer"] = {
            "oneOf": [
                {
                    "properties": {
                        "optimizer": {"const": GARelaxOptimizer.XTB.value},
                        "xtb_parameters": xtb_schema,
                    },
                    "required": ["xtb_parameters"],
                },
                {
                    "properties": {
                        "optimizer": {"const": GARelaxOptimizer.DFTB.value},
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
        ui["optimizer"] = {
            "ui:widget": "select",
            "ui:title": "Final relaxation",
        }
        ui.setdefault("xtb_parameters", {})["ui:condition"] = {
            "optimizer": GARelaxOptimizer.XTB.value
        }
        ui.setdefault("dftb_parameters", {})["ui:condition"] = {
            "optimizer": GARelaxOptimizer.DFTB.value
        }
        return ui


@node
async def ga_relax(opts: GARelaxInput, **kwargs) -> SimstackResult:
    """Relax a genetic-algorithm population with xTB or DFTB.

    The genetic algorithm runs first. ``optimizer`` then selects the relaxation
    and the form exposes only that calculator's settings. xTB calls
    ``xtb_optimize_molecule_list``, which always optimizes the molecule list.
    DFTB calls ``dftb_list_calculator``. That node optimizes each molecule when
    ``DftbInput.optimization`` is true, so this node requires the flag on.

    There is no ``xtb_list_calculator``. ``xtb_molecule_list`` optimizes by
    forwarding to ``xtb_optimize_molecule_list`` when ``XTBInput.optimize`` is
    true. DFTB list optimization needs this node's Slurm task settings, because
    ``dftb_list_calculator`` reads them from the parent parameters.

    Parameters:
        opts (GARelaxInput): Genetic algorithm settings and the selected optimizer.

    Called Nodes:
        run_ga_conformer_gen
        dftb_list_calculator
        xtb_optimize_molecule_list

    Raises:
        ValueError: missing settings, an empty population, or a relaxation result without energies

    SimstackResult:
        molecules (MoleculeList): Relaxed conformers, lowest energy first
    """
    node_runner = kwargs.get("node_runner")
    if node_runner is None:
        raise ValueError("ga_relax requires node_runner")
    if opts.ga_config is None:
        raise ValueError("GA settings are required")

    from molecular_qm_search.optimization.ga import run_ga_conformer_gen

    node_runner.info(
        f"Starting GA ({opts.ga_config.mode}) before {opts.optimizer.value} relaxation"
    )
    ga_result = await run_ga_conformer_gen(opts.ga_config, **dict(kwargs))
    if getattr(ga_result, "error_message", None):
        raise ValueError(f"run_ga_conformer_gen failed: {ga_result.error_message}")
    if isinstance(ga_result, MoleculeList):
        conformers = ga_result
    else:
        conformers = None if ga_result is None else getattr(ga_result, "molecules", None)
    if conformers is None:
        conformers = getattr(node_runner, "molecules", None)
    if not isinstance(conformers, MoleculeList):
        raise ValueError("run_ga_conformer_gen did not return molecules")
    if len(conformers) == 0:
        raise ValueError("run_ga_conformer_gen returned no conformers")

    node_runner.info(
        f"Relaxing {len(conformers)} GA conformers with {opts.optimizer.value}"
    )
    if opts.optimizer == GARelaxOptimizer.XTB:
        from molecular_qm_psi4.models.crest_input import XTBInput
        from molecular_qm_psi4.nodes.crest import xtb_optimize_molecule_list

        if opts.xtb_parameters is None:
            raise ValueError(
                "xTB optimizer parameters are required when xTB relaxation is selected"
            )
        xtb_input = XTBInput(
            molecules=conformers,
            level_of_theory=opts.xtb_parameters.level_of_theory,
            compute_gradients=opts.xtb_parameters.compute_gradients,
            optimize=True,
            additional_keywords=opts.xtb_parameters.additional_keywords,
        )
        calc_result = await xtb_optimize_molecule_list(xtb_input, **dict(kwargs))
        if getattr(calc_result, "error_message", None):
            raise ValueError(
                f"xtb_optimize_molecule_list failed: {calc_result.error_message}"
            )
        molecules_out = getattr(calc_result, "molecule_list", None)
        if molecules_out is None:
            molecules_out = getattr(node_runner, "molecule_list", None)
        if molecules_out is None:
            raise ValueError("xtb_optimize_molecule_list did not return molecules")
        relaxed_pairs = []
        for molecule in molecules_out:
            energy = molecule.properties.get("energy")
            if energy is None:
                raise ValueError("xTB relaxation returned a conformer without an energy")
            energy = float(energy)
            molecule.properties["energy"] = energy
            relaxed_pairs.append((molecule, energy))
    elif opts.optimizer == GARelaxOptimizer.DFTB:
        from molecular_qm_dftb.nodes.dftb_list_calculator import dftb_list_calculator
        from simstack.core.context import context

        if opts.dftb_parameters is None:
            raise ValueError(
                "DFTB optimizer parameters are required when DFTB relaxation is selected"
            )
        if not opts.dftb_parameters.optimization:
            raise ValueError("DFTB optimization must be enabled to relax conformers")
        calc_result = await dftb_list_calculator(
            conformers, opts.dftb_parameters, **dict(kwargs)
        )
        if getattr(calc_result, "error_message", None):
            raise ValueError(f"dftb_list_calculator failed: {calc_result.error_message}")
        dataset = getattr(calc_result, "dataset", None)
        if dataset is None and hasattr(calc_result, "sections"):
            dataset = calc_result
        if dataset is None:
            dataset = getattr(node_runner, "dataset", None)
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
        relaxed_pairs = []
        for _row_name, row in results.items():
            success = row.get("success")
            if success is None:
                raise ValueError("DFTB result row is missing success")
            success_value = success.value if hasattr(success, "value") else success
            if not success_value:
                error = row.get("error")
                if error is None:
                    node_runner.warning("DFTB relaxation failed for a conformer")
                else:
                    error_msg = error.value if hasattr(error, "value") else error
                    node_runner.warning(f"DFTB relaxation failed for a conformer: {error_msg}")
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
            relaxed_pairs.append((opt_mol, energy))
    else:
        raise ValueError("optimizer must be xtb or dftb")

    if not relaxed_pairs:
        raise ValueError(f"{opts.optimizer.value} relaxation produced no conformers")
    relaxed_pairs.sort(key=lambda item: item[1])
    relaxed = MoleculeList()
    for molecule, _energy in relaxed_pairs:
        relaxed.append(molecule)
    node_runner.molecules = relaxed
    node_runner.info(f"ga-relax finished with {len(relaxed)} conformers")
    return node_runner.succeed()


ga_relax._node_name = "ga-relax"
