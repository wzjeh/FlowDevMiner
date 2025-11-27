#!/usr/bin/env python3
"""
Convert a simple reaction JSON (project-specific schema) into an Open Reaction Database Dataset (pbtxt).

Usage:
    python ord_convert.py --input finetune/ground_truth/1_reaction_annotated.json --output ord/1.dataset.pbtxt --name "My Dataset" --description "Converted from project JSON"
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional
import glob

from ord_schema import message_helpers
from ord_schema.proto import dataset_pb2, reaction_pb2
from core.chem_identifiers import annotate_compound_identifiers


def _strip_to_float(text: Any) -> float | None:
    """Extracts a float from a string like '80 °C' or returns None if not found."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        try:
            return float(text)
        except Exception:
            return None
    if not isinstance(text, str):
        return None
    # Replace non-ASCII degree or stray characters and grab the first float
    cleaned = text.replace("℃", "").replace("°C", "").replace("°", "").replace("ﾂｰC", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    return float(match.group(0)) if match else None


def _ensure_input(reaction: reaction_pb2.Reaction, key: str) -> reaction_pb2.ReactionInput:
    """Gets or creates a ReactionInput in the Reaction.inputs map."""
    # inputs is a map<string, ReactionInput>
    _ = reaction.inputs[key]  # accessing the key ensures it exists
    return reaction.inputs[key]


def _add_component_by_name(input_slot: reaction_pb2.ReactionInput, name: str) -> None:
    """Adds a component and annotates NAME/SMILES/InChI if possible."""
    component = input_slot.components.add()
    annotate_compound_identifiers(component, name)


def _add_product(outcome: reaction_pb2.ReactionOutcome, name: str, yield_percent: float | None) -> None:
    """Adds a product with optional yield percentage."""
    product = outcome.products.add()
    identifier = product.identifiers.add()
    identifier.type = reaction_pb2.CompoundIdentifier.NAME
    identifier.value = name
    product.is_desired_product = True
    if yield_percent is not None:
        measurement = product.measurements.add()
        measurement.type = reaction_pb2.ProductMeasurement.YIELD
        measurement.percentage.value = float(yield_percent)


def _set_temperature(reaction: reaction_pb2.Reaction, value: Optional[float], unit_hint: Optional[str]) -> None:
    if value is None:
        return
    temperature = reaction.conditions.temperature
    temperature.setpoint.value = float(value)
    # Default to Celsius; support 'K' as Kelvin
    units = reaction_pb2.Temperature.CELSIUS
    if isinstance(unit_hint, str) and unit_hint.strip().lower().startswith("k"):
        units = getattr(reaction_pb2.Temperature, "KELVIN", reaction_pb2.Temperature.CELSIUS)
    temperature.setpoint.units = units


def _set_residence_time(reaction: reaction_pb2.Reaction, value: Optional[float], unit_hint: Optional[str]) -> None:
    if value is None:
        return
    t = reaction.conditions.residence_time.setpoint
    t.value = float(value)
    units = reaction_pb2.Time.HOUR
    if isinstance(unit_hint, str):
        u = unit_hint.strip().lower()
        if u.startswith("s"):
            units = getattr(reaction_pb2.Time, "SECOND", reaction_pb2.Time.HOUR)
        elif u.startswith("min"):
            units = getattr(reaction_pb2.Time, "MINUTE", reaction_pb2.Time.HOUR)
        elif u.startswith("h"):
            units = getattr(reaction_pb2.Time, "HOUR", reaction_pb2.Time.HOUR)
    t.units = units


def _set_flow_rate(reaction: reaction_pb2.Reaction, value: Optional[float], unit_hint: Optional[str]) -> None:
    if value is None:
        return
    fr = reaction.conditions.flow_rate.setpoint
    fr.value = float(value)
    # Default milliliter per hour
    default_unit = None
    try:
        default_unit = getattr(reaction_pb2.FlowRate, "MILLILITER_PER_HOUR")
    except Exception:
        default_unit = None
    units = default_unit
    if isinstance(unit_hint, str):
        u = unit_hint.strip().lower()
        try:
            if "ml/h" in u or "mL/h" in u:
                units = getattr(reaction_pb2.FlowRate, "MILLILITER_PER_HOUR")
            elif "ml/min" in u or "mL/min" in u:
                units = getattr(reaction_pb2.FlowRate, "MILLILITER_PER_MINUTE")
            elif "ul/min" in u or "µl/min" in u:
                units = getattr(reaction_pb2.FlowRate, "MICROLITER_PER_MINUTE")
            elif "l/min" in u:
                units = getattr(reaction_pb2.FlowRate, "LITER_PER_MINUTE")
            elif "l/h" in u:
                units = getattr(reaction_pb2.FlowRate, "LITER_PER_HOUR")
        except Exception:
            pass
    if units is not None:
        fr.units = units


def _set_pressure(reaction: reaction_pb2.Reaction, value: Optional[float], unit_hint: Optional[str]) -> None:
    if value is None:
        return
    p = reaction.conditions.pressure.setpoint
    p.value = float(value)
    units = None
    try:
        units = getattr(reaction_pb2.Pressure, "BAR")
    except Exception:
        units = None
    if isinstance(unit_hint, str):
        u = unit_hint.strip().lower()
        try:
            if "bar" in u:
                units = getattr(reaction_pb2.Pressure, "BAR")
            elif "kpa" in u:
                units = getattr(reaction_pb2.Pressure, "KILOPASCAL")
            elif "mpa" in u:
                units = getattr(reaction_pb2.Pressure, "MEGAPASCAL")
            elif "atm" in u:
                units = getattr(reaction_pb2.Pressure, "ATMOSPHERE")
            elif "pa" in u:
                units = getattr(reaction_pb2.Pressure, "PASCAL")
        except Exception:
            pass
    if units is not None:
        p.units = units


def _parse_value_and_unit(raw: Any) -> (Optional[float], Optional[str]):
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    if not isinstance(raw, str):
        return None, None
    # Extract numeric value and leave the remainder as unit hint
    cleaned = raw.replace("℃", " C").replace("°C", " C").replace("°", " ")
    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cleaned)
    value = float(match.group(1)) if match else None
    unit_hint = cleaned[match.end():].strip() if match else cleaned.strip()
    return value, unit_hint or None


def _map_conditions_structured(reaction: reaction_pb2.Reaction, data: Dict[str, Any]) -> None:
    # Recognize by 'type' field in a list of dicts
    for cond in data.get("conditions", []):
        ctype = (cond.get("type") or "").strip().lower()
        raw = cond.get("value")
        value, unit_hint = _parse_value_and_unit(raw)
        if ctype in ("temperature",):
            _set_temperature(reaction, value, unit_hint)
        elif ctype in ("residence_time", "residence time", "time"):
            _set_residence_time(reaction, value, unit_hint)
        elif ctype in ("flow_rate_total", "flow rate", "flow_rate"):
            _set_flow_rate(reaction, value, unit_hint)
        elif ctype in ("pressure",):
            _set_pressure(reaction, value, unit_hint)


def convert_single_json_to_reaction(data: Dict[str, Any]) -> reaction_pb2.Reaction:
    """Converts one project JSON dict into an ORD Reaction."""
    reaction = reaction_pb2.Reaction()

    # Map reactants by role into input buckets
    role_to_bucket = {
        "reactant": "reactants",
        "catalyst": "catalysts",
        "initiator": "initiators",
        "solvent": "solvent",
    }
    for item in data.get("reactants", []):
        name = item.get("name")
        role = (item.get("role") or "").lower()
        if not name:
            continue
        bucket = role_to_bucket.get(role, "others")
        input_slot = _ensure_input(reaction, bucket)
        _add_component_by_name(input_slot, name)

    # Add products and yields (as an outcome)
    outcome = reaction.outcomes.add()  # create a single outcome
    for prod in data.get("products", []):
        prod_name = prod.get("name") or "product"
        y = prod.get("yield")
        if y is None:
            # Some files use 'yield_optimal'
            y = prod.get("yield_optimal")
        y = _strip_to_float(y)
        _add_product(outcome, prod_name, y)

    # Map structured conditions
    _map_conditions_structured(reaction, data)

    # Reactor info (kept textual in notes)
    reactor = data.get("reactor") or {}
    reactor_bits: List[str] = []
    if isinstance(reactor, dict):
        for k, v in reactor.items():
            if v is not None:
                reactor_bits.append(f"{k}: {v}")
    if reactor_bits:
        if reaction.notes.details:
            reaction.notes.details += "; " + "; ".join(reactor_bits)
        else:
            reaction.notes.details = "; ".join(reactor_bits)

    # Metrics (store into outcome only if clearly mappable; otherwise into notes)
    metrics = data.get("metrics") or {}
    # If yield is available only here, add a measurement to the first product
    metrics_yield = _strip_to_float(metrics.get("yield"))
    if metrics_yield is not None:
        if outcome.products:
            measurement = outcome.products[0].measurements.add()
            measurement.type = reaction_pb2.ProductMeasurement.YIELD
            measurement.percentage.value = float(metrics_yield)
        else:
            # No product created; create one to hold the yield
            _add_product(outcome, "product", float(metrics_yield))

    return reaction


def convert_file_to_dataset(
    input_path: str,
    dataset_name: str,
    description: str | None = None,
) -> dataset_pb2.Dataset:
    """Reads a project JSON file and returns an ORD Dataset with one reaction."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ds = dataset_pb2.Dataset()
    ds.name = dataset_name
    if description:
        ds.description = description
    reaction = convert_single_json_to_reaction(data)
    ds.reactions.add().CopyFrom(reaction)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert project JSON to ORD Dataset pbtxt")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to project JSON (e.g., finetune/ground_truth/1_reaction_annotated.json)")
    group.add_argument("--input_dir", help="Directory containing JSON files to merge into a single Dataset")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern under --input_dir (default: *.json)")
    parser.add_argument("--output", required=True, help="Output path for ORD Dataset (e.g., ord/1.dataset.pbtxt)")
    parser.add_argument("--name", default="Auto-generated dataset", help="Dataset name")
    parser.add_argument("--description", default="Converted from project JSON", help="Dataset description")
    parser.add_argument("--validate", action="store_true", help="Validate the resulting Dataset against the ORD schema")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.input_dir:
        dataset = dataset_pb2.Dataset()
        dataset.name = args.name
        dataset.description = args.description
        files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reaction = convert_single_json_to_reaction(data)
                dataset.reactions.add().CopyFrom(reaction)
            except Exception as e:
                # Skip problematic files but continue
                print(f"Warning: failed to convert {path}: {e}")
    else:
        dataset = convert_file_to_dataset(args.input, args.name, args.description)

    if args.validate:
        try:
            from ord_schema.validations import validate_message  # type: ignore
            validate_message(dataset)
        except Exception as e:
            print(f"Validation warning/error: {e}")

    message_helpers.write_message(dataset, args.output)
    print(f"Wrote ORD Dataset to {args.output}")


if __name__ == "__main__":
    main()


