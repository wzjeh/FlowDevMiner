from __future__ import annotations

from typing import Optional

try:
    # Two possible opsin python interfaces exist on PyPI; support both.
    import opsin  # type: ignore
    _OPSIN_STYLE = "simple"
except Exception:
    try:
        from opsin import opsin as opsin_mod  # type: ignore
        _OPSIN_STYLE = "module"
    except Exception:
        opsin = None  # type: ignore
        opsin_mod = None  # type: ignore
        _OPSIN_STYLE = "none"

from rdkit import Chem

try:
    from rdkit.Chem.inchi import MolToInchi  # type: ignore
except Exception:
    MolToInchi = None  # type: ignore


def name_to_smiles_via_opsin(name: str) -> Optional[str]:
    if not name or _OPSIN_STYLE == "none":
        return None
    try:
        if _OPSIN_STYLE == "simple":
            # Some packages expose opsin.name_to_smiles directly
            # Fallback to getattr in case function name differs.
            func = getattr(opsin, "name_to_smiles", None)  # type: ignore
            if callable(func):
                smiles = func(name)  # type: ignore
                return smiles or None
        elif _OPSIN_STYLE == "module":
            # Java-backed interface returning an object with .SMILES
            result = opsin_mod.parseIUPACName(name)  # type: ignore
            smiles = getattr(result, "SMILES", None)
            return smiles or None
    except Exception:
        return None
    return None


def smiles_to_inchi(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if MolToInchi is None:
            return None
        return MolToInchi(mol)
    except Exception:
        return None


def annotate_compound_identifiers(component, name: str) -> None:
    """
    Adds NAME (always), then tries to add SMILES and InChI identifiers to the given component.
    `component` is an ord_schema.proto.reaction_pb2.ReactionInput.components[x].
    """
    # NAME
    ident = component.compound.identifiers.add()
    from ord_schema.proto import reaction_pb2  # local import to avoid global dependency at import time

    ident.type = reaction_pb2.CompoundIdentifier.NAME
    ident.value = name

    # SMILES
    smiles = name_to_smiles_via_opsin(name)
    if smiles:
        ident_smi = component.compound.identifiers.add()
        ident_smi.type = reaction_pb2.CompoundIdentifier.SMILES
        ident_smi.value = smiles
        # InChI from SMILES
        inchi = smiles_to_inchi(smiles)
        if inchi:
            ident_inchi = component.compound.identifiers.add()
            ident_inchi.type = reaction_pb2.CompoundIdentifier.INCHI
            ident_inchi.value = inchi



