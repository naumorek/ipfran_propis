"""
Extract computed values from Mathcad 2000 RTF exports.

Mathcad 2000 exports formulas as WMF (Windows Metafile) pictures embedded in RTF.
Text within WMF contains variable names and their computed values as ASCII strings.

Strategy: find "display blocks" — short WMF blocks with varname and a number,
WITHOUT ':=' operator. These are the computed results at the bottom of the document.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class MathcadResults:
    """Computed results extracted from Mathcad RTF export."""
    filename: str
    # Key output parameters
    te: Optional[float] = None       # equilibrium temperature (°C)
    te1: Optional[float] = None      # te1 parameter
    tn: Optional[float] = None       # saturation temperature (°C)
    Td: Optional[float] = None       # dead zone width (°C)
    Sigm: Optional[float] = None     # dead zone (relative supersaturation %)
    s1: Optional[float] = None       # s1 parameter
    s2: Optional[float] = None       # power law exponent
    Sig035: Optional[float] = None   # supersaturation at F1=0.35
    s0: Optional[float] = None       # growth rate coefficient
    # Input parameters
    Salt: Optional[int] = None
    Acid: Optional[int] = None
    Face: Optional[int] = None
    n1: Optional[int] = None
    n2: Optional[int] = None
    im: Optional[int] = None
    isat: Optional[int] = None
    im1: Optional[int] = None
    isat1: Optional[int] = None
    dtau: Optional[float] = None
    ww: Optional[int] = None
    d: Optional[float] = None
    l: Optional[float] = None


# Sanity ranges for each variable
_RANGES = {
    'te':     (30, 60),
    'te1':    (0, 60),
    'tn':     (30, 60),
    'Td':     (0.01, 5),
    'Sigm':   (-2, 10),
    's1':     (-5, 5),
    's2':     (-5, 5),
    'Sig035': (0.1, 20),
    's0':     (0.01, 500),
    'Salt':   (0, 3),
    'Acid':   (0, 2),
    'Face':   (0, 2),
    'n1':     (0, 100000),
    'n2':     (0, 100000),
    'im':     (0, 100000),
    'isat':   (0, 100000),
    'im1':    (0, 100000),
    'isat1':  (0, 100000),
    'dtau':   (0.001, 1),
    'ww':     (0, 10),
    'd':      (0.1, 100),
}


def _extract_wmf_blocks(rtf_data: bytes) -> list[list[str]]:
    """Extract ASCII text strings from all WMF picture objects in RTF."""
    hex_blocks = re.findall(
        rb'\{\\result \{\\pict[^}]*?\\picwgoal\d+\\pichgoal\d+\s+([0-9a-f\s]+)\}',
        rtf_data
    )

    results = []
    for block in hex_blocks:
        hex_str = block.decode('ascii').replace('\n', '').replace('\r', '').replace(' ', '')
        try:
            wmf_bytes = bytes.fromhex(hex_str)
        except ValueError:
            continue

        ascii_parts = re.findall(rb'[\x20-\x7e]{2,}', wmf_bytes)
        text_parts = [p.decode('ascii').strip() for p in ascii_parts]
        results.append(text_parts)

    return results


def _is_definition(texts: list[str]) -> bool:
    """Check if WMF block is a definition (contains ':=')."""
    return any(':=' in t for t in texts)


def _find_number_in_block(texts: list[str]) -> Optional[float]:
    """Find the first reasonable number in a WMF block.

    Skip font names, markers, and known non-data strings.
    """
    skip = {'w@', 'Times New Roman', 'Symbol', 'MS Sans Serif',
            'Arial', 'System', 'Kudriashov', ':=', '=='}

    for t in texts:
        tc = t.strip()
        if tc in skip:
            continue
        # Remove trailing junk chars that sometimes appear
        tc_clean = re.sub(r'[#$%()\[\]{}|;:,<>!&*^~`\'"\\]', '', tc).strip()
        if not tc_clean:
            continue
        try:
            return float(tc_clean)
        except ValueError:
            continue

    return None


def parse_rtf(rtf_path: str | Path) -> MathcadResults:
    """Parse a Mathcad RTF export and extract computed values."""
    rtf_path = Path(rtf_path)

    with open(rtf_path, 'rb') as f:
        data = f.read()

    blocks = _extract_wmf_blocks(data)
    result = MathcadResults(filename=rtf_path.name)

    # Collect all display blocks (no :=) that mention known variable names
    target_vars = list(_RANGES.keys())

    # For each variable, collect (block_index, value) from display blocks
    var_values: dict[str, list[tuple[int, float]]] = {v: [] for v in target_vars}

    for block_idx, texts in enumerate(blocks):
        if _is_definition(texts):
            continue

        # Check which target variables appear in this block
        for var in target_vars:
            if any(t.strip() == var for t in texts):
                # Find the first reasonable number
                val = _find_number_in_block(
                    [t for t in texts if t.strip() != var]  # exclude varname itself
                )
                if val is not None:
                    lo, hi = _RANGES[var]
                    if lo <= val <= hi:
                        var_values[var].append((block_idx, val))

    # For each variable, take the LAST display value
    # (results are displayed at the bottom of the Mathcad document)
    type_map = {
        'Salt': int, 'Acid': int, 'Face': int,
        'n1': int, 'n2': int, 'im': int, 'isat': int,
        'im1': int, 'isat1': int, 'ww': int,
    }

    for var in target_vars:
        if var_values[var]:
            _, val = var_values[var][-1]
            converter = type_map.get(var, float)
            try:
                setattr(result, var, converter(val))
            except (ValueError, TypeError):
                pass

    return result


def parse_all_rtf(base_dir: str | Path) -> list[MathcadResults]:
    """Parse all RTF files in the project directory tree."""
    base_dir = Path(base_dir)
    results = []

    for rtf_path in sorted(base_dir.rglob('*.rtf')):
        if not rtf_path.name.startswith('__'):
            continue
        try:
            r = parse_rtf(rtf_path)
            r.filename = str(rtf_path.relative_to(base_dir))
            results.append(r)
        except Exception as e:
            print(f"Error parsing {rtf_path}: {e}")

    return results


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = Path(__file__).parent.parent.parent

    results = parse_all_rtf(base_dir)

    # Header
    print(f"{'File':<55} {'te':>6} {'tn':>6} {'Td':>5} {'Sigm':>5} "
          f"{'s2':>5} {'Sig035':>6}")
    print('-' * 90)

    for r in results:
        def fmt(val, width, decimals=2):
            if val is None:
                return f"{'---':>{width}}"
            return f"{val:>{width}.{decimals}f}"

        print(f"{r.filename:<55} "
              f"{fmt(r.te, 6)} {fmt(r.tn, 6)} {fmt(r.Td, 5)} "
              f"{fmt(r.Sigm, 5)} {fmt(r.s2, 5)} {fmt(r.Sig035, 6, 1)}")
