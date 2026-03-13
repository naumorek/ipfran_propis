"""
Extract input parameters from Mathcad 2000 MCD binary files.

Mathcad 2000 stores variable definitions as:
  - Variable name as ASCII text
  - Numeric values as length-prefixed ASCII digit strings

The key parameters extracted:
  n1, n2 — cycle boundaries (PRN array indices)
  im, isat — precise dead zone / dissolution boundaries (relative to n1)
  im1, isat1 — rough dead zone / dissolution boundaries (relative to n1)
  Salt, Acid, Face — experiment type flags
  dtau — time step (minutes)
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class McdParams:
    """Parameters extracted from a Mathcad MCD file."""
    filename: str
    n1: int = 1
    n2: int = 0
    im: int = 0
    isat: int = 0
    im1: int = 0
    isat1: int = 0
    Salt: int = 1
    Acid: int = 0
    Face: int = 0
    dtau: float = 0.055


def _extract_labeled_values(data: bytes) -> dict[str, list[tuple[int, str]]]:
    """
    Find ASCII number strings preceded by known variable labels.

    In MCD binary format, assignments follow a pattern:
      ... variable_name \x00 ... length_byte digit_string \x00 ...

    Returns dict mapping variable name -> list of (position, value_string).
    """
    results: dict[str, list[tuple[int, str]]] = {}

    # Target variables and their binary label patterns
    targets = {
        'n1': b'n1\x00',
        'n2': b'n2\x00',
        'im1': b'im1\x00',
        'isat1': b'isat1\x00',
        'im': b'im\x00',
        'isat': b'isat\x00',
        'Salt': b'Salt\x00',
        'Acid': b'Acid\x00',
        'Face': b'Face\x00',
        'dtau': b'dtau\x00',
    }

    # Extract all length-prefixed ASCII number strings
    number_entries = []
    for length in range(1, 7):
        pattern = bytes([length]) + rb'([0-9]{' + str(length).encode() + rb'})\x00'
        for m in re.finditer(pattern, data):
            number_entries.append((m.start(), m.group(1).decode('ascii')))

    # Also find decimal numbers (like "0.055")
    for m in re.finditer(rb'([0-9]+\.[0-9]+)', data):
        number_entries.append((m.start(), m.group(1).decode('ascii')))

    number_entries.sort()

    # For each target variable, find its first assignment
    # (the definition block where the variable is set, not used in formulas)
    for var, label in targets.items():
        results[var] = []
        # Find all occurrences of the label
        idx = 0
        while True:
            idx = data.find(label, idx)
            if idx == -1:
                break

            # Disambiguate: 'im' should not match 'im1' or 'isat'
            if var == 'im':
                if idx > 0 and data[idx - 1:idx] in (b'1', b't'):
                    idx += 1
                    continue

            # Find the nearest number AFTER this label (within 200 bytes)
            for pos, val_str in number_entries:
                if pos > idx and pos < idx + 200:
                    results[var].append((pos, val_str))
                    break

            idx += 1

    return results


def parse_mcd(mcd_path: str | Path) -> McdParams:
    """
    Parse a Mathcad MCD file and extract cycle parameters.

    The binary format stores variable assignments at specific positions.
    Through empirical analysis of 15 MCD files, the structure is:
      ~4400-4700: n1, n2 (first large integers)
      ~8000-8700: im1, isat1
      ~12000-13000: im, isat
    """
    mcd_path = Path(mcd_path)
    data = mcd_path.read_bytes()
    result = McdParams(filename=mcd_path.name)

    # Extract all length-prefixed numbers
    number_entries = []
    for length in range(1, 7):
        pattern = bytes([length]) + rb'([0-9]{' + str(length).encode() + rb'})\x00'
        for m in re.finditer(pattern, data):
            val = int(m.group(1))
            number_entries.append((m.start(), val))
    number_entries.sort()

    # Find labeled values by looking at context around variable names
    labeled = _find_variable_values(data, number_entries)

    # Apply found values
    if 'n1' in labeled:
        result.n1 = labeled['n1']
    if 'n2' in labeled:
        result.n2 = labeled['n2']
    if 'im1' in labeled:
        result.im1 = labeled['im1']
    if 'isat1' in labeled:
        result.isat1 = labeled['isat1']
    if 'im' in labeled:
        result.im = labeled['im']
    if 'isat' in labeled:
        result.isat = labeled['isat']

    return result


def _find_variable_values(data: bytes,
                          number_entries: list[tuple[int, int]]) -> dict[str, int]:
    """
    Find variable values using the known binary structure of MCD files.

    Each variable definition appears as:
      ... label_text ... number_value ...
    at characteristic file positions.
    """
    result = {}

    # Find labeled positions
    label_positions = {}
    for label_name, label_bytes in [
        ('n1', b'\x02n1\x00'),
        ('n2', b'\x02n2\x00'),
        ('im1', b'\x03im1\x00'),
        ('isat1', b'\x05isat1\x00'),
        ('im', b'\x02im\x00'),
        ('isat', b'\x04isat\x00'),
    ]:
        positions = []
        idx = 0
        while True:
            idx = data.find(label_bytes, idx)
            if idx == -1:
                break
            positions.append(idx)
            idx += 1
        label_positions[label_name] = positions

    # For each variable, find the FIRST occurrence that has a number
    # in the "definition" region (roughly first half of file)
    file_mid = len(data) // 2

    for var_name, positions in label_positions.items():
        # Filter to positions in first half (definitions, not usage in formulas)
        early_positions = [p for p in positions if p < file_mid]
        if not early_positions:
            early_positions = positions[:1]

        if not early_positions:
            continue

        # Take the first occurrence
        label_pos = early_positions[0]

        # Find the nearest number after this label (within ~150 bytes)
        for num_pos, num_val in number_entries:
            if num_pos > label_pos and num_pos < label_pos + 150:
                # Sanity check based on variable
                if var_name in ('n1', 'n2') and num_val >= 1000:
                    # n1/n2 are always >= 1000 when explicitly stored
                    result[var_name] = num_val
                    break
                elif var_name in ('im', 'isat', 'im1', 'isat1') and 100 <= num_val <= 100000:
                    result[var_name] = num_val
                    break

    # Heuristic: if n1 was not found (or equals n2), default to 1
    # For first cycles, n1=1 is not explicitly stored in MCD
    if 'n1' not in result or ('n2' in result and result.get('n1') == result['n2']):
        result['n1'] = 1

    return result


def parse_all_mcd(base_dir: str | Path) -> list[McdParams]:
    """Parse all MCD files in the project directory tree."""
    base_dir = Path(base_dir)
    results = []

    for mcd_path in sorted(base_dir.rglob('__*.mcd')):
        try:
            r = parse_mcd(mcd_path)
            r.filename = str(mcd_path.relative_to(base_dir))
            results.append(r)
        except Exception as e:
            print(f"Error parsing {mcd_path}: {e}")

    return results


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = Path(__file__).parent.parent.parent

    results = parse_all_mcd(base_dir)

    print(f"{'File':<55} {'n1':>6} {'n2':>6} {'im':>6} {'isat':>6} "
          f"{'im1':>6} {'isat1':>6}")
    print('-' * 95)

    for r in results:
        print(f"{r.filename:<55} {r.n1:>6} {r.n2:>6} {r.im:>6} {r.isat:>6} "
              f"{r.im1:>6} {r.isat1:>6}")
