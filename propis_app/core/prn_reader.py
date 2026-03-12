"""
Reading and parsing PRN files from the interferometric stand.

PRN format: ~65000 rows, 7 columns separated by whitespace:
  col 0: sample index (int)
  col 1: LED1 signal (V) — interferometric channel 1 (λ=470nm)
  col 2: LED2 signal (V) — interferometric channel 2 (λ=590nm)
  col 3: flag/status (always 0.00)
  col 4: temperature in conditional units (~273+T)
  col 5: temperature in °C
  col 6: time string HH:MM:SS
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PrnData:
    """Parsed PRN file data."""
    filepath: Path
    index: np.ndarray         # sample indices
    led1: np.ndarray          # LED1 signal (V)
    led2: np.ndarray          # LED2 signal (V)
    flag: np.ndarray          # status flag
    temp_raw: np.ndarray      # temperature in conditional units
    temp_c: np.ndarray        # temperature in °C
    time_strings: list        # time as HH:MM:SS strings
    time_seconds: np.ndarray  # time in seconds from start

    @property
    def n_samples(self) -> int:
        return len(self.index)

    @property
    def dt(self) -> float:
        """Average time step in seconds."""
        if len(self.time_seconds) < 2:
            return 1.0
        return float(np.median(np.diff(self.time_seconds)))

    def slice(self, n1: int, n2: int) -> "PrnData":
        """Return a copy with data sliced to [n1:n2]."""
        return PrnData(
            filepath=self.filepath,
            index=self.index[n1:n2],
            led1=self.led1[n1:n2],
            led2=self.led2[n1:n2],
            flag=self.flag[n1:n2],
            temp_raw=self.temp_raw[n1:n2],
            temp_c=self.temp_c[n1:n2],
            time_strings=self.time_strings[n1:n2],
            time_seconds=self.time_seconds[n1:n2],
        )


def _parse_time_to_seconds(time_strings: list) -> np.ndarray:
    """Convert HH:MM:SS strings to seconds from the first timestamp."""
    seconds = np.zeros(len(time_strings), dtype=np.float64)
    for i, ts in enumerate(time_strings):
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            seconds[i] = h * 3600 + m * 60 + s

    # Handle midnight crossing
    for i in range(1, len(seconds)):
        if seconds[i] < seconds[i - 1]:
            seconds[i:] += 86400
            break

    # Make relative to start
    seconds -= seconds[0]
    return seconds


def read_prn(filepath: str | Path) -> PrnData:
    """
    Read a PRN file and return parsed data.

    Parameters
    ----------
    filepath : str or Path
        Path to the PRN file.

    Returns
    -------
    PrnData
        Parsed data structure.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PRN file not found: {filepath}")

    indices = []
    led1 = []
    led2 = []
    flags = []
    temp_raw = []
    temp_c = []
    time_strings = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                indices.append(int(parts[0]))
                led1.append(float(parts[1]))
                led2.append(float(parts[2]))
                flags.append(float(parts[3]))
                temp_raw.append(float(parts[4]))
                temp_c.append(float(parts[5]))
                time_strings.append(parts[6])
            except (ValueError, IndexError):
                continue

    if not indices:
        raise ValueError(f"No valid data rows found in {filepath}")

    ts = _parse_time_to_seconds(time_strings)

    return PrnData(
        filepath=filepath,
        index=np.array(indices, dtype=np.int64),
        led1=np.array(led1, dtype=np.float64),
        led2=np.array(led2, dtype=np.float64),
        flag=np.array(flags, dtype=np.float64),
        temp_raw=np.array(temp_raw, dtype=np.float64),
        temp_c=np.array(temp_c, dtype=np.float64),
        time_strings=time_strings,
        time_seconds=ts,
    )
