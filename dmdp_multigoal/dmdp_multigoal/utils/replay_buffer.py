from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RolloutBuffer:
    data: dict[str, list[Any]] = field(default_factory=dict)

    def add(self, **items: Any) -> None:
        for key, value in items.items():
            self.data.setdefault(key, []).append(value)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self.data.items()}

    def clear(self) -> None:
        self.data.clear()

    def __len__(self) -> int:
        if not self.data:
            return 0
        return len(next(iter(self.data.values())))
