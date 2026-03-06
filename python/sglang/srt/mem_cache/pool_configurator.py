from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class BasePoolConfigurator(ABC):
    @abstractmethod
    def __init__(self, model_runner: ModelRunner):
        pass

    @abstractmethod
    def configure(
        self,
        available_bytes: int,
        page_size: int,
        full_token_constraint: Optional[int] = None,
    ):
        pass
