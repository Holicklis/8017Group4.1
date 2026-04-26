"""Financial Synthesis model package."""

from .synthesis_engine import (
    generate_synthesis,
    get_synapse_alerts,
    get_synthesis_context,
)

__all__ = [
    "generate_synthesis",
    "get_synthesis_context",
    "get_synapse_alerts",
]
