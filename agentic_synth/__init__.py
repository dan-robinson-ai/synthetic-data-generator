"""
Agentic Synth - AI-powered synthetic data generation for ML training, RAG systems, and testing.
"""

from agentic_synth.core.config import GeneratorConfig
from agentic_synth.core.generator import SyntheticGenerator
from agentic_synth.core.cache import CacheManager

__version__ = "0.1.0"
__all__ = [
    "SyntheticGenerator",
    "GeneratorConfig",
    "CacheManager",
]
