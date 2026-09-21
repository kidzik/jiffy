"""Parallel typed decisions with frozen DiffusionGemma."""
from .schema import Question

__all__ = ["Question", "DiffusionDecisions", "image_data_url"]


def __getattr__(name):
    if name == "DiffusionDecisions":
        from .diffusion_decisions import DiffusionDecisions
        return DiffusionDecisions
    if name == "image_data_url":
        from .media import image_data_url
        return image_data_url
    raise AttributeError(name)
