"""Explicit cooperative cancellation shared by inference and experiment layers."""


class PreparationCancelled(RuntimeError):
    """Preparation stopped between operations after a cancellation request."""
