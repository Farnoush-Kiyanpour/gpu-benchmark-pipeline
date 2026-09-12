#!/usr/bin/env python3
"""
Tool wrapper registry. Maps tool names to their wrapper classes.
"""
from .base_wrapper import (
    BaseWrapper, Pocket3DWrapper, SequenceWrapper,
    InteractionWrapper, PharmacophoreWrapper
)
from .pocket3d_wrappers import POCKET3D_WRAPPERS
from .sequence_wrappers import SEQUENCE_WRAPPERS
from .interaction_wrappers import INTERACTION_WRAPPERS
from .pharmacophore_wrappers import PHARMACOPHORE_WRAPPERS
from .classical_wrappers import CLASSICAL_WRAPPERS

# Combined registry
TOOL_WRAPPERS = {}
TOOL_WRAPPERS.update(POCKET3D_WRAPPERS)
TOOL_WRAPPERS.update(SEQUENCE_WRAPPERS)
TOOL_WRAPPERS.update(INTERACTION_WRAPPERS)
TOOL_WRAPPERS.update(PHARMACOPHORE_WRAPPERS)
TOOL_WRAPPERS.update(CLASSICAL_WRAPPERS)


def get_wrapper(tool_name: str, tool_config: dict, bench_root: str):
    """Get the appropriate wrapper instance for a tool.

    Args:
        tool_name: Name of the tool (must be in tools_config.json)
        tool_config: Configuration dict for this tool
        bench_root: Root directory of the benchmark

    Returns:
        Wrapper instance, or None if tool not found
    """
    wrapper_class = TOOL_WRAPPERS.get(tool_name)
    if wrapper_class is None:
        # Fallback: use category-based default
        category = tool_config.get('category', 'pocket3d')
        if category == 'pocket3d':
            wrapper_class = Pocket3DWrapper
        elif category == 'sequence':
            wrapper_class = SequenceWrapper
        elif category == 'interaction':
            wrapper_class = InteractionWrapper
        elif category == 'pharmacophore':
            wrapper_class = PharmacophoreWrapper
        elif category == 'classical':
            wrapper_class = CLASSICAL_WRAPPERS.get('AutoGrow4', BaseWrapper)
        else:
            return None

    return wrapper_class(tool_name, tool_config, bench_root)
