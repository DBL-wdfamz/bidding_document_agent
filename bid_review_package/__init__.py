"""
Bid Review Package - 招标评审和文书生成系统
"""

from .content_extractor import extract_content
from .draft_generator import generate_draft, generate_draft_stream

__version__ = "1.0.0"
__all__ = [
    "extract_content",
    "generate_draft",
    "generate_draft_stream"
]
