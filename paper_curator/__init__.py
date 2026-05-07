"""Daily curated paper digest.

Pulls from arXiv / OSF / RSS feeds, ranks against scraped lab-member profiles
with a sentence-transformers bi-encoder, judges with an Ollama LLM, and posts
~10 picks to a Slack channel each weekday.

Public entrypoint:
- run_curation(dry_run=False, preview_dm=None) -> bool
"""
from .curator import run_curation  # noqa: F401
