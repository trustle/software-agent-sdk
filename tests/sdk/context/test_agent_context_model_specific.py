"""Tests covering how CLAUDE.md and GEMINI.md repo-skill files render in the
system-message suffix.

Fork-specific behavior: upstream SDK gates these files out when the active
LLM family doesn't match (drops CLAUDE.md when running Gemini, etc.). The
Trustle fork treats CLAUDE.md and GEMINI.md as universal repo guidance and
always renders them regardless of the active model — see commit aa87ec89
("fork: drop vendor-gate on CLAUDE.md/GEMINI.md repo skills").

These tests assert the new contract to prevent silent regressions if the gate
is ever re-introduced.
"""

from pathlib import Path

import pytest

from openhands.sdk.context.agent_context import AgentContext
from openhands.sdk.skills import load_project_skills


_REPO_BASELINE_TEXT = (
    "---\n# type: repo\nversion: 1.0.0\nagent: CodeActAgent\n---\n\nRepo baseline\n"
)
# Different baseline formats for testing backward compatibility:
# - _REPO_BASELINE_TEXT: legacy format with frontmatter (used in
#   .openhands/skills/repo.md)
# - _AGENTS_BASELINE_TEXT: simple markdown format (used in AGENTS.md)
_AGENTS_BASELINE_TEXT = "# Project Guidelines\n\nRepo baseline\n"


def _write_repo_with_vendor_files(root: Path, baseline_source: str) -> None:
    """Create test repository with baseline and vendor-specific skill files.

    Args:
        root: Root directory for the test repository
        baseline_source: Either "repo_md" (legacy .openhands/skills/repo.md)
                        or "agents_md" (AGENTS.md in repo root)
    """
    if baseline_source == "repo_md":
        skills_dir = root / ".openhands" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "repo.md").write_text(_REPO_BASELINE_TEXT)
    elif baseline_source == "agents_md":
        (root / "AGENTS.md").write_text(_AGENTS_BASELINE_TEXT)
    else:
        raise ValueError(f"Unknown baseline_source: {baseline_source}")

    (root / "claude.md").write_text("Claude-Specific Instructions")
    (root / "gemini.md").write_text("Gemini-Specific Instructions")


# Parametrize over LLM families that used to trigger vendor gating upstream.
# All three must now render both CLAUDE.md and GEMINI.md content.
@pytest.mark.parametrize("baseline_source", ["repo_md", "agents_md"])
@pytest.mark.parametrize(
    "llm_model",
    [
        "litellm_proxy/anthropic/claude-sonnet-4",
        "gemini-2.5-pro",
        "openai/gpt-4o",
    ],
)
def test_context_always_includes_vendor_files_regardless_of_model(
    tmp_path: Path, baseline_source: str, llm_model: str
):
    _write_repo_with_vendor_files(tmp_path, baseline_source)
    skills = load_project_skills(tmp_path)
    ac = AgentContext(skills=skills)
    suffix = ac.get_system_message_suffix(llm_model=llm_model)
    assert suffix is not None
    assert "Repo baseline" in suffix
    assert "Claude-Specific Instructions" in suffix
    assert "Gemini-Specific Instructions" in suffix


@pytest.mark.parametrize("baseline_source", ["repo_md", "agents_md"])
def test_context_always_includes_vendor_files_with_canonical_name(
    tmp_path: Path, baseline_source: str
):
    """Exercise the llm_model_canonical fallback path — behavior is unchanged:
    both vendor files render."""
    _write_repo_with_vendor_files(tmp_path, baseline_source)
    skills = load_project_skills(tmp_path)
    ac = AgentContext(skills=skills)
    suffix = ac.get_system_message_suffix(
        llm_model="proxy/test-model",
        llm_model_canonical="anthropic/claude-sonnet-4",
    )
    assert suffix is not None
    assert "Repo baseline" in suffix
    assert "Claude-Specific Instructions" in suffix
    assert "Gemini-Specific Instructions" in suffix


@pytest.mark.parametrize("baseline_source", ["repo_md", "agents_md"])
def test_context_includes_all_when_model_unknown(tmp_path: Path, baseline_source: str):
    """When no model info is provided at all, both vendor files render
    (same as the per-model cases above — there is no model-aware gating)."""
    _write_repo_with_vendor_files(tmp_path, baseline_source)
    skills = load_project_skills(tmp_path)
    ac = AgentContext(skills=skills)
    suffix = ac.get_system_message_suffix()
    assert suffix is not None
    assert "Repo baseline" in suffix
    assert "Claude-Specific Instructions" in suffix
    assert "Gemini-Specific Instructions" in suffix
