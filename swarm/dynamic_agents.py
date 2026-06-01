"""Dynamic Agent Generation - Auto-generate agents based on user query."""

from __future__ import annotations

import logging
import re
from typing import Optional

from .models import (
    FREE_MODELS,
    ParticipantRole,
    ProviderType,
    SwarmParticipant,
    TaskPlan,
)

logger = logging.getLogger(__name__)


TASK_KEYWORDS = {
    "code": ["code", "implement", "build", "create", "write", "function", "class", "api", "endpoint", "component"],
    "design": ["design", "layout", "ui", "ux", "visual", "style", "color", "theme", "css", "tailwind"],
    "review": ["review", "audit", "check", "test", "validate", "fix", "debug", "error"],
    "plan": ["plan", "architect", "structure", "organize", "design", "schema", "flow"],
    "data": ["data", "parse", "extract", "transform", "etl", "csv", "json", "database"],
    "document": ["document", "readme", "docs", "comment", "explain", "describe"],
}

ROLE_ASSIGNMENTS = {
    "code": [ParticipantRole.GENERATOR, ParticipantRole.GENERATOR, ParticipantRole.CRITIC],
    "design": [ParticipantRole.GENERATOR, ParticipantRole.GENERATOR, ParticipantRole.CRITIC],
    "review": [ParticipantRole.CRITIC, ParticipantRole.CRITIC, ParticipantRole.GENERATOR],
    "plan": [ParticipantRole.PLANNER, ParticipantRole.GENERATOR, ParticipantRole.CRITIC],
    "data": [ParticipantRole.GENERATOR, ParticipantRole.GENERATOR, ParticipantRole.REVIEWER],
    "document": [ParticipantRole.GENERATOR, ParticipantRole.REVIEWER, ParticipantRole.CRITIC],
}

MODEL_ASSIGNMENTS = {
    "code": ["coder", "assembler", "reviewer"],
    "design": ["designer", "stylist", "reviewer"],
    "review": ["reviewer", "planner", "fast"],
    "plan": ["planner", "architect", "reviewer"],
    "data": ["extractor", "tagger", "reviewer"],
    "document": ["assembler", "reviewer", "fast"],
}


class DynamicAgentGenerator:
    """Generate agents dynamically based on user query analysis."""

    def __init__(self):
        self.task_patterns = self._build_patterns()

    def _build_patterns(self) -> dict[str, list[str]]:
        return TASK_KEYWORDS

    def analyze_task(self, task: str) -> TaskPlan:
        task_lower = task.lower()
        scores: dict[str, int] = {}
        for category, keywords in self.task_patterns.items():
            scores[category] = sum(1 for kw in keywords if kw in task_lower)

        primary_task = max(scores, key=scores.get) if any(scores.values()) else "code"
        complexity = min(1.0, len(task.split()) / 50 + 0.3)

        subtasks = self._decompose_task(task, primary_task)
        roles = ROLE_ASSIGNMENTS.get(primary_task, ROLE_ASSIGNMENTS["code"])
        estimated_rounds = 3 if complexity < 0.5 else 5

        return TaskPlan(
            task_type=primary_task,
            subtasks=subtasks,
            required_roles=roles,
            complexity=complexity,
            estimated_rounds=estimated_rounds,
        )

    def _decompose_task(self, task: str, task_type: str) -> list[str]:
        subtasks = []
        if task_type == "code":
            subtasks = ["Implement core logic", "Write tests", "Review code quality"]
        elif task_type == "design":
            subtasks = ["Create layout structure", "Define visual style", "Review accessibility"]
        elif task_type == "review":
            subtasks = ["Identify issues", "Suggest improvements", "Validate fixes"]
        elif task_type == "plan":
            subtasks = ["Analyze requirements", "Design architecture", "Create implementation plan"]
        elif task_type == "data":
            subtasks = ["Parse input data", "Transform and validate", "Output results"]
        elif task_type == "document":
            subtasks = ["Generate documentation", "Review accuracy", "Format output"]
        return subtasks

    def generate_agents(
        self,
        task: str,
        max_agents: int = 6,
        api_key: Optional[str] = None,
        reputation_scores: Optional[dict[str, float]] = None,
    ) -> list[SwarmParticipant]:
        plan = self.analyze_task(task)
        agents = []
        model_keys = MODEL_ASSIGNMENTS.get(plan.task_type, MODEL_ASSIGNMENTS["code"])

        for i, model_key in enumerate(model_keys[:max_agents]):
            if model_key not in FREE_MODELS:
                continue
            model_info = FREE_MODELS[model_key]
            role = plan.required_roles[i] if i < len(plan.required_roles) else ParticipantRole.GENERATOR

            agent = SwarmParticipant(
                name=model_info["name"],
                provider=ProviderType.OPENROUTER,
                model=model_info["model"],
                role=role,
                api_key=api_key,
                context_window=model_info["ctx"],
                auto_generated=True,
                task_description=plan.subtasks[i] if i < len(plan.subtasks) else task,
            )
            agents.append(agent)

        if len(agents) < 2:
            agents = self._get_fallback_agents(api_key)

        return agents[:max_agents]

    def _get_fallback_agents(self, api_key: Optional[str] = None) -> list[SwarmParticipant]:
        return [
            SwarmParticipant(
                name="Laguna Coder",
                provider=ProviderType.OPENROUTER,
                model=FREE_MODELS["coder"]["model"],
                role=ParticipantRole.GENERATOR,
                api_key=api_key,
                auto_generated=True,
            ),
            SwarmParticipant(
                name="Nemotron Reviewer",
                provider=ProviderType.OPENROUTER,
                model=FREE_MODELS["reviewer"]["model"],
                role=ParticipantRole.CRITIC,
                api_key=api_key,
                auto_generated=True,
            ),
        ]

    def rank_agents_by_reputation(
        self,
        agents: list[SwarmParticipant],
        reputation_scores: dict[str, float],
    ) -> list[SwarmParticipant]:
        return sorted(
            agents,
            key=lambda a: reputation_scores.get(a.id, 0.5),
            reverse=True,
        )


dynamic_generator = DynamicAgentGenerator()
