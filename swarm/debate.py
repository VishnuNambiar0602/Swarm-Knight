"""Debate and refinement logic for multi-LLM collaboration."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .models import (
    Critique,
    DebateRound,
    ParticipantRole,
    SwarmParticipant,
    SwarmStatus,
)
from .providers import OpenSourceProvider, create_provider

logger = logging.getLogger(__name__)


CRITIQUE_SYSTEM_PROMPT = """You are an expert code reviewer and critic. Your role is to analyze code solutions 
and provide constructive, specific feedback. Be thorough but fair.

Focus on:
1. Correctness and bugs
2. Code quality and best practices
3. Performance and efficiency
4. Security concerns
5. Maintainability and readability
6. Missing features or edge cases

Format your response as:
STRENGTHS:
- [list specific strengths]

WEAKNESSES:
- [list specific weaknesses]

SUGGESTIONS:
- [list actionable improvements]

SCORE: [0-10 rating of overall quality]"""


REFINEMENT_SYSTEM_PROMPT = """You are an expert developer refining code based on feedback. 
Your goal is to improve the solution while maintaining its core functionality.

Instructions:
1. Address ALL weaknesses identified in the critiques
2. Implement the suggested improvements where appropriate
3. Keep the strengths of the original solution
4. Ensure the refined code is complete and runnable
5. Add comments explaining significant changes

Output ONLY the refined code with no additional explanation."""


class DebateManager:
    """Manages debate rounds and refinement cycles."""

    def __init__(self):
        self.providers: dict[str, OpenSourceProvider] = {}

    def _get_provider(self, participant: SwarmParticipant) -> OpenSourceProvider:
        """Get or create a provider for a participant."""
        if participant.id not in self.providers:
            self.providers[participant.id] = create_provider(participant)
        return self.providers[participant.id]

    async def close(self):
        """Close all provider connections."""
        for provider in self.providers.values():
            await provider.close()
        self.providers.clear()

    async def generate_initial_solutions(
        self,
        task: str,
        participants: list[SwarmParticipant],
    ) -> dict[str, str]:
        """Have all participants generate initial solutions in parallel."""
        logger.info(f"Generating initial solutions from {len(participants)} participants")

        async def _generate_one(participant: SwarmParticipant) -> tuple[str, str]:
            provider = self._get_provider(participant)
            try:
                solution = await provider.generate_with_retry(
                    prompt=task,
                    system_prompt=self._get_generation_prompt(participant),
                    max_retries=3,
                )
                return participant.id, solution
            except Exception as e:
                logger.error(f"Participant {participant.name} failed to generate: {e}")
                return participant.id, f"Error: {str(e)}"

        tasks = [_generate_one(p) for p in participants]
        results = await asyncio.gather(*tasks)

        return dict(results)

    async def run_debate_round(
        self,
        task: str,
        current_solutions: dict[str, str],
        participants: list[SwarmParticipant],
        round_number: int,
    ) -> DebateRound:
        """Run a single round of debate and refinement."""
        logger.info(f"Starting debate round {round_number}")

        round_data = DebateRound(round_number=round_number)
        round_data.solutions = current_solutions.copy()

        # Phase 1: Cross-critique
        critiques = await self._generate_critiques(
            task, current_solutions, participants
        )
        round_data.critiques = critiques

        # Phase 2: Refinement based on critiques
        refinements = await self._refine_solutions(
            task, current_solutions, critiques, participants
        )
        round_data.refinements = refinements

        # Calculate consensus score
        round_data.consensus_score = self._calculate_consensus(refinements)
        round_data.completed_at = datetime.now()

        logger.info(f"Round {round_number} completed. Consensus: {round_data.consensus_score:.2f}")
        return round_data

    async def _generate_critiques(
        self,
        task: str,
        solutions: dict[str, str],
        participants: list[SwarmParticipant],
    ) -> list[Critique]:
        """Have each participant critique all other solutions."""
        critiques = []

        # Filter to participants who can critique
        critics = [p for p in participants if p.role in (ParticipantRole.CRITIC, ParticipantRole.COORDINATOR)]
        if not critics:
            critics = participants  # All participate in critique

        async def _critique_one(
            critic: SwarmParticipant,
            target_id: str,
            target_solution: str,
        ) -> Critique:
            provider = self._get_provider(critic)
            try:
                critique_prompt = f"""Task: {task}

Solution to critique (by participant {target_id}):

{target_solution}

Provide a detailed critique of this solution."""

                response = await provider.generate_with_retry(
                    prompt=critique_prompt,
                    system_prompt=CRITIQUE_SYSTEM_PROMPT,
                    max_retries=2,
                )

                return self._parse_critique(critic.id, target_id, response)
            except Exception as e:
                logger.error(f"Critique failed for {critic.name} on {target_id}: {e}")
                return Critique(
                    critic_id=critic.id,
                    target_id=target_id,
                    content=str(e),
                    score=0.0,
                )

        # Generate critiques in parallel
        critique_tasks = []
        for critic in critics:
            for target_id, solution in solutions.items():
                if target_id != critic.id:  # Don't critique your own
                    critique_tasks.append(_critique_one(critic, target_id, solution))

        critique_results = await asyncio.gather(*critique_tasks)
        critiques.extend(critique_results)

        return critiques

    async def _refine_solutions(
        self,
        task: str,
        current_solutions: dict[str, str],
        critiques: list[Critique],
        participants: list[SwarmParticipant],
    ) -> dict[str, str]:
        """Have each participant refine their solution based on critiques."""
        refinements = {}

        async def _refine_one(
            participant: SwarmParticipant,
            original_solution: str,
            received_critiques: list[Critique],
        ) -> tuple[str, str]:
            provider = self._get_provider(participant)
            try:
                # Format critiques for the participant
                critique_text = self._format_critiques_for_refinement(received_critiques)

                refinement_prompt = f"""Original Task: {task}

Your Original Solution:
{original_solution}

Feedback from other participants:
{critique_text}

Refine your solution based on this feedback. Address the weaknesses and implement the suggestions while keeping the strengths."""

                refined = await provider.generate_with_retry(
                    prompt=refinement_prompt,
                    system_prompt=REFINEMENT_SYSTEM_PROMPT,
                    max_retries=2,
                )
                return participant.id, refined
            except Exception as e:
                logger.error(f"Refinement failed for {participant.name}: {e}")
                return participant.id, original_solution  # Keep original on failure

        # Refine in parallel
        refine_tasks = []
        for participant in participants:
            if participant.id in current_solutions:
                # Get critiques targeting this participant
                target_critiques = [c for c in critiques if c.target_id == participant.id]
                refine_tasks.append(
                    _refine_one(participant, current_solutions[participant.id], target_critiques)
                )

        refine_results = await asyncio.gather(*refine_tasks)
        refinements = dict(refine_results)

        return refinements

    def _parse_critique(self, critic_id: str, target_id: str, content: str) -> Critique:
        """Parse a critique response into structured format."""
        strengths = []
        weaknesses = []
        suggestions = []
        score = 5.0

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "STRENGTH" in line.upper():
                current_section = "strengths"
            elif "WEAKNESS" in line.upper() or "ISSUE" in line.upper():
                current_section = "weaknesses"
            elif "SUGGESTION" in line.upper() or "IMPROVEMENT" in line.upper():
                current_section = "suggestions"
            elif "SCORE" in line.upper():
                try:
                    score = float(line.split(":")[-1].strip().split("/")[0])
                    score = max(0.0, min(10.0, score))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("-") or line.startswith("*"):
                item = line.lstrip("-* ").strip()
                if current_section == "strengths":
                    strengths.append(item)
                elif current_section == "weaknesses":
                    weaknesses.append(item)
                elif current_section == "suggestions":
                    suggestions.append(item)

        return Critique(
            critic_id=critic_id,
            target_id=target_id,
            content=content,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            score=score,
        )

    def _format_critiques_for_refinement(self, critiques: list[Critique]) -> str:
        """Format critiques into a readable string for refinement."""
        if not critiques:
            return "No critiques received."

        parts = []
        for c in critiques:
            parts.append(f"From Critic {c.critic_id} (Score: {c.score}/10):")
            if c.weaknesses:
                parts.append("  Issues found:")
                for w in c.weaknesses[:5]:  # Limit to top 5
                    parts.append(f"    - {w}")
            if c.suggestions:
                parts.append("  Suggestions:")
                for s in c.suggestions[:5]:
                    parts.append(f"    - {s}")
            parts.append("")

        return "\n".join(parts)

    def _calculate_consensus(self, solutions: dict[str, str]) -> float:
        """Calculate consensus score based on solution similarity."""
        if len(solutions) <= 1:
            return 1.0

        # Simple heuristic: check if solutions have converged
        # by comparing lengths and key patterns
        lengths = [len(s) for s in solutions.values()]
        avg_length = sum(lengths) / len(lengths)

        if avg_length == 0:
            return 0.0

        # Normalize length similarity
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        length_similarity = 1.0 / (1.0 + (length_variance / (avg_length ** 2)))

        # Check for common patterns (imports, class names, function names)
        all_words = []
        for solution in solutions.values():
            words = set(solution.lower().split())
            all_words.append(words)

        if all_words:
            # Calculate Jaccard similarity between all pairs
            total_similarity = 0
            pairs = 0
            for i in range(len(all_words)):
                for j in range(i + 1, len(all_words)):
                    intersection = len(all_words[i] & all_words[j])
                    union = len(all_words[i] | all_words[j])
                    if union > 0:
                        total_similarity += intersection / union
                        pairs += 1
            word_similarity = total_similarity / pairs if pairs > 0 else 0
        else:
            word_similarity = 0

        # Combine metrics
        consensus = (length_similarity * 0.3 + word_similarity * 0.7)
        return min(1.0, max(0.0, consensus))

    def _get_generation_prompt(self, participant: SwarmParticipant) -> str:
        """Get a tailored system prompt based on participant role."""
        role_prompts = {
            ParticipantRole.GENERATOR: "You are an expert developer. Generate high-quality, complete code.",
            ParticipantRole.CRITIC: "You are an expert reviewer. Generate code while keeping review criteria in mind.",
            ParticipantRole.REFINER: "You are an expert at improving code. Generate clean, efficient code.",
            ParticipantRole.COORDINATOR: "You are a technical lead. Generate well-architected, comprehensive solutions.",
        }
        return role_prompts.get(participant.role, role_prompts[ParticipantRole.GENERATOR])
