"""Debate protocol definitions.

Defines the structure and rules of a Ploidy debate, including:
- Debate phases and valid transitions between them
- Message types exchanged during debate
- Constraints and termination conditions

The protocol ensures debates are structured rather than free-form,
making disagreements interpretable and convergence tractable.

Phases:
    1. INDEPENDENT  -- All sessions analyze the prompt independently
    2. POSITION     -- Each session declares their stance
    3. CHALLENGE    -- Each session critiques the others' positions
    4. CONVERGENCE  -- Find common ground or articulate irreducible disagreement
    5. COMPLETE     -- Debate concluded, result available
"""

from dataclasses import dataclass
from enum import Enum

from ploidy.exceptions import ProtocolError


class DebatePhase(Enum):
    """Phases of a structured debate."""

    INDEPENDENT = "independent"
    POSITION = "position"
    CHALLENGE = "challenge"
    CONVERGENCE = "convergence"
    COMPLETE = "complete"


class SemanticAction(Enum):
    """Semantic actions a session can take during a debate.

    These classify the *intent* of a debate message, making
    the debate transcript machine-interpretable for convergence analysis.
    """

    AGREE = "agree"
    CHALLENGE = "challenge"
    PROPOSE_ALTERNATIVE = "propose_alternative"
    SYNTHESIZE = "synthesize"


@dataclass
class DebateMessage:
    """A single message in the debate protocol.

    Attributes:
        session_id: Which session authored this message.
        phase: The debate phase this message belongs to.
        content: The message content.
        timestamp: When the message was created.
        action: Optional semantic action classifying this message's intent.
    """

    session_id: str
    phase: DebatePhase
    content: str
    timestamp: str
    action: SemanticAction | None = None


class DebateProtocol:
    """Manages the state machine of a single debate.

    Enforces valid phase transitions and tracks which sessions
    have submitted their contributions for each phase.
    """

    VALID_TRANSITIONS: dict[DebatePhase, DebatePhase] = {
        DebatePhase.INDEPENDENT: DebatePhase.POSITION,
        DebatePhase.POSITION: DebatePhase.CHALLENGE,
        DebatePhase.CHALLENGE: DebatePhase.CONVERGENCE,
        DebatePhase.CONVERGENCE: DebatePhase.COMPLETE,
    }

    def __init__(self, debate_id: str, prompt: str) -> None:
        """Initialize a new debate protocol instance.

        Args:
            debate_id: Unique identifier for this debate.
            prompt: The decision prompt all sessions will address.
        """
        self.debate_id = debate_id
        self.prompt = prompt
        self.phase = DebatePhase.INDEPENDENT
        self.messages: list[DebateMessage] = []

    def advance_phase(self) -> DebatePhase:
        """Attempt to advance to the next debate phase.

        Returns:
            The new phase after advancement.

        Raises:
            ProtocolError: If advancement conditions are not met.
        """
        next_phase = self.VALID_TRANSITIONS.get(self.phase)
        if next_phase is None:
            raise ProtocolError(f"Cannot advance from terminal phase {self.phase.value}")
        self.phase = next_phase
        return self.phase

    def submit_message(self, message: DebateMessage) -> None:
        """Submit a message to the current phase.

        Args:
            message: The debate message to record.

        Raises:
            ProtocolError: If the message is not valid for the current phase.
        """
        if message.phase != self.phase:
            raise ProtocolError(
                f"Message phase {message.phase.value} does not match "
                f"current phase {self.phase.value}"
            )
        if message.phase in {DebatePhase.POSITION, DebatePhase.CHALLENGE} and self.has_submitted(
            message.session_id, message.phase
        ):
            raise ProtocolError(
                f"Session {message.session_id} already submitted a {message.phase.value} message"
            )
        self.messages.append(message)

    def has_submitted(self, session_id: str, phase: DebatePhase) -> bool:
        """Return whether a session already submitted in the given phase."""
        return any(
            message.session_id == session_id and message.phase == phase for message in self.messages
        )

    def _all_sessions_submitted(
        self,
        session_ids: set[str],
        phase: DebatePhase,
    ) -> bool:
        """Return whether every member of a valid roster submitted once."""
        submitted = {message.session_id for message in self.messages if message.phase == phase}
        return len(session_ids) >= 2 and session_ids <= submitted

    def all_positions_submitted(self, session_ids: set[str]) -> bool:
        """Return whether every participating session has submitted a position.

        At least two sessions are required before the position barrier can
        open.  This preserves the Independent -> Position anti-anchoring
        contract even when the Deep session submits before a Fresh session
        has joined.
        """
        return self._all_sessions_submitted(session_ids, DebatePhase.POSITION)

    def all_challenges_submitted(self, session_ids: set[str]) -> bool:
        """Return whether every frozen-roster session submitted a challenge."""
        return self._all_sessions_submitted(session_ids, DebatePhase.CHALLENGE)
