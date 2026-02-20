"""
Player abstraction for the card game.

Implementations:
- PolicyBasedPlayer: uses a learned RL policy (state -> action).
- SimplePlayer: same strategy as the current dealer (hold on 17+, else draw up to 3).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from game import Action, Game, best_hand_value


class Player(ABC):
    """Abstract base for a player (not the dealer). Chooses HOLD or DRAW (1–3 cards)."""

    @abstractmethod
    def choose_action(self, game: Game, position: int) -> tuple[Action, int]:
        """
        Choose an action for the current turn.

        :param game: current game state
        :param position: this player's seat (1 to N-1; 0 is dealer)
        :return: (Action.HOLD, 0) or (Action.DRAW, n) with n in 1..3
        """
        ...


class SimplePlayer(Player):
    """
    Same strategy as the current dealer bot: hold on 17+, otherwise draw
    up to 3 cards (no REVEAL; that is dealer-only).
    """

    def choose_action(self, game: Game, position: int) -> tuple[Action, int]:
        p = game.players[position]
        val = best_hand_value(p.hand)
        if val >= 17:
            return Action.HOLD, 0
        cards_left = 5 - len(p.hand)
        if cards_left == 0:
            return Action.HOLD, 0
        return Action.DRAW, min(3, cards_left)


class PolicyBasedPlayer(Player):
    """
    Uses a learned policy: state (hand_value, usable_ace) -> action index.
    Expects the same policy format as the MC-trained agent (e.g. from agent_policy.json).
    """

    def __init__(
        self,
        policy: dict[tuple[int, int], int],
        *,
        epsilon: float = 0.0,
    ):
        """
        :param policy: state (value, usable_ace) -> action index (0=hold, 1–3=draw 1–3)
        :param epsilon: probability of random action (0 = greedy)
        """
        self._policy = policy
        self._epsilon = epsilon

    def choose_action(self, game: Game, position: int) -> tuple[Action, int]:
        from agent import (
            action_to_hold_or_draw,
            get_legal_actions,
            state_from_hand,
        )
        p = game.players[position]
        state = state_from_hand(p.hand)
        legal = get_legal_actions(len(p.hand))
        if self._epsilon > 0 and random.random() < self._epsilon:
            action = random.choice(legal)
        else:
            action = self._policy.get(state, 0)
            if action not in legal:
                action = legal[0]
        return action_to_hold_or_draw(action)
