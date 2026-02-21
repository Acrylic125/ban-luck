"""
Player abstraction for the card game.

Implementations:
- PolicyBasedPlayer: uses a learned RL policy (state -> action).
- SimplePlayer: same strategy as the current dealer (hold on 17+, else draw up to 3).
"""

from __future__ import annotations

from agent import (
    action_to_hold_or_draw,
    get_legal_actions,
    state_from_hand,
)
import random
from abc import ABC, abstractmethod

from game import Action, Game, best_hand_value


class Player(ABC):
    @abstractmethod
    def choose_action(self, game: Game, position: int) -> tuple[Action, int]:
        """Action and id"""
        ...


class SimplePlayer(Player):
    def choose_action(self, game: Game, position: int) -> Action:
        p = game.players[position]
        val = best_hand_value(p.hand)
        if val >= 17:
            return Action.HOLD, 0
        cards_left = 5 - len(p.hand)
        if cards_left == 0:
            return Action.HOLD, 0
        return Action.DRAW, 1


class PolicyBasedPlayer(Player):
    def __init__(
        self,
        policy: dict[str, int],
        *,
        epsilon: float = 0.0,
    ):
        self._policy = policy
        self._epsilon = epsilon

    def choose_action(self, game: Game, position: int) -> tuple[Action, int]:
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
