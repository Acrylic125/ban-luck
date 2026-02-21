"""
Dealer abstraction for the card game.

Implementations:
- SimpleDealer: reveal on 17+, else draw one (house rule).
- PolicyBasedDealer: uses a learned RL policy (state -> REVEAL or DRAW).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from game import Action, Game, best_hand_value


class Dealer(ABC):
    """Abstract base for the dealer. Chooses REVEAL or DRAW (one card at a time)."""

    @abstractmethod
    def choose_action(self, game: Game) -> Action:
        """
        Choose an action for the dealer's turn.

        :param game: current game state (dealer is game.players[0])
        :return: Action.REVEAL or Action.DRAW
        """
        ...


class SimpleDealer(Dealer):
    """
    House rule: reveal on 17+, otherwise draw one card at a time
    (caller repeats until REVEAL or 5 cards).
    """

    def choose_action(self, game: Game) -> Action:
        dealer = game.players[0]
        val = best_hand_value(dealer.hand)
        if val >= 17:
            return Action.REVEAL
        if len(dealer.hand) >= 5:
            return Action.REVEAL
        return Action.DRAW


# Dealer action indices for RL: 0 = REVEAL, 1 = DRAW
DEALER_NUM_ACTIONS = 2


def dealer_action_to_enum(action: int) -> Action:
    """Map dealer action index to Action.REVEAL or Action.DRAW."""
    return Action.REVEAL if action == 0 else Action.DRAW


def get_dealer_legal_actions(game: Game) -> list[int]:
    """Legal dealer action indices: REVEAL (0) always; DRAW (1) only if can draw more."""
    d = game.players[0]
    if len(d.hand) >= 5 or best_hand_value(d.hand) >= 21:
        return [0]  # must reveal
    return [0, 1]


class PolicyBasedDealer(Dealer):
    """Uses a learned policy: state string (hand) -> action index (0=REVEAL, 1=DRAW)."""

    def __init__(self, policy: dict[str, int], *, epsilon: float = 0.0):
        self._policy = policy
        self._epsilon = epsilon

    def choose_action(self, game: Game) -> Action:
        from agent import state_from_hand
        state = state_from_hand(game.players[0].hand)
        legal = get_dealer_legal_actions(game)
        if self._epsilon > 0 and random.random() < self._epsilon:
            action = random.choice(legal)
        else:
            action = self._policy.get(state, 0)
            if action not in legal:
                action = legal[0]
        return dealer_action_to_enum(action)
