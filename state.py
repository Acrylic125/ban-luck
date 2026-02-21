"""
Shared state representation and action helpers for the card game.

Used by players, dealer, and agent to avoid circular imports.
Depends only on game.
"""

from __future__ import annotations

from game import Action, Card


def serialize_card_state(card: Card) -> str:
    value = card.blackjack_value()
    if isinstance(value, tuple):
        return f"{value[0]}_{value[1]}"
    return str(value)


def state_from_hand(cards: list[Card]) -> str:
    """Canonical state string for a hand (sorted card values)."""
    v = [serialize_card_state(c) for c in cards]
    v.sort()
    return ",".join(v)


# Player actions: 0 = hold, 1–3 = draw 1–3
NUM_ACTIONS = 4


def action_to_hold_or_draw(action: int) -> tuple[Action, int]:
    """Map action index to (Action.HOLD or Action.DRAW, num_cards)."""
    if action == 0:
        return Action.HOLD, 0
    return Action.DRAW, action  # 1, 2, or 3


def get_legal_actions(num_cards: int) -> list[int]:
    """Legal action indices when agent has num_cards (2–5). At 2 cards: all 4; at 5: only hold."""
    if num_cards >= 5:
        return [0]
    max_draw = min(3, 5 - num_cards)
    return [0] + list(range(1, max_draw + 1))
