"""
Deck creation (standard sorted order) and shuffle strategies.

Initial deck order: standard playing order 1 (Ace), 2, 3, ... 10, K, Q, J per suit.
Caller can specify different strategies for first shuffle vs subsequent shuffles.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

# Deck is a list of cards (game.Card); typed as list to avoid circular import.


class DeckShuffleStrategy(ABC):
    """Base class for how to shuffle a deck (in-place)."""

    @abstractmethod
    def shuffle(self, deck: list[Any], *, is_first: bool = True) -> None:
        """
        Shuffle the deck in place.

        :param deck: list of cards (modified in place)
        :param is_first: True for the first shuffle of a session, False for subsequent shuffles
        """
        ...


class WashShuffleStrategy(DeckShuffleStrategy):
    """Randomly sort the cards (full random shuffle)."""

    def shuffle(self, deck: list[Any], *, is_first: bool = True) -> None:
        random.shuffle(deck)


class DeckCuttingStrategy(DeckShuffleStrategy):
    """
    Each step: split deck into top and bottom at a random proportion (proportion_min/max).
    With probability deck_interleaving_probability: split the top chunk into two (at
    interleaving_proportion_min/max) and put the bottom chunk in between (top1 + bottom + top2).
    Otherwise: move bottom to top (bottom + top). Repeated n times.
    """

    def __init__(
        self,
        proportion_min: float = 0.3,
        proportion_max: float = 0.7,
        deck_interleaving_probability: float = 1/3,
        interleaving_proportion_min: float = 0.3,
        interleaving_proportion_max: float = 0.7,
        n: int = 5,
    ):
        self.proportion_min = proportion_min
        self.proportion_max = proportion_max
        self.deck_interleaving_probability = deck_interleaving_probability
        self.interleaving_proportion_min = interleaving_proportion_min
        self.interleaving_proportion_max = interleaving_proportion_max
        self.n = n

    def _interleave(self, deck: list[Any], cut_index: int) -> None:
        """Split top chunk into two at interleaving proportion; result = top1 + bottom + top2."""
        top = deck[:cut_index]
        bottom = deck[cut_index:]
        proportion = random.uniform(
            self.interleaving_proportion_min, self.interleaving_proportion_max
        )
        split_at = int(len(top) * proportion)
        if split_at <= 0:
            split_at = 1
        if split_at >= len(top):
            split_at = len(top) - 1
        top1, top2 = top[:split_at], top[split_at:]
        deck[:] = top1 + bottom + top2

    def shuffle(self, deck: list[Any], *, is_first: bool = True) -> None:
        for _ in range(self.n):
            if len(deck) < 2:
                return
            proportion = random.uniform(self.proportion_min, self.proportion_max)
            cut_index = int(len(deck) * proportion)
            if cut_index <= 0 or cut_index >= len(deck):
                continue
            if random.random() < self.deck_interleaving_probability:
                # Interleave: split top into two, put bottom in between
                if cut_index >= 2:  # need at least 2 cards in top to split
                    self._interleave(deck, cut_index)
            else:
                # Cut: move bottom to top
                deck[:] = deck[cut_index:] + deck[:cut_index]
