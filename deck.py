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
    A random proportion of the deck is cut to the top, repeated n times.

    Each cut: pick a proportion in [proportion_min, proportion_max] (e.g. 0.1 to 0.7),
    then move that many cards from the 'bottom' of the deck to the top.
    So deck becomes deck[cut_index:] + deck[:cut_index].
    """

    def __init__(
        self,
        proportion_min: float = 0.1,
        proportion_max: float = 0.7,
        n: int = 5,
    ):
        """
        :param proportion_min: minimum fraction of deck to cut (0–1)
        :param proportion_max: maximum fraction of deck to cut (0–1)
        :param n: number of cuts to perform
        """
        self.proportion_min = proportion_min
        self.proportion_max = proportion_max
        self.n = n

    def shuffle(self, deck: list[Any], *, is_first: bool = True) -> None:
        for _ in range(self.n):
            if len(deck) < 2:
                return
            proportion = random.uniform(self.proportion_min, self.proportion_max)
            cut_index = int(len(deck) * proportion)
            if cut_index <= 0 or cut_index >= len(deck):
                continue
            # Move deck[cut_index:] to top: new deck = bottom part + top part
            deck[:] = deck[cut_index:] + deck[:cut_index]
