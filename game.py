"""Core game logic for the card game (dealer + N-1 players, blackjack-style)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Suit(str, Enum):
    SPADES = "♠"
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"


class Rank(str, Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


# 10-value cards for blackjack (10, J, Q, K)
TEN_VALUE_RANKS = {Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING}


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"

    def blackjack_value(self) -> int | tuple[int, int]:
        """Return numeric value(s). Ace returns (1, 11)."""
        if self.rank == Rank.ACE:
            return (1, 11)
        if self.rank in TEN_VALUE_RANKS:
            return 10
        return int(self.rank.value)

    def is_ace(self) -> bool:
        return self.rank == Rank.ACE

    def is_ten_value(self) -> bool:
        return self.rank in TEN_VALUE_RANKS


def best_hand_value(cards: List[Card]) -> int:
    """Best non-bust total (<=21). If all totals bust, return minimum (soft)."""
    total = 0
    aces = 0
    for c in cards:
        v = c.blackjack_value()
        if isinstance(v, tuple):
            aces += 1
            total += 1
        else:
            total += v
    # Prefer 11 for aces when it doesn't bust
    while aces and total + 10 <= 21:
        total += 10
        aces -= 1
    return total


def _min_total(cards: List[Card]) -> int:
    """Minimum total (all aces as 1)."""
    t = 0
    for c in cards:
        v = c.blackjack_value()
        t += 1 if isinstance(v, tuple) else v
    return t


def is_natural_blackjack(cards: List[Card]) -> bool:
    """True if exactly two cards: Ace and a 10-value (10, J, Q, K)."""
    if len(cards) != 2:
        return False
    a, b = cards[0], cards[1]
    return (a.is_ace() and b.is_ten_value()) or (b.is_ace() and a.is_ten_value())


def is_bust(cards: List[Card]) -> bool:
    """True if minimum total (all aces as 1) > 21."""
    return _min_total(cards) > 21


def raw_total(cards: List[Card]) -> int:
    """Total using best interpretation (Ace 1 or 11). Same as best_hand_value but can exceed 21."""
    return best_hand_value(cards) if not is_bust(cards) else _min_total(cards)


def make_deck() -> List[Card]:
    return [Card(s, r) for s in Suit for r in Rank]


class Action(str, Enum):
    HOLD = "hold"
    DRAW = "draw"
    REVEAL = "reveal"  # dealer only


@dataclass
class PlayerState:
    """State for one seat (dealer or player)."""
    position: int  # k: 0 = dealer, 1..N-1 = players
    hand: List[Card] = field(default_factory=list)
    reward: Optional[int] = None  # set when outcome is decided
    done: bool = False  # natural blackjack, or drew (outcome set), or compared at reveal
    drew_cards: bool = False  # chose to draw (so reward already set if done)

    def is_dealer(self) -> bool:
        return self.position == 0

    def hand_value(self) -> int:
        return best_hand_value(self.hand)

    def hand_value_for_compare(self) -> int:
        """Value for comparison. If bust, return 0 so dealer wins vs this hand."""
        if is_bust(self.hand):
            return 0
        return best_hand_value(self.hand)

    def can_draw_more(self) -> bool:
        return len(self.hand) < 5 and not self.done


class Game:
    """Single round: one dealer, positions 1..n_players (N = n_players + 1)."""

    def __init__(self, n_players: int):
        assert n_players >= 1
        self.n_players = n_players
        self.N = n_players + 1  # dealer + players
        self.deck: List[Card] = []
        self.players: List[PlayerState] = []
        self.current_turn: int = 1  # 1 to n_players, then 0 for dealer
        self.revealed: bool = False

    def _seat_order_for_deal(self) -> List[int]:
        """Order to deal: first card to 1, 2, ..., n_players, 0; then same again."""
        return list(range(1, self.N)) + [0]

    def deal(self) -> None:
        self.deck = make_deck()
        random.shuffle(self.deck)
        self.players = [PlayerState(position=k) for k in range(self.N)]
        order = self._seat_order_for_deal()
        for _ in range(2):
            for k in order:
                self.players[k].hand.append(self.deck.pop())
        self.current_turn = 1
        self.revealed = False
        # Check natural blackjacks (only for players, not dealer) after deal
        for k in range(1, self.N):
            p = self.players[k]
            if is_natural_blackjack(p.hand):
                p.reward = 2
                p.done = True

    def turn_order(self) -> List[int]:
        """Order of turns: 1, 2, ..., n_players, then 0 (dealer)."""
        return list(range(1, self.N)) + [0]

    def current_player(self) -> PlayerState:
        return self.players[self.current_turn]

    def advance_turn(self) -> None:
        """Move to next turn (next in circle: 1..N-1 then 0)."""
        order = self.turn_order()
        idx = order.index(self.current_turn)
        self.current_turn = order[(idx + 1) % len(order)]

    def apply_hold(self, position: int) -> None:
        p = self.players[position]
        p.done = True
        # Reward set later when dealer reveals (unless they drew)
        if not p.drew_cards:
            pass  # reward stays None until reveal

    def apply_draw(self, position: int, num_cards: int) -> None:
        p = self.players[position]
        assert p.can_draw_more() and 1 <= num_cards <= min(3, 5 - len(p.hand))
        for _ in range(num_cards):
            if self.deck:
                p.hand.append(self.deck.pop())
        p.drew_cards = True
        p.done = True
        total = best_hand_value(p.hand)
        if is_bust(p.hand):
            p.reward = -2
        elif total == 21:
            p.reward = 3
        else:
            p.reward = 2

    def dealer_reveal(self) -> None:
        """Dealer reveals: compare dealer hand to each player who held (not drew)."""
        self.revealed = True
        dealer = self.players[0]
        X = dealer.hand_value_for_compare() if not is_bust(dealer.hand) else 0
        for k in range(1, self.N):
            p = self.players[k]
            if p.reward is not None:
                continue  # already has reward (natural or drew)
            if not p.done:
                continue
            # Held: compare
            Y = p.hand_value_for_compare()
            if X < Y:
                p.reward = 2 if Y == 21 else 1
            elif X == Y:
                p.reward = 0
            else:
                p.reward = -2 if Y == 21 else -1

    def all_turns_done(self) -> bool:
        return all(self.players[k].done for k in self.turn_order())

    def get_player_reward(self, position: int) -> Optional[int]:
        return self.players[position].reward if 0 <= position < self.N else None


def bot_hold_or_draw(game: Game, position: int) -> tuple[Action, int]:
    """Simple bot for non-dealer players: hold on 17+, else draw up to 3."""
    p = game.players[position]
    val = best_hand_value(p.hand)
    if val >= 17:
        return Action.HOLD, 0
    cards_left = 5 - len(p.hand)
    if cards_left == 0:
        return Action.HOLD, 0
    return Action.DRAW, min(3, cards_left)


def dealer_bot_action(game: Game) -> tuple[Action, int]:
    """Dealer bot: reveal if 17+, else draw to 17+ then reveal."""
    dealer = game.players[0]
    val = best_hand_value(dealer.hand)
    if val >= 17:
        return Action.REVEAL, 0
    cards_left = 5 - len(dealer.hand)
    if cards_left == 0:
        return Action.REVEAL, 0
    return Action.DRAW, min(3, cards_left)
