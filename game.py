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

# Standard deck order: 1 (Ace), 2, 3, ... 10, K, Q, J per suit
STANDARD_RANK_ORDER = [
    Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
    Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.KING, Rank.QUEEN, Rank.JACK,
]


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
    # We evaluate all possibilities.
    interpretations: List[List[int]] = []

    for c in cards:
        v = c.blackjack_value()
        if isinstance(v, tuple):
            og = interpretations.copy()
            new_interpretations = []
            for i in v:
                for interpretation in og:
                    new_interpretations.append([*interpretation, i])
            interpretations = new_interpretations
        else:
            for interpretation in interpretations:
                interpretation.append(v)
    
    assert interpretations
    best = sum(interpretations[0])
    for interpretation in interpretations:
        s = sum(interpretation)
        if s == 21:
            return 21
        if s > best and s <= 21:
            best = s
    return best

def _min_total(cards: List[Card]) -> int:
    """Minimum total (all aces as 1)."""
    t = 0
    for c in cards:
        v = c.blackjack_value()
        t += 1 if isinstance(v, tuple) else v
    return t


def is_natural_blackjack(cards: List[Card]) -> bool:
    """True if exactly two cards: Ace and a 10-value (10, J, Q, K) OR Ace and an Ace."""
    if len(cards) != 2:
        return False
    a, b = cards[0], cards[1]
    if a.is_ace() and b.is_ace():
        return True
    return (a.is_ace() and b.is_ten_value()) or (b.is_ace() and a.is_ten_value())

def is_bust(cards: List[Card]) -> bool:
    """True if minimum total (all aces as 1) > 21."""
    return _min_total(cards) > 21


def raw_total(cards: List[Card]) -> int:
    """Total using best interpretation (Ace 1 or 11). Same as best_hand_value but can exceed 21."""
    return best_hand_value(cards) if not is_bust(cards) else _min_total(cards)


def make_deck() -> List[Card]:
    """Return a new deck in standard order: 1, 2, ... 10, K, Q, J per suit."""
    return [Card(s, r) for s in Suit for r in STANDARD_RANK_ORDER]


class Action(str, Enum):
    HOLD = "hold"
    DRAW = "draw"
    REVEAL = "reveal"  # dealer only


@dataclass
class PlayerState:
    """State for one seat (dealer or player)."""
    # k: 0 = dealer, 1..N-1 = players
    position: int  
    hand: List[Card] = field(default_factory=list)
    # Not None indicates the user already revealed their hand.
    reward: Optional[int] = None  

    def is_dealer(self) -> bool:
        return self.position == 0

    def hand_value(self) -> int:
        return best_hand_value(self.hand)

    def can_draw_more(self) -> bool:
        return len(self.hand) < 5 and self.reward is None

class Game:
    """Single round: one dealer, positions 1..n_players (N = n_players + 1). Caller passes the deck to deal()."""

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

    def deal(self, deck: List[Card]) -> None:
        """Deal from the provided deck (caller is responsible for shuffling). Deck is used in place."""
        self.deck = deck
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

    # def apply_hold(self, position: int) -> None:
    #     p = self.players[position]

    def apply_draw(self, position: int, num_cards: int) -> None:
        p = self.players[position]
        assert p.can_draw_more() and 1 <= num_cards <= min(3, 5 - len(p.hand))
        for _ in range(num_cards):
            if self.deck:
                p.hand.append(self.deck.pop())
            else:
                raise ValueError("No cards left in deck")
        total = best_hand_value(p.hand)
        if len(p.hand) >= 5:
            if is_bust(p.hand):
                p.reward = -2
            elif total == 21:
                p.reward = 3
            else:
                p.reward = 2

    def dealer_reveal_position(self, position: int) -> None:
        dealer = self.players[0]
        assert position > 0
        p = self.players[position]
        # Already revealed
        if p.reward is not None:
            print(f"Player {position} already revealed with reward {p.reward}")
            return
        # Ensure dealer has a reward set.
        if dealer.reward is None:
            dealer.reward = 0
        # Check if both dealer and player are bust.
        dealer_value = dealer.hand_value()
        player_value = p.hand_value()
        # Tie
        if (dealer_value > 21 and player_value > 21) or (dealer_value == player_value):
            p.reward = 0
            return
        # Player wins
        if dealer_value < player_value:
            reward = 2 if player_value == 21 else 1
            # Check if player hand is >= 5.
            if len(p.hand) >= 5:
                reward += 1
            p.reward = reward
            dealer.reward -= reward
            return
        # Dealer wins
        if dealer_value > player_value:
            reward = 2 if dealer_value == 21 else 1
            # Check if dealer hand is >= 5.
            if len(dealer.hand) >= 5:
                reward += 1
            p.reward = -reward
            dealer.reward += reward
            return
    
    def dealer_reveal_all(self) -> None:
        for k in range(1, self.N):
            self.dealer_reveal_position(k)

    def all_turns_done(self) -> bool:
        return all(self.players[k].done for k in self.turn_order())

    def get_player_reward(self, position: int) -> Optional[int]:
        return self.players[position].reward if 0 <= position < self.N else None

    def soft_reset(self) -> None:
        """
        Put all players' hands back onto the top of the deck (in random player order),
        then reset each player state and current_turn. Use before the next round when
        reusing the same game instance.
        """
        player_order = list(range(self.N))
        random.shuffle(player_order)
        for k in player_order:
            p = self.players[k]
            for c in p.hand:
                self.deck.append(c)
            p.hand.clear()
            p.reward = None
            p.done = False
        self.current_turn = 1
        self.revealed = False

    def deal_round(self) -> None:
        """
        Deal 2 cards to each player from the current deck (no shuffle).
        Assumes deck is full and player states were cleared (e.g. after soft_reset).
        """
        order = self._seat_order_for_deal()
        for _ in range(2):
            for k in order:
                self.players[k].hand.append(self.deck.pop())
        self.current_turn = 1
        self.revealed = False
        for k in range(1, self.N):
            p = self.players[k]
            if is_natural_blackjack(p.hand):
                p.reward = 2
                p.done = True


# def bot_hold_or_draw(game: Game, position: int) -> tuple[Action, int]:
#     """Simple bot for non-dealer players: hold on 17+, else draw up to 3."""
#     p = game.players[position]
#     val = best_hand_value(p.hand)
#     if val >= 17:
#         return Action.HOLD, 0
#     cards_left = 5 - len(p.hand)
#     if cards_left == 0:
#         return Action.HOLD, 0
#     return Action.DRAW, min(3, cards_left)


def dealer_bot_action(game: Game) -> Action:
    dealer = game.players[0]
    val = best_hand_value(dealer.hand)
    if val >= 17:
        return Action.REVEAL
    cards_left = 5 - len(dealer.hand)
    if cards_left == 0:
        return Action.REVEAL
    return Action.DRAW
