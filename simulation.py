"""
Simulation: run one or more games with a given player and dealer strategy.

Single source of truth for the game loop (players act, then dealer acts until reveal).
"""

from __future__ import annotations

from game import Action, Card, Game, is_natural_blackjack
from players import Player
from dealer import Dealer


def run_game(
    game: Game,
    player: Player,
    dealer: Dealer,
    deck: list[Card],
) -> list[float]:
    game.deal(deck)
    for _ in range(game.N):
        current = game.current_turn
        p = game.players[current]
        if current != 0:
            if is_natural_blackjack(p.hand):
                game.advance_turn()
                continue
            while True:
                p = game.players[current]
                if p.reward is not None:
                    break
                act, _ = player.choose_action(game, current)
                if act == Action.HOLD:
                    game.advance_turn()
                    break
                game.apply_draw(current, 1)
                if game.players[current].reward is not None:
                    game.advance_turn()
                    break
            continue
        if current == 0:
            while True:
                act = dealer.choose_action(game)
                if act == Action.REVEAL:
                    game.dealer_reveal_all()
                    break
                if act == Action.DRAW:
                    game.apply_draw(0, 1)
                else:
                    game.dealer_reveal_all()
                    break
    return [game.get_player_reward(k) or 0 for k in range(game.N)]
