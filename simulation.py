"""
Simulation: run one or more games with a given player and dealer strategy.

Single source of truth for the game loop (players act, then dealer acts until reveal).
"""

from __future__ import annotations

from itertools import combinations

from game import Action, Card, Game, make_deck, is_natural_blackjack
from players import Player
from dealer import Dealer
from state import state_from_hand


def run_game(
    game: Game,
    player: Player | list[Player],
    dealer: Dealer,
    deck: list[Card],
) -> list[float]:
    # strategies[k] is the Player strategy for game.players[k + 1]
    if isinstance(player, list):
        strategies = player
        if len(strategies) < game.N - 1:
            raise ValueError(
                f"list of players has length {len(strategies)}, need at least {game.N - 1} for {game.N - 1} non-dealer seats"
            )
    else:
        strategies = [player] * (game.N - 1)

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
                act, _ = strategies[current - 1].choose_action(game, current)
                if act == Action.HOLD:
                    game.advance_turn()
                    break
                game.apply_draw(current, 1)
                if len(p.hand) >= 5:
                    game.dealer_reveal_position(current)
                    game.advance_turn()
                    break
                # if game.players[current].reward is not None:
                #     game.advance_turn()
                #     break
            continue
        if current == 0:
            while True:
                act = dealer.choose_action(game)
                if act == Action.REVEAL:
                    game.dealer_reveal_all()
                    break
                if act == Action.DRAW:
                    game.apply_draw(0, 1)
                    if len(game.players[0].hand) >= 5:
                        game.dealer_reveal_all()
                        break
                else:
                    game.dealer_reveal_all()
                    break
    return [game.get_player_reward(k) or 0 for k in range(game.N)]


def _action_to_index(act: Action, num_cards: int) -> int:
    if act == Action.HOLD:
        return 0
    return num_cards  # 1, 2, or 3


def trace_policy(player: Player, n_players: int = 1) -> dict[str, int]:
    deck = make_deck()
    result: dict[str, int] = {}
    seen: set[str] = set()
    position = 1  # non-dealer position whose policy we trace
    for k in range(2, 6):
        for hand in combinations(deck, k):
            state = state_from_hand(list(hand))
            if state in seen:
                continue
            seen.add(state)
            game = Game(n_players=n_players)
            game.players[position].hand = list(hand)
            game.current_turn = position
            act, num = player.choose_action(game, position)
            result[state] = _action_to_index(act, num)
    return result
