#!/usr/bin/env python3
"""
Train the Dealer agent using Monte Carlo control, then save the policy.

Players use a fixed strategy (SimplePlayer). Dealer learns when to REVEAL vs DRAW.
uv run train_dealer_agent.py --n-players 3 --episodes 500000 --epsilon 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from game import Action, Game, make_deck, is_natural_blackjack
from dealer import (
    SimpleDealer,
    DEALER_NUM_ACTIONS,
    dealer_action_to_enum,
    get_dealer_legal_actions,
)
from players import Player, SimplePlayer
from agent import state_from_hand, policy_to_dict
from deck import DeckCuttingStrategy, WashShuffleStrategy

DEFAULT_EPISODES = 500_000
DEFAULT_EPSILON = 0.1
DEFAULT_N_PLAYERS = 2


def run_episode_dealer(
    game: Game,
    deck: list,
    get_dealer_action: Callable[[str], int],
    player: Player,
) -> list[tuple[str, int, int]]:
    """Run one episode: deal, all players act (same strategy), dealer acts until REVEAL. Return (state, action, reward) for dealer."""
    game.deal(deck)
    dealer_state_actions: list[tuple[str, int]] = []

    for _ in range(game.N):
        current = game.current_turn
        p = game.players[current]

        if current != 0:
            # Players: use SimplePlayer
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

        # Dealer
        if current == 0:
            while True:
                state = state_from_hand(game.players[0].hand)
                legal = get_dealer_legal_actions(game)
                action = get_dealer_action(state)
                if action not in legal:
                    action = legal[0]
                dealer_state_actions.append((state, action))
                act = dealer_action_to_enum(action)
                if act == Action.REVEAL:
                    game.dealer_reveal_all()
                    break
                game.apply_draw(0, 1)
            break

    reward = game.get_player_reward(0)
    if reward is None:
        reward = 0
    return [(s, a, reward) for s, a in dealer_state_actions]


def mc_control_dealer(
    n_players: int = 2,
    num_episodes: int = 500_000,
    epsilon: float = 0.1,
    player: Player | None = None,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Monte Carlo control for dealer. Returns (Q, policy) with state str -> [Q(s,0), Q(s,1)] and best action."""
    if player is None:
        player = SimplePlayer()
    game = Game(n_players=n_players)
    deck = make_deck()

    Q_sum: dict[str, list[float]] = defaultdict(lambda: [0.0] * DEALER_NUM_ACTIONS)
    Q_count: dict[str, list[int]] = defaultdict(lambda: [0] * DEALER_NUM_ACTIONS)

    def get_action(state: str) -> int:
        # Legal actions depend on game state; we get them from a dummy game state. For learning we use [0, 1] when unknown.
        if state not in Q_sum:
            Q_sum[state] = [0.0] * DEALER_NUM_ACTIONS
            Q_count[state] = [0] * DEALER_NUM_ACTIONS
        # Use [0, 1] for exploration; run_episode_dealer will clamp to legal
        legal = [0, 1]
        if random.random() < epsilon:
            return random.choice(legal)
        qs = Q_sum[state]
        qc = Q_count[state]
        best_a = max(legal, key=lambda a: qs[a] / qc[a] if qc[a] > 0 else float("-inf"))
        return best_a

    WashShuffleStrategy().shuffle(deck, is_first=True)
    p10 = max(1, num_episodes // 10)
    for ep in range(num_episodes):
        if ep % p10 == 0:
            print(f"Episode {ep} of {num_episodes} ({ep/num_episodes*100:.1f}%)")
        for state, action, reward in run_episode_dealer(game, deck, get_action, player):
            Q_sum[state][action] += reward
            Q_count[state][action] += 1
        game.soft_reset()
        DeckCuttingStrategy().shuffle(deck, is_first=False)

    Q: dict[str, list[float]] = {}
    policy: dict[str, int] = {}
    for state in Q_sum:
        q_list = [
            Q_sum[state][a] / Q_count[state][a] if Q_count[state][a] > 0 else 0.0
            for a in range(DEALER_NUM_ACTIONS)
        ]
        Q[state] = q_list
        # Greedy policy: when we don't have game context here, default legal to [0, 1]
        policy[state] = max(range(DEALER_NUM_ACTIONS), key=lambda a: q_list[a])

    return Q, policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Dealer agent (Monte Carlo control).")
    p.add_argument("-n", "--n-players", type=int, default=DEFAULT_N_PLAYERS, metavar="N", help=f"Number of players (default: {DEFAULT_N_PLAYERS})")
    p.add_argument("-e", "--episodes", type=int, default=DEFAULT_EPISODES, metavar="N", help=f"Number of training episodes (default: {DEFAULT_EPISODES})")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, metavar="F", help=f"Epsilon for epsilon-greedy (default: {DEFAULT_EPSILON})")
    args = p.parse_args()
    if args.n_players < 1 or args.n_players > 20:
        p.error("--n-players must be between 1 and 20")
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if not 0 <= args.epsilon <= 1:
        p.error("--epsilon must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    n_players = args.n_players
    num_episodes = args.episodes
    epsilon = args.epsilon
    policy_path = Path(__file__).resolve().parent / f"agent_dealer_policy_{n_players}.json"

    print("Training Dealer agent (Monte Carlo control)...")
    print(f"  n_players={n_players}, num_episodes={num_episodes}, epsilon={epsilon}")
    Q, policy = mc_control_dealer(n_players=n_players, num_episodes=num_episodes, epsilon=epsilon)

    policy = dict(sorted(policy.items(), key=lambda x: (len(x[0]), x[0])))
    data = policy_to_dict(policy)
    with open(policy_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Dealer policy saved to {policy_path}")
    print("Done.")


if __name__ == "__main__":
    main()
