#!/usr/bin/env python3
"""
Alternating training: train player with SimpleDealer, then alternate
training dealer (vs current player) and player (vs current dealer) until
convergence or 100 iterations. Save both policies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game import Card, Game, make_deck
from deck import DeckCuttingStrategy, SwooshShuffleStrategy
from players import Player, SimplePlayer, PolicyBasedPlayer
from dealer import Dealer, PolicyBasedDealer, SimpleDealer
from agent import mc_control, policy_to_dict
from train_dealer_agent import mc_control_dealer
from simulation import run_game

NUM_EVAL_GAMES = 500
MAX_ALTERNATING_ITERS = 100
PLAYER_CONVERGENCE_THRESHOLD = 0.02  # stop when player improvement < this
DEALER_CONVERGENCE_FACTOR = 0.02     # dealer threshold = this * n_players
DEFAULT_N_PLAYERS = 2
DEFAULT_EPISODES_PER_ROUND = 500_000
DEFAULT_EPSILON = 0.1


def evaluate(
    game: Game,
    player: Player,
    dealer: Dealer,
    deck: list[Card],
    num_games: int,
) -> tuple[float, float]:
    """Run num_games with given player and dealer. Return (mean_player_reward, mean_dealer_reward)."""
    shuffle = DeckCuttingStrategy()
    player_rewards: list[float] = []
    dealer_rewards: list[float] = []
    for _ in range(num_games):
        rewards = run_game(game, player, dealer, deck=deck)
        dealer_rewards.append(rewards[0])
        player_rewards.append(sum(rewards[1:]) / len(rewards[1:]) if len(rewards) > 1 else 0.0)
        game.soft_reset()
        shuffle.shuffle(deck, is_first=False)
    mean_player = sum(player_rewards) / len(player_rewards) if player_rewards else 0.0
    mean_dealer = sum(dealer_rewards) / len(dealer_rewards) if dealer_rewards else 0.0
    return mean_player, mean_dealer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alternating train player and dealer agents.")
    p.add_argument("-n", "--n-players", type=int, default=DEFAULT_N_PLAYERS, metavar="N")
    p.add_argument("-e", "--episodes", type=int, default=DEFAULT_EPISODES_PER_ROUND, metavar="N", help="Episodes per training round")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, metavar="F")
    p.add_argument("--eval-games", type=int, default=NUM_EVAL_GAMES, metavar="N", help="Games per evaluation")
    args = p.parse_args()
    if args.n_players < 1 or args.n_players > 20:
        p.error("--n-players must be between 1 and 20")
    if args.episodes < 1 or args.eval_games < 1:
        p.error("--episodes and --eval-games must be positive")
    if not 0 <= args.epsilon <= 1:
        p.error("--epsilon must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    n_players = args.n_players
    num_episodes = args.episodes
    epsilon = args.epsilon
    num_eval_games = args.eval_games

    game = Game(n_players=n_players)
    deck = make_deck()
    SwooshShuffleStrategy().shuffle(deck, is_first=True)

    player_policy_path = Path(__file__).resolve().parent / f"agent_policy_alternating_{n_players}.json"
    dealer_policy_path = Path(__file__).resolve().parent / f"agent_dealer_policy_alternating_{n_players}.json"

    # Round 1: train player with SimpleDealer
    print("=== Round 1: Train player (vs SimpleDealer) ===")
    _, player_policy = mc_control(
        n_players=n_players,
        agent_position=1,
        num_episodes=num_episodes,
        epsilon=epsilon,
        dealer=SimpleDealer(),
    )
    policy_player = PolicyBasedPlayer(player_policy, epsilon=0.0)
    mean_player, mean_dealer = evaluate(game, policy_player, SimpleDealer(), deck, num_eval_games)
    prev_player_mean = mean_player
    prev_dealer_mean = None
    dealer_convergence_threshold = DEALER_CONVERGENCE_FACTOR * n_players
    print(f"  Eval: mean player reward = {mean_player:.4f}, mean dealer reward = {mean_dealer:.4f}")
    print(f"  Convergence: player Δ < {PLAYER_CONVERGENCE_THRESHOLD}, dealer Δ < {dealer_convergence_threshold:.4f} (0.02 × {n_players})")

    for iteration in range(MAX_ALTERNATING_ITERS):
        dealer_converged = False
        player_converged = False

        # Train dealer with current player policy
        print(f"\n=== Alternating iteration {iteration + 1}: Train dealer (vs current player) ===")
        policy_player_current = PolicyBasedPlayer(player_policy, epsilon=0.0)
        _, dealer_policy = mc_control_dealer(
            n_players=n_players,
            num_episodes=num_episodes,
            epsilon=epsilon,
            player=policy_player_current,
        )
        policy_dealer = PolicyBasedDealer(dealer_policy, epsilon=0.0)
        mean_player, mean_dealer = evaluate(game, policy_player_current, policy_dealer, deck, num_eval_games)
        print(f"  Eval: mean player reward = {mean_player:.4f}, mean dealer reward = {mean_dealer:.4f}")
        if prev_dealer_mean is not None and abs(mean_dealer - prev_dealer_mean) < dealer_convergence_threshold:
            dealer_converged = True
            print(f"  Dealer converged: |Δ| = {abs(mean_dealer - prev_dealer_mean):.4f} < {dealer_convergence_threshold:.4f}")
        prev_dealer_mean = mean_dealer

        # Train player with current dealer policy
        print(f"\n=== Alternating iteration {iteration + 1}: Train player (vs current dealer) ===")
        policy_dealer_current = PolicyBasedDealer(dealer_policy, epsilon=0.0)
        _, player_policy = mc_control(
            n_players=n_players,
            agent_position=1,
            num_episodes=num_episodes,
            epsilon=epsilon,
            dealer=policy_dealer_current,
        )
        policy_player = PolicyBasedPlayer(player_policy, epsilon=0.0)
        mean_player, mean_dealer = evaluate(game, policy_player, policy_dealer_current, deck, num_eval_games)
        print(f"  Eval: mean player reward = {mean_player:.4f}, mean dealer reward = {mean_dealer:.4f}")
        if abs(mean_player - prev_player_mean) < PLAYER_CONVERGENCE_THRESHOLD:
            player_converged = True
            print(f"  Player converged: |Δ| = {abs(mean_player - prev_player_mean):.4f} < {PLAYER_CONVERGENCE_THRESHOLD}")
        prev_player_mean = mean_player

        if dealer_converged and player_converged:
            print("\n  Both dealer and player converged. Stopping.")
            break

    # Save both policies
    player_policy_sorted = dict(sorted(player_policy.items(), key=lambda x: (len(x[0]), x[0])))
    dealer_policy_sorted = dict(sorted(dealer_policy.items(), key=lambda x: (len(x[0]), x[0])))
    with open(player_policy_path, "w") as f:
        json.dump(policy_to_dict(player_policy_sorted), f, indent=2)
    with open(dealer_policy_path, "w") as f:
        json.dump(policy_to_dict(dealer_policy_sorted), f, indent=2)
    print(f"\nSaved player policy to {player_policy_path}")
    print(f"Saved dealer policy to {dealer_policy_path}")
    print("Done.")


if __name__ == "__main__":
    main()
