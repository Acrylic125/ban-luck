#!/usr/bin/env python3
"""
Benchmark: trained player vs trained dealer (alternating policies).

Loads agent_policy_alternating_{n}.json and agent_dealer_policy_alternating_{n}.json,
runs games with PolicyBasedPlayer vs PolicyBasedDealer, reports mean rewards per seat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from game import Game, make_deck
from deck import DeckCuttingStrategy, SwooshShuffleStrategy
from players import PolicyBasedPlayer
from dealer import PolicyBasedDealer
from agent import policy_from_dict
from simulation import run_game

NUM_RUNS_DEFAULT = 500
N_PLAYERS_DEFAULT = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark alternating player vs alternating dealer.")
    p.add_argument("-n", "--n-players", type=int, default=N_PLAYERS_DEFAULT, metavar="N")
    p.add_argument("-r", "--runs", type=int, default=NUM_RUNS_DEFAULT, metavar="N")
    args = p.parse_args()
    if args.n_players < 1 or args.n_players > 20:
        p.error("--n-players must be between 1 and 20")
    if args.runs < 1:
        p.error("--runs must be positive")
    return args


def main() -> None:
    args = parse_args()
    n_players = args.n_players
    num_runs = args.runs
    base = Path(__file__).resolve().parent
    player_path = base / f"agent_policy_alternating_{n_players}.json"
    dealer_path = base / f"agent_dealer_policy_alternating_{n_players}.json"

    if not player_path.exists():
        raise SystemExit(f"Player policy not found: {player_path}. Run train_dealer_and_player.py -n {n_players} first.")
    if not dealer_path.exists():
        raise SystemExit(f"Dealer policy not found: {dealer_path}. Run train_dealer_and_player.py -n {n_players} first.")

    with open(player_path) as f:
        player_policy = policy_from_dict(json.load(f))
    with open(dealer_path) as f:
        dealer_policy = policy_from_dict(json.load(f))

    player = PolicyBasedPlayer(player_policy, epsilon=0.0)
    dealer = PolicyBasedDealer(dealer_policy, epsilon=0.0)

    game = Game(n_players=n_players)
    deck = make_deck()
    SwooshShuffleStrategy().shuffle(deck, is_first=True)
    subsequent_shuffle = DeckCuttingStrategy()

    rewards_runs: list[list[float]] = []
    for _ in range(num_runs):
        rewards_runs.append(run_game(game, player, dealer, deck=deck))
        game.soft_reset()
        subsequent_shuffle.shuffle(deck, is_first=False)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    n_seats = game.N
    by_seat = list(zip(*rewards_runs))
    mean_by_seat = [mean(list(by_seat[k])) for k in range(n_seats)]
    dealer_rewards = [r[0] for r in rewards_runs]
    player_rewards = [sum(r[1:]) / len(r[1:]) for r in rewards_runs]

    mean_dealer = mean(dealer_rewards)
    mean_player = mean(player_rewards)

    print(f"Benchmark (alternating): {num_runs} games, {n_players} players")
    print("  PolicyBasedPlayer (alternating) vs PolicyBasedDealer (alternating)\n")
    print(f"  Mean dealer reward:  {mean_dealer:.4f}")
    print(f"  Mean player reward:  {mean_player:.4f}")
    print("\n  Mean reward per seat:")
    for k in range(n_seats):
        label = "Dealer" if k == 0 else f"Player {k}"
        print(f"    {label}: {mean_by_seat[k]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bin_edges = [x * 0.5 for x in range(-10, 12)]

    ax1 = axes[0]
    ax1.hist(dealer_rewards, bins=bin_edges, edgecolor="black", alpha=0.8, color="C0", label="Dealer")
    ax1.axvline(mean_dealer, color="red", linestyle="--", label=f"Mean = {mean_dealer:.3f}")
    ax1.set_xlabel("Dealer reward")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Dealer reward (n={num_runs})")
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(player_rewards, bins=bin_edges, edgecolor="black", alpha=0.8, color="orange", label="Players (avg)")
    ax2.axvline(mean_player, color="red", linestyle="--", label=f"Mean = {mean_player:.3f}")
    ax2.set_xlabel("Mean reward per player")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Player reward (n={num_runs})")
    ax2.legend()

    plt.tight_layout()
    out_path = base / f"benchmark_alternating_results_{n_players}p.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlots saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
