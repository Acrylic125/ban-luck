#!/usr/bin/env python3
"""
Benchmark PolicyBasedPlayer vs SimplePlayer over multiple games.

All non-dealer players use the same strategy in each game.
Each game is dealt the same for both strategies (same seed per run).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
from game import Card, Game, make_deck
from deck import DeckCuttingStrategy, DeckShuffleStrategy, SwooshShuffleStrategy
from players import Player, SimplePlayer, PolicyBasedPlayer
from dealer import SimpleDealer
from agent import policy_from_dict
from scipy import stats
from simulation import run_game

NUM_RUNS_DEFAULT = 100
N_PLAYERS_DEFAULT = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark PolicyBasedPlayer vs SimplePlayer (all players same strategy).",
    )
    p.add_argument(
        "-n",
        "--n-players",
        type=int,
        default=N_PLAYERS_DEFAULT,
        metavar="N",
        help=f"Number of players (default: {N_PLAYERS_DEFAULT})",
    )
    p.add_argument(
        "-r",
        "--runs",
        type=int,
        default=NUM_RUNS_DEFAULT,
        metavar="N",
        help=f"Number of games per strategy (default: {NUM_RUNS_DEFAULT})",
    )
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
    policy_path = Path(__file__).resolve().parent / f"agent_policy_{n_players}.json"

    if not policy_path.exists():
        raise SystemExit(
            f"Policy file not found: {policy_path}. Run train_agent.py -n {n_players} first."
        )
    with open(policy_path) as f:
        policy = policy_from_dict(json.load(f))

    policy_player = PolicyBasedPlayer(policy, epsilon=0.0)
    simple_player = SimplePlayer()

    # Each run yields per-player rewards; we keep lists of those lists
    rewards_policy_runs: list[list[float]] = []
    rewards_simple_runs: list[list[float]] = []
    game = Game(n_players=n_players)
    first_shuffle = SwooshShuffleStrategy()
    subsequent_shuffle = DeckCuttingStrategy()

    deck = make_deck()
    first_shuffle.shuffle(deck, is_first=True)
    dealer = SimpleDealer()
    for run in range(num_runs):
        # Same deck for both strategies (paired comparison)
        rewards_policy_runs.append(
            run_game(game, policy_player, dealer, deck)
        )
        game.soft_reset()
        subsequent_shuffle.shuffle(deck, is_first=False)
        rewards_simple_runs.append(
            run_game(game, simple_player, dealer, deck)
        )
        game.soft_reset()
        subsequent_shuffle.shuffle(deck, is_first=False)

    # Per-run mean over players only (exclude dealer) for stats and histograms
    rewards_policy = [sum(r[1:]) / len(r[1:]) for r in rewards_policy_runs] if rewards_policy_runs else []
    rewards_simple = [sum(r[1:]) / len(r[1:]) for r in rewards_simple_runs] if rewards_simple_runs else []

    # Stats
    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def std(xs: list[float], m: float | None = None) -> float:
        if len(xs) < 2:
            return 0.0
        m = m if m is not None else mean(xs)
        variance = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return variance ** 0.5

    mean_policy = mean(rewards_policy)
    std_policy = std(rewards_policy, mean_policy)
    mean_simple = mean(rewards_simple)
    std_simple = std(rewards_simple, mean_simple)

    print(f"Benchmark: {num_runs} games, {n_players} players (all same strategy per game)\n")
    print("PolicyBasedPlayer:")
    print(f"  Mean reward (per player): {mean_policy:.4f}")
    print(f"  Std dev:                   {std_policy:.4f}")
    print("\nSimplePlayer:")
    print(f"  Mean reward (per player): {mean_simple:.4f}")
    print(f"  Std dev:                  {std_simple:.4f}")

    # Significance (paired: same run for both strategies)
    mean_diff = mean_policy - mean_simple
    diffs = [p - s for p, s in zip(rewards_policy, rewards_simple)]
    n = len(diffs)
    if n >= 2:
        t_stat, p_value = stats.ttest_rel(rewards_policy, rewards_simple)
        std_diff = std(diffs, None)
        se_diff = std_diff / (n ** 0.5) if n else 0.0
        # Approximate 95% CI for mean difference (t ~ 1.96 for large n)
        t_95 = 1.96 if n > 30 else {2: 4.30, 5: 2.57, 10: 2.23, 20: 2.09, 30: 2.04}.get(n, 2.0)
        ci_lo = mean_diff - t_95 * se_diff
        ci_hi = mean_diff + t_95 * se_diff
        print("\n--- Significance (paired comparison) ---")
        print(f"  Mean difference (Policy − Simple): {mean_diff:.4f}")
        print(f"  95% CI for difference:             [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

    # Per-seat mean reward across all runs (index 0 = dealer, 1..n_players = players)
    n_seats = game.N  # dealer + n_players
    by_seat_policy = list(zip(*rewards_policy_runs))
    by_seat_simple = list(zip(*rewards_simple_runs))
    mean_by_seat_policy = [mean(list(by_seat_policy[k])) for k in range(n_seats)]
    mean_by_seat_simple = [mean(list(by_seat_simple[k])) for k in range(n_seats)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bin_edges = [x * 0.5 for x in range(-5, 8)]  # -2.5 to 3.5

    # Chart 1: PolicyBasedPlayer reward distribution
    ax1 = axes[0]
    ax1.hist(rewards_policy, bins=bin_edges, edgecolor="black", alpha=0.8)
    ax1.axvline(mean_policy, color="red", linestyle="--", label=f"Mean = {mean_policy:.3f}")
    ax1.set_xlabel("Mean reward per player")
    ax1.set_ylabel("Count")
    ax1.set_title(f"PolicyBasedPlayer (n={num_runs}, {n_players} players)\nμ = {mean_policy:.3f}, σ = {std_policy:.3f}")
    ax1.legend()

    # Chart 2: SimplePlayer reward distribution
    ax2 = axes[1]
    ax2.hist(rewards_simple, bins=bin_edges, edgecolor="black", alpha=0.8, color="orange")
    ax2.axvline(mean_simple, color="red", linestyle="--", label=f"Mean = {mean_simple:.3f}")
    ax2.set_xlabel("Mean reward per player")
    ax2.set_ylabel("Count")
    ax2.set_title(f"SimplePlayer (n={num_runs}, {n_players} players)\nμ = {mean_simple:.3f}, σ = {std_simple:.3f}")
    ax2.legend()

    # Chart 3: Mean reward of each seat (dealer + players) across all runs
    ax3 = axes[2]
    x = list(range(n_seats))
    width = 0.35
    ax3.bar([i - width / 2 for i in x], mean_by_seat_policy, width, label="PolicyBased", color="C0", edgecolor="black")
    ax3.bar([i + width / 2 for i in x], mean_by_seat_simple, width, label="Simple", color="orange", edgecolor="black")
    ax3.set_xlabel("Seat")
    ax3.set_ylabel("Mean reward across runs")
    ax3.set_title(f"Mean reward per seat (n={num_runs} runs)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Dealer"] + [str(i) for i in range(1, n_players + 1)])
    ax3.legend()
    ax3.axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / f"benchmark_results_{n_players}p.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlots saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
