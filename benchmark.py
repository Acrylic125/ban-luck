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

from game import (
    Game,
    Action,
    is_natural_blackjack,
    dealer_bot_action,
)
from players import Player, SimplePlayer, PolicyBasedPlayer
from agent import policy_from_dict


NUM_RUNS_DEFAULT = 100
N_PLAYERS_DEFAULT = 2


def run_game_all_same_strategy(
    n_players: int,
    player: Player,
    seed: int,
) -> float:
    """
    Run one game with the given seed. All non-dealer players use the same strategy.
    Returns the mean reward per player (over positions 1..n_players).
    """
    random.seed(seed)
    game = Game(n_players=n_players)
    game.deal()

    while not game.all_turns_done():
        current = game.current_turn
        p = game.players[current]

        if p.done:
            game.advance_turn()
            continue

        if current == 0:
            act, num = dealer_bot_action(game)
            if act == Action.REVEAL:
                game.dealer_reveal()
                break
            if act == Action.DRAW:
                game.apply_draw(0, num)
            else:
                game.apply_hold(0)
            game.dealer_reveal()
            break

        # All players use the same strategy
        if is_natural_blackjack(p.hand):
            game.advance_turn()
            continue
        act, num = player.choose_action(game, current)
        if act == Action.HOLD:
            game.apply_hold(current)
        else:
            game.apply_draw(current, num)
        game.advance_turn()

    if not game.revealed:
        game.dealer_reveal()

    rewards = [
        game.get_player_reward(k) or 0
        for k in range(1, game.N)
    ]
    return sum(rewards) / len(rewards) if rewards else 0.0


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
    if args.n_players < 1 or args.n_players > 5:
        p.error("--n-players must be between 1 and 5")
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

    rewards_policy: list[float] = []
    rewards_simple: list[float] = []

    for run in range(num_runs):
        seed = run
        rewards_policy.append(
            run_game_all_same_strategy(n_players, policy_player, seed)
        )
        rewards_simple.append(
            run_game_all_same_strategy(n_players, simple_player, seed)
        )

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

    # Plots
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / f"benchmark_results_{n_players}p.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlots saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
