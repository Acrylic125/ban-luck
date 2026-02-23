#!/usr/bin/env python3
"""
Benchmark DQN-based player vs SimplePlayer over multiple games.

All non-dealer players use the same strategy in each game.
Each game is dealt the same for both strategies (same seed per run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy import stats

from agent import (
    QNetwork,
    encode_state_with_history,
)
from deck import DeckCuttingStrategy, WashShuffleStrategy
from game import Card, Game, make_deck
from players import Player, SimplePlayer
from dealer import SimpleDealer
from simulation import run_game
from state import NUM_ACTIONS, action_to_hold_or_draw, get_legal_actions


NUM_RUNS_DEFAULT = 100
N_PLAYERS_DEFAULT = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark DQN-based Player vs SimplePlayer (all players same strategy).",
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


class DqnPlayer(Player):
    """
    Player implementation that uses a trained QNetwork with the
    [hand] + M * [round history] state representation.
    """

    def __init__(
        self,
        q_net: QNetwork,
        history_length: int,
    ) -> None:
        self._net = q_net.eval()
        self._history_length = history_length

    def choose_action(self, game: Game, position: int) -> tuple[Action, int]:  # type: ignore[name-defined]
        # First 5 entries of the state vector are the current hand.
        state_vec = encode_state_with_history(
            game,
            agent_position=position,
            history_length=self._history_length,
        )
        num_cards = sum(1 for v in state_vec[:5] if v != 0)
        legal = get_legal_actions(num_cards)
        if not legal:
            return action_to_hold_or_draw(0)

        with torch.no_grad():
            s = torch.tensor(state_vec, dtype=torch.long).unsqueeze(0)
            q_values = self._net(s)[0]
            best_action = max(legal, key=lambda a: float(q_values[a].item()))
        return action_to_hold_or_draw(best_action)


def load_dqn_player(n_players: int) -> tuple[Player, int]:
    """
    Load a trained Q-network and construct a DqnPlayer.
    Returns (player, history_length).
    """
    base = Path(__file__).resolve().parent
    model_path = base / f"agent_qnet_{n_players}.pt"
    meta_path = base / f"agent_qnet_{n_players}.meta.json"

    if not model_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"DQN model or metadata not found for {n_players} players.\n"
            f"Expected:\n  {model_path}\n  {meta_path}\n"
            f"Run train_player.py without --tabular first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    history_length = int(meta.get("history_length", 0))
    n_players_meta = int(meta.get("n_players", n_players))
    if n_players_meta != n_players:
        raise SystemExit(
            f"Trained model is for n_players={n_players_meta}, "
            f"but benchmark requested n_players={n_players}."
        )

    n_seats = n_players + 1  # dealer + players
    state_len = 5 + history_length * 5 * n_seats
    num_card_tokens = 14  # must match training setup

    q_net = QNetwork(num_card_tokens=num_card_tokens, state_len=state_len, num_actions=NUM_ACTIONS)
    q_net.load_state_dict(torch.load(model_path, map_location="cpu"))

    player = DqnPlayer(q_net=q_net, history_length=history_length)
    return player, history_length


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float], m: float | None = None) -> float:
    if len(xs) < 2:
        return 0.0
    m = m if m is not None else mean(xs)
    variance = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return variance ** 0.5


def main() -> None:
    args = parse_args()
    n_players = args.n_players
    num_runs = args.runs

    dqn_player, history_length = load_dqn_player(n_players)
    simple_player = SimplePlayer()

    rewards_dqn_runs: list[list[float]] = []
    rewards_simple_runs: list[list[float]] = []

    game = Game(n_players=n_players)
    dealer = SimpleDealer()
    first_shuffle = WashShuffleStrategy()
    subsequent_shuffle = DeckCuttingStrategy(
        deck_interleaving_probability=0,
    )

    deck = make_deck()
    first_shuffle.shuffle(deck, is_first=True)

    for run in range(num_runs):
        # DQN strategy
        rewards_dqn_runs.append(
            run_game(game, dqn_player, dealer, deck)
        )

        game.soft_reset()
        subsequent_shuffle.shuffle(deck, is_first=False)

        # Simple strategy on the same deck (paired)
        rewards_simple_runs.append(
            run_game(game, simple_player, dealer, deck)
        )
        game.soft_reset()
        subsequent_shuffle.shuffle(deck, is_first=False)

    rewards_dqn = [sum(r[1:]) / len(r[1:]) for r in rewards_dqn_runs] if rewards_dqn_runs else []
    rewards_simple = [sum(r[1:]) / len(r[1:]) for r in rewards_simple_runs] if rewards_simple_runs else []

    mean_dqn = mean(rewards_dqn)
    std_dqn = std(rewards_dqn, mean_dqn)
    mean_simple = mean(rewards_simple)
    std_simple = std(rewards_simple, mean_simple)

    print(f"Benchmark: {num_runs} games, {n_players} players (all same strategy per game)\n")
    print("DQN Player:")
    print(f"  Mean reward (per player): {mean_dqn:.4f}")
    print(f"  Std dev:                   {std_dqn:.4f}")
    print("\nSimplePlayer:")
    print(f"  Mean reward (per player): {mean_simple:.4f}")
    print(f"  Std dev:                  {std_simple:.4f}")

    # Significance (paired: same run for both strategies)
    mean_diff = mean_dqn - mean_simple
    diffs = [p - s for p, s in zip(rewards_dqn, rewards_simple)]
    n = len(diffs)
    if n >= 2:
        t_stat, p_value = stats.ttest_rel(rewards_dqn, rewards_simple)
        std_diff = std(diffs, None)
        se_diff = std_diff / (n ** 0.5) if n else 0.0
        t_95 = 1.96 if n > 30 else {2: 4.30, 5: 2.57, 10: 2.23, 20: 2.09, 30: 2.04}.get(n, 2.0)
        ci_lo = mean_diff - t_95 * se_diff
        ci_hi = mean_diff + t_95 * se_diff
        print("\n--- Significance (paired comparison) ---")
        print(f"  Mean difference (DQN − Simple): {mean_diff:.4f}")
        print(f"  95% CI for difference:          [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

    # Per-seat mean reward across all runs (index 0 = dealer, 1..n_players = players)
    n_seats = game.N  # dealer + n_players
    by_seat_dqn = list(zip(*rewards_dqn_runs))
    by_seat_simple = list(zip(*rewards_simple_runs))
    mean_by_seat_dqn = [mean(list(by_seat_dqn[k])) for k in range(n_seats)]
    mean_by_seat_simple = [mean(list(by_seat_simple[k])) for k in range(n_seats)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bin_edges = [x * 0.5 for x in range(-5, 8)]  # -2.5 to 3.5

    # Chart 1: DQN reward distribution
    ax1 = axes[0]
    ax1.hist(rewards_dqn, bins=bin_edges, edgecolor="black", alpha=0.8)
    ax1.axvline(mean_dqn, color="red", linestyle="--", label=f"Mean = {mean_dqn:.3f}")
    ax1.set_xlabel("Mean reward per player")
    ax1.set_ylabel("Count")
    ax1.set_title(f"DQN Player (n={num_runs}, {n_players} players)\nμ = {mean_dqn:.3f}, σ = {std_dqn:.3f}")
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
    ax3.bar([i - width / 2 for i in x], mean_by_seat_dqn, width, label="DQN", color="C0", edgecolor="black")
    ax3.bar([i + width / 2 for i in x], mean_by_seat_simple, width, label="Simple", color="orange", edgecolor="black")
    ax3.set_xlabel("Seat")
    ax3.set_ylabel("Mean reward across runs")
    ax3.set_title(f"Mean reward per seat (n={num_runs} runs)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Dealer"] + [str(i) for i in range(1, n_players + 1)])
    ax3.legend()
    ax3.axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / f"benchmark_dqn_results_{n_players}p.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlots saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

