#!/usr/bin/env python3
"""
Train the Player agent using Monte Carlo control, then save the policy.

uv run train_agent.py --n-players 3 --episodes 500000 --epsilon 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from agent import mc_control, mc_control_dqn, policy_to_dict, QNetwork
from deck import DeckCuttingStrategy, WashShuffleStrategy

DEFAULT_EPISODES = 500_000
DEFAULT_EPSILON = 0.1
DEFAULT_N_PLAYERS = 2
DEFAULT_HISTORY_LENGTH = 0
ACTION_NAMES = ["hold", "draw 1", "draw 2", "draw 3"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Player agent (Monte Carlo control).",
    )
    p.add_argument(
        "-n",
        "--n-players",
        type=int,
        default=DEFAULT_N_PLAYERS,
        metavar="N",
        help=f"Number of players (default: {DEFAULT_N_PLAYERS})",
    )
    p.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        metavar="N",
        help=f"Number of training episodes (default: {DEFAULT_EPISODES})",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        metavar="F",
        help=f"Epsilon for epsilon-greedy exploration (default: {DEFAULT_EPSILON})",
    )
    p.add_argument(
        "--history-length",
        type=int,
        default=DEFAULT_HISTORY_LENGTH,
        metavar="M",
        help=(
            "Number of past rounds to include in the state history (M in the spec). "
            "Each step adds up to 5*N cards per round."
        ),
    )
    p.add_argument(
        "--tabular",
        action="store_true",
        help="Use the original tabular Monte Carlo control instead of the DQN.",
    )
    args = p.parse_args()
    if args.n_players < 1 or args.n_players > 20:
        p.error("--n-players must be between 1 and 20")
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if not 0 <= args.epsilon <= 1:
        p.error("--epsilon must be between 0 and 1")
    if args.history_length < 0:
        p.error("--history-length must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    n_players = args.n_players
    num_episodes = args.episodes
    epsilon = args.epsilon
    history_length = args.history_length

    if args.tabular:
        policy_path = Path(__file__).resolve().parent / f"agent_policy_{n_players}.json"

        print("Training Player agent with tabular Monte Carlo control...")
        print(
            f"  n_players={n_players}, agent_position=1, num_episodes={num_episodes}, "
            f"epsilon={epsilon}"
        )
        Q, policy = mc_control(
            n_players=n_players,
            agent_position=1,
            num_episodes=num_episodes,
            epsilon=epsilon,
        )

        # Sort by state string for reproducible output (2-card, then 3-card, then 4-card hands)
        policy = dict(sorted(policy.items(), key=lambda x: (len(x[0]), x[0])))
        # Save policy
        data = policy_to_dict(policy)
        with open(policy_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Policy saved to {policy_path}")
        print("Done.")
        return

    # DQN training path
    model_path = Path(__file__).resolve().parent / f"agent_qnet_{n_players}.pt"
    meta_path = Path(__file__).resolve().parent / f"agent_qnet_{n_players}.meta.json"

    print("Training Player agent with DQN (neural Q-network)...")
    print(
        f"  n_players={n_players}, agent_position=1, num_episodes={num_episodes}, "
        f"epsilon={epsilon}, history_length={history_length}"
    )

    q_net: QNetwork = mc_control_dqn(
        n_players=n_players,
        agent_position=1,
        num_episodes=num_episodes,
        epsilon=epsilon,
        history_length=history_length,
        first_shuffle_strategy=WashShuffleStrategy(),
        subsequent_shuffle_strategy=DeckCuttingStrategy(deck_interleaving_probability=0),
    )

    torch.save(q_net.state_dict(), model_path)
    meta = {
        "n_players": n_players,
        "history_length": history_length,
        "epsilon": epsilon,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Q-network saved to {model_path}")
    print(f"  Metadata saved to {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
