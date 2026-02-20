"""
Reinforcement learning agent for the Player (not dealer) using Monte Carlo control.

Episode: 1) Cards dealt  2) Agent acts  3) Other players act  4) Dealer acts  5) Reward assigned.
The agent chooses one action (hold or draw 1–3 cards) to maximize expected reward.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

from game import (
    Game,
    Action,
    best_hand_value,
    is_natural_blackjack,
    _min_total,
    bot_hold_or_draw,
    dealer_bot_action,
)
from game import Card  # noqa: F401 - for type hints


# --- State representation (at decision time: 2 cards) ---

def has_usable_ace(cards: list) -> bool:
    """True if hand has an ace counted as 11 in the best total."""
    if not cards:
        return False
    total_min = _min_total(cards)
    best = best_hand_value(cards)
    return best != total_min and total_min + 10 <= 21


def state_from_hand(cards: list[Card]) -> tuple[int, int]:
    """
    Encode current hand as (hand_value, usable_ace).
    hand_value in [4, 21], usable_ace in {0, 1}.
    Used only when agent has exactly 2 cards (decision time).
    """
    value = best_hand_value(cards)
    usable = 1 if has_usable_ace(cards) else 0
    return (value, usable)


# --- Actions: 0 = hold, 1 = draw 1, 2 = draw 2, 3 = draw 3 ---

NUM_ACTIONS = 4  # hold, draw 1, draw 2, draw 3


def action_to_hold_or_draw(action: int) -> tuple[Action, int]:
    """Map action index to (Action.HOLD or Action.DRAW, num_cards)."""
    if action == 0:
        return Action.HOLD, 0
    return Action.DRAW, action  # 1, 2, or 3


def get_legal_actions(num_cards: int) -> list[int]:
    """Legal action indices when agent has num_cards (2–5). At 2 cards: all 4; at 5: only hold."""
    if num_cards >= 5:
        return [0]
    max_draw = min(3, 5 - num_cards)
    return [0] + list(range(1, max_draw + 1))


# --- Episode runner ---

def run_episode(
    n_players: int,
    agent_position: int,
    get_agent_action: Callable[[tuple[int, int]], int],
    *,
    seed: int | None = None,
) -> tuple[tuple[int, int], int, int]:
    """
    Run one episode: deal, agent acts when it's their turn, others and dealer use bots.
    Returns (state, action, reward) for the agent. If agent had natural blackjack, action is -1.
    """
    if seed is not None:
        random.seed(seed)
    game = Game(n_players=n_players)
    game.deal()

    agent_state: tuple[int, int] | None = None
    agent_action: int = -1

    while not game.all_turns_done():
        current = game.current_turn
        p = game.players[current]

        if p.done:
            game.advance_turn()
            continue

        if current == agent_position:
            # Agent's turn
            if is_natural_blackjack(p.hand):
                # No choice; reward already set to 2
                game.advance_turn()
                continue
            state = state_from_hand(p.hand)
            legal = get_legal_actions(len(p.hand))
            action = get_agent_action(state)
            if action not in legal:
                action = legal[0]
            agent_state = state
            agent_action = action
            act, num = action_to_hold_or_draw(action)
            if act == Action.HOLD:
                game.apply_hold(agent_position)
            else:
                game.apply_draw(agent_position, num)
            game.advance_turn()
            continue

        if current == 0:
            # Dealer
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

        # Other player (bot)
        act, num = bot_hold_or_draw(game, current)
        if act == Action.HOLD:
            game.apply_hold(current)
        else:
            game.apply_draw(current, num)
        game.advance_turn()

    if not game.revealed:
        game.dealer_reveal()

    reward = game.get_player_reward(agent_position)
    if reward is None:
        reward = 0
    if agent_state is None:
        # Natural blackjack: no (s,a) to record; we could use (21, 1) and action -1
        agent_state = (21, 1)
    return (agent_state, agent_action, reward)


# --- Monte Carlo control ---

def mc_control(
    n_players: int = 2,
    agent_position: int = 1,
    num_episodes: int = 500_000,
    epsilon: float = 0.1,
    *,
    seed: int | None = None,
) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, int], int]]:
    """
    Monte Carlo control (first-visit) with epsilon-greedy policy.
    Returns (Q, policy) where Q[state] = [Q(s,0), Q(s,1), Q(s,2), Q(s,3)] and policy[state] = best action.
    """
    if seed is not None:
        random.seed(seed)

    # Q[s][a] = average return for (s,a). We store sum and count for incremental update.
    Q_sum: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0] * NUM_ACTIONS)
    Q_count: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0] * NUM_ACTIONS)

    def get_action(state: tuple[int, int]) -> int:
        legal = get_legal_actions(2)
        if random.random() < epsilon:
            return random.choice(legal)
        qs = Q_sum[state]
        best_a = max(legal, key=lambda a: qs[a] if Q_count[state][a] > 0 else float("-inf"))
        return best_a

    for ep in range(num_episodes):
        state, action, reward = run_episode(n_players, agent_position, get_action)
        if action < 0:
            continue  # natural blackjack; no (s,a) to update
        # First-visit: this is the only (s,a) in the episode, so return = reward
        G = reward
        Q_sum[state][action] += G
        Q_count[state][action] += 1

    # Convert to Q values (means) and derive greedy policy
    Q: dict[tuple[int, int], list[float]] = {}
    policy: dict[tuple[int, int], int] = {}
    for state in Q_sum:
        q_list = [
            Q_sum[state][a] / Q_count[state][a] if Q_count[state][a] > 0 else 0.0
            for a in range(NUM_ACTIONS)
        ]
        Q[state] = q_list
        legal = get_legal_actions(2)
        policy[state] = max(legal, key=lambda a: q_list[a])

    return Q, policy


# --- Trained agent policy (use after mc_control) ---

def make_agent_policy(
    policy: dict[tuple[int, int], int],
    epsilon: float = 0.0,
) -> Callable[[tuple[int, int]], int]:
    """Return a callable that chooses action from learned policy (with optional exploration)."""

    def get_action(state: tuple[int, int]) -> int:
        legal = get_legal_actions(2)
        if epsilon > 0 and random.random() < epsilon:
            return random.choice(legal)
        return policy.get(state, 0)

    return get_action


# --- Save / load policy (for CLI use) ---

def policy_to_dict(policy: dict[tuple[int, int], int]) -> dict[str, int]:
    """Serialize policy to JSON-friendly dict: keys like '12_0' for (12, 0)."""
    return {f"{s[0]}_{s[1]}": a for s, a in policy.items()}


def policy_from_dict(data: dict[str, int]) -> dict[tuple[int, int], int]:
    """Deserialize policy from dict."""
    policy: dict[tuple[int, int], int] = {}
    for k, a in data.items():
        parts = k.split("_")
        if len(parts) == 2:
            try:
                state = (int(parts[0]), int(parts[1]))
                policy[state] = a
            except ValueError:
                pass
    return policy
