"""
Reinforcement learning agent for the Player (not dealer) using Monte Carlo control.

Episode: 
1. Cards dealt
2. Agent acts
3. Other players act
4. Dealer acts
5. Reward assigned.
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
    dealer_bot_action,
    make_deck,
)
from game import Card  # noqa: F401 - for type hints
from deck import DeckCuttingStrategy, SwooshShuffleStrategy


# --- State representation (at decision time: 2 cards) ---

def has_usable_ace(cards: list) -> bool:
    """True if hand has an ace counted as 11 in the best total."""
    if not cards:
        return False
    total_min = _min_total(cards)
    best = best_hand_value(cards)
    return best != total_min and total_min + 10 <= 21

def serialize_card_state(card: Card) -> str:
    value = card.blackjack_value()
    if isinstance(value, tuple):
        return f"{value[0]}_{value[1]}"
    return str(value)
    # return f"{card.rank}{card.suit}"

def state_from_hand(cards: list[Card]) -> str:
    v = [serialize_card_state(c) for c in cards]
    v.sort()
    return ",".join(v)

# Actions: 
NUM_ACTIONS = 4  # hold 0, draw 1, draw 2, draw 3

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


def run_episode(
    agent_position: int,
    get_agent_action: Callable[[str], int],
    deck: list[Card],
    game: Game,
) -> list[tuple[int, int], int, int]:
    assert len(deck) == 52
    game.deal(deck)

    agent_state_acitons: list[tuple[str, int]] = []

    # while not game.all_turns_done():
    for _ in range(game.N):
        current = game.current_turn
        p = game.players[current]

        # Player
        if current != 0:
            # Agent's turn: decide each time (hold or draw one); repeat until hold or done
            if is_natural_blackjack(p.hand):
                # Done. 
                game.advance_turn()
                continue
            while True:
                p = game.players[agent_position]
                if p.reward is not None:
                    break
                state = state_from_hand(p.hand)
                legal = get_legal_actions(len(p.hand))
                action = get_agent_action(state)
                if action not in legal:
                    action = legal[0]
                if current == agent_position:
                    agent_state_acitons.append((state, action))
                act, _num = action_to_hold_or_draw(action)
                if act == Action.HOLD:
                    game.advance_turn()
                    break
                # Draw exactly one card; agent will be asked again unless hand is done
                game.apply_draw(agent_position, 1)
                if game.players[agent_position].reward is not None:
                    game.advance_turn()
                    break
            continue

        # Dealer
        if current == 0:
            while True:
                act = dealer_bot_action(game)
                if act == Action.REVEAL:
                    game.dealer_reveal_all()
                    break
                if act == Action.DRAW:
                    game.apply_draw(0, 1)
                else:
                    game.dealer_reveal_all()
                    break

    reward = game.get_player_reward(agent_position)
    if reward is None:
        raise ValueError("Agent reward is None")
    return list(map(lambda x: (x[0], x[1], reward), agent_state_acitons))

def mc_control(
    n_players: int = 2,
    agent_position: int = 1,
    num_episodes: int = 500_000,
    epsilon: float = 0.1,
) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, int], int]]:
    """
    Monte Carlo control (first-visit) with epsilon-greedy policy.
    Returns (Q, policy) where Q[state] = [Q(s,0), Q(s,1), Q(s,2), Q(s,3)] and policy[state] = best action.
    """
    game = Game(n_players=n_players)
    deck = make_deck()

    # Q[s][a] = average return for (s,a). We store sum and count for incremental update.
    Q_sum: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0] * NUM_ACTIONS)
    Q_count: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0] * NUM_ACTIONS)

    def get_action(state: str) -> int:
        legal = get_legal_actions(2)
        # Policy action = 1 - e - e / (actions_count)
        # Other actions = e / (actions_count)
        true_epsilon = epsilon - (epsilon / len(legal))
        if random.random() < true_epsilon:
            return random.choice(legal)
        qs = Q_sum[state]
        q_counts = Q_count[state]
        # Argmax over Q values
        best_a = max(legal, key=lambda a: qs[a] / q_counts[a] if q_counts[a] > 0 else float("-inf"))
        return best_a

    SwooshShuffleStrategy().shuffle(deck, is_first=True)
    p10 = (num_episodes // 10)
    for ep in range(num_episodes):
        if ep % p10 == 0:
            print(f"Episode {ep} of {num_episodes} ({ep/num_episodes*100:.1f}%)")
        state_actions_rewards = run_episode(agent_position, get_action, deck, game)
        for state, action, reward in state_actions_rewards:
            Q_sum[state][action] += reward
            Q_count[state][action] += 1
        game.soft_reset()
        DeckCuttingStrategy().shuffle(deck, is_first=False)

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
