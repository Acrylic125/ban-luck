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
    make_deck,
)
from game import Card  # noqa: F401 - for type hints
from deck import DeckCuttingStrategy, SwooshShuffleStrategy
from dealer import Dealer, SimpleDealer
from state import (
    NUM_ACTIONS,
    action_to_hold_or_draw,
    get_legal_actions,
    state_from_hand,
)


def _bot_hold_or_draw(game: Game, position: int) -> tuple[Action, int]:
    """Simple bot for non-agent players: hold on 17+, else draw 1."""
    p = game.players[position]
    if best_hand_value(p.hand) >= 17:
        return Action.HOLD, 0
    if len(p.hand) >= 5:
        return Action.HOLD, 0
    return Action.DRAW, 1


def run_episode(
    agent_position: int,
    get_agent_action: Callable[[str], int],
    deck: list[Card],
    game: Game,
    dealer: Dealer | None = None,
) -> list[tuple[str, int], int]:
    assert len(deck) == 52
    game.deal(deck)
    if dealer is None:
        dealer = SimpleDealer()

    agent_state_acitons: list[tuple[str, int]] = []

    for _ in range(game.N):
        current = game.current_turn
        p = game.players[current]

        # Player
        if current != 0:
            if is_natural_blackjack(p.hand):
                game.advance_turn()
                continue
            # Only the agent acts when it's their turn; other players use the bot
            if current == agent_position:
                while True:
                    p = game.players[agent_position]
                    if p.reward is not None:
                        break
                    state = state_from_hand(p.hand)
                    legal = get_legal_actions(len(p.hand))
                    action = get_agent_action(state)
                    if action not in legal:
                        action = legal[0]
                    agent_state_acitons.append((state, action))
                    act, _num = action_to_hold_or_draw(action)
                    if act == Action.HOLD:
                        game.advance_turn()
                        break
                    game.apply_draw(agent_position, 1)
                    if game.players[agent_position].reward is not None:
                        game.advance_turn()
                        break
            else:
                # Non-agent player: use bot (hold on 17+, else draw 1)
                while True:
                    p = game.players[current]
                    if p.reward is not None:
                        break
                    act, _num = _bot_hold_or_draw(game, current)
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
                act = dealer.choose_action(game)
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
    dealer: Dealer | None = None,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """
    Monte Carlo control (first-visit) with epsilon-greedy policy.
    Returns (Q, policy) where Q[state] = [Q(s,0), Q(s,1), Q(s,2), Q(s,3)] and policy[state] = best action.
    """
    if dealer is None:
        dealer = SimpleDealer()
    game = Game(n_players=n_players)
    deck = make_deck()

    # Q[s][a] = average return for (s,a). We store sum and count for incremental update.
    Q_sum: dict[str, list[float]] = defaultdict(lambda: [0.0] * NUM_ACTIONS)
    Q_count: dict[str, list[int]] = defaultdict(lambda: [0] * NUM_ACTIONS)

    def get_action(state: str) -> int:
        num_cards = state.count(",") + 1
        if state not in Q_sum:
            Q_sum[state] = [0.0] * NUM_ACTIONS
        if state not in Q_count:
            Q_count[state] = [0] * NUM_ACTIONS
        legal = get_legal_actions(num_cards)
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
        state_actions_rewards = run_episode(agent_position, get_action, deck, game, dealer=dealer)
        for state, action, reward in state_actions_rewards:
            Q_sum[state][action] += reward
            Q_count[state][action] += 1
        game.soft_reset()
        DeckCuttingStrategy().shuffle(deck, is_first=False)

    # Convert to Q values (means) and derive greedy policy
    Q: dict[str, list[float]] = {}
    policy: dict[str, int] = {}
    for state in Q_sum:
        q_list = [
            Q_sum[state][a] / Q_count[state][a] if Q_count[state][a] > 0 else 0.0
            for a in range(NUM_ACTIONS)
        ]
        Q[state] = q_list
        num_cards = state.count(",") + 1
        legal = get_legal_actions(num_cards)
        policy[state] = max(legal, key=lambda a: q_list[a])

    return Q, policy


def make_agent_policy(
    policy: dict[str, int],
    epsilon: float = 0.0,
) -> Callable[[str], int]:
    """Return a callable that chooses action from learned policy (with optional exploration)."""

    def get_action(state: str) -> int:
        num_cards = state.count(",") + 1
        legal = get_legal_actions(num_cards)
        if epsilon > 0 and random.random() < epsilon:
            return random.choice(legal)
        return policy.get(state, 0)

    return get_action

def policy_to_dict(policy: dict[str, int]) -> dict[str, int]:
    """Serialize policy to JSON-friendly dict. State keys are full hand strings (e.g. '1_11,10' or '2,3,5')."""
    return dict(policy)

def policy_from_dict(data: dict[str, int]) -> dict[str, int]:
    """Deserialize policy from dict (string state -> action)."""
    return dict(data)
