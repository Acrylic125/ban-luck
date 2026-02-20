#!/usr/bin/env python3
"""CLI for the card game. You play as one of the players (never the dealer)."""

import json
from pathlib import Path

from game import (
    Game,
    Action,
    best_hand_value,
    is_bust,
    is_natural_blackjack,
    bot_hold_or_draw,
    dealer_bot_action as _dealer_bot_action,
)

# Optional: trained RL agent (load when user chooses "trained agent")
AGENT_POLICY_PATH = Path(__file__).resolve().parent / "agent_policy.json"


def print_hand(label: str, cards: list, hidden: bool = False) -> None:
    if hidden:
        print(f"  {label}: ?? ??  (value ?)")
    else:
        parts = " ".join(str(c) for c in cards)
        val = best_hand_value(cards)
        bust = " (BUST)" if is_bust(cards) else ""
        print(f"  {label}: {parts}  (value {val}{bust})")


def run_cli() -> None:
    print("=== Card Game (Dealer + N-1 Players) ===\n")
    use_agent = False
    agent_get_action = None
    if AGENT_POLICY_PATH.exists():
        choice = input("Play as (1) Human (2) Trained agent? [1]: ").strip() or "1"
        if choice == "2":
            try:
                from agent import (
                    policy_from_dict,
                    make_agent_policy,
                )
                with open(AGENT_POLICY_PATH) as f:
                    policy = policy_from_dict(json.load(f))
                agent_get_action = make_agent_policy(policy, epsilon=0.0)
                use_agent = True
                print("Using trained agent policy.")
            except Exception as e:
                print(f"Could not load agent: {e}. Playing as human.")
    else:
        pass  # no policy file; human only

    while True:
        try:
            n_players = int(input("Number of players (1 to 5): ").strip())
            if 1 <= n_players <= 5:
                break
        except ValueError:
            pass
        print("Enter a number from 1 to 5.")

    # Human (or agent) position k (1 = first to act, never 0)
    while True:
        try:
            human_k = int(input(f"Your position (1 to {n_players}, 1 = first to act): ").strip())
            if 1 <= human_k <= n_players:
                break
        except ValueError:
            pass
        print(f"Enter a number from 1 to {n_players}.")

    game = Game(n_players=n_players)
    game.deal()

    print("\n--- Dealt hands (others hidden until reveal) ---")
    for k in game.turn_order():
        p = game.players[k]
        name = "Dealer" if k == 0 else f"Player {k}"
        if k == human_k:
            print_hand(name, p.hand, hidden=False)
        else:
            print_hand(name, p.hand, hidden=True)

    # Turn order: 1, 2, ..., n_players, then 0 (dealer)
    while not game.all_turns_done():
        current = game.current_turn
        p = game.players[current]
        if p.done:
            game.advance_turn()
            continue

        name = "Dealer" if current == 0 else f"Player {current}"
        print(f"\n--- {name}'s turn ---")

        if current == 0:
            # Dealer acts once: reveal, or hold/draw then we reveal
            act, num = _dealer_bot_action(game)
            if act == Action.REVEAL:
                print("Dealer reveals.")
                game.dealer_reveal()
                break
            if act == Action.DRAW:
                game.apply_draw(0, num)
                print(f"Dealer draws {num} card(s). Hand: {' '.join(str(c) for c in game.players[0].hand)}")
            else:
                game.apply_hold(0)
                print("Dealer holds.")
            game.dealer_reveal()
            print("Dealer reveals all.")
            break

        if current == human_k:
            # Your turn (human or trained agent)
            print_hand("Your hand", p.hand, hidden=False)
            if is_natural_blackjack(p.hand):
                print("Natural blackjack! Reward = 2. (Already applied.)")
                game.advance_turn()
                continue
            max_draw = min(3, 5 - len(p.hand))
            if use_agent and agent_get_action is not None:
                from agent import (
                    state_from_hand,
                    get_legal_actions,
                    action_to_hold_or_draw,
                )
                state = state_from_hand(p.hand)
                legal = get_legal_actions(len(p.hand))
                action = agent_get_action(state)
                if action not in legal:
                    action = legal[0]
                act, num = action_to_hold_or_draw(action)
                if act == Action.HOLD:
                    game.apply_hold(current)
                    print("Agent holds.")
                else:
                    game.apply_draw(current, num)
                    print(f"Agent draws {num} card(s). New hand: {' '.join(str(c) for c in p.hand)}")
                    print(f"Value: {best_hand_value(p.hand)}. Reward: {p.reward}.")
                game.advance_turn()
                continue
            while True:
                choice = input("Hold or Draw? (h/d): ").strip().lower()
                if choice in ("h", "hold"):
                    game.apply_hold(current)
                    print("You hold.")
                    break
                if choice in ("d", "draw"):
                    if max_draw == 0:
                        print("You cannot draw more cards.")
                        continue
                    prompt = f"How many cards to draw (1 to {max_draw})? "
                    try:
                        num = int(input(prompt).strip())
                        if 1 <= num <= max_draw:
                            game.apply_draw(current, num)
                            print(f"You draw {num} card(s). New hand: {' '.join(str(c) for c in p.hand)}")
                            val = best_hand_value(p.hand)
                            print(f"Value: {val}. Reward: {p.reward}.")
                            break
                    except ValueError:
                        pass
                    print(f"Enter a number from 1 to {max_draw}.")
                    continue
                print("Type 'h' to hold or 'd' to draw.")
            game.advance_turn()
            continue

        # Other player (bot)
        if is_natural_blackjack(p.hand):
            print(f"Player {current} has natural blackjack. Reward = 2.")
            game.advance_turn()
            continue
        act, num = bot_hold_or_draw(game, current)
        if act == Action.HOLD:
            game.apply_hold(current)
            print(f"Player {current} holds.")
        else:
            game.apply_draw(current, num)
            print(f"Player {current} draws {num} card(s). Reward = {game.players[current].reward}.")
        game.advance_turn()

    # If dealer didn't reveal yet (e.g. drew to 17+ then we need to reveal)
    if not game.revealed:
        game.dealer_reveal()
        print("\nDealer reveals all.")

    # Final hands and rewards
    print("\n--- Final hands and rewards ---")
    dealer = game.players[0]
    print_hand("Dealer", dealer.hand, hidden=False)
    for k in range(1, game.N):
        p = game.players[k]
        name = "You" if k == human_k else f"Player {k}"
        print_hand(name, p.hand, hidden=False)
        r = p.reward
        print(f"    Reward: {r}")
    your_reward = game.get_player_reward(human_k)
    print(f"\nYour total reward this round: {your_reward}")

    again = input("\nPlay again? (y/n): ").strip().lower()
    if again in ("y", "yes"):
        run_cli()


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
