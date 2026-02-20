# Ban Luck

## Game Rules
- 52 Standard Deck of Playing Cards, no jokers 
- There is 1 dealer and N-1 players. 
- Players are seated in a circle, lets call the position k, relative to the dealer. k=0 being the dealer, k=1 being 1 player being the first player being dealt. 
- Each player is given 2 cards, the cards are dealt in a 2 round circular manner. So player 1 gets 1 card, player 2.... then finally the dealer, then one more round, player 1 gets the 2nd card, etc... 
- An Ace can be either counted as 1 or 11. 
- Player k has 2 possible actions, hold OR to draw up to 3 more cards (5 cards total in hand). 
- Player k=1 is the first to decide the action, followed by k=2, .... up to the dealer (k=0) 
- Player k instantly wins if the cards that were initially dealt are an ace and a 10 (Including Q, K, J), the reward = 2 
- Player k instantly wins if they draw up to 5 cards. If the sum of the cards = 21, they get reward = 3, else if it is below 21, reward = 2, else reward = -2 
- When it is the dealer's turn, aside from holding/drawing, they can also decide to reveal players cards. Let X = dealer's hand value, and Y = player k's hand value. If X < Y, player gets reward = 2 if Y = 21 else reward = 1. Else if X == Y, reward = 0, Else, reward = -2 if Y = 21 else reward = -1.

