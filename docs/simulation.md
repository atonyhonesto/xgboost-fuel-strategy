# The simulation (`simulation.py`)

XGBoost's ranker gives every car a score. A score isn't a chance, though: "Blaney
scores 1.2 and Hamlin scores 0.9" doesn't tell you how often Blaney wins. This file
turns scores into chances by playing the race out thousands of times.

## One simulated race

1. **Who doesn't finish?** Each car "rolls the dice" on its DNF chance from the DNF
   model. Cars that fail go to the back, in random order.
2. **Everyone else:** each car's score gets a dose of random luck, and the cars are
   ordered by score plus luck.

Do that 10,000 times and count: how often each car won, finished top 3, top 5 or top
10, failed to finish, and where it finished on average.

## Why the luck has a particular shape

The random luck follows a **Gumbel distribution**, and that's not an arbitrary
choice. XGBoost's `rank:pairwise` objective trains the scores so that

> the chance A finishes ahead of B = sigmoid(score A − score B)

Adding Gumbel noise to each score gives exactly that head-to-head chance. So the
simulation is faithful to what the ranker actually learned. It's the classic
"Plackett–Luce" model of rankings.

## The temperature

One number scales the luck: the **temperature**. At 1.0 the ranker is taken at face
value. Higher means more luck and lower means the scores count for more.

Rather than guess, `fit_temperature()` tries values from 0.5 to 3.0 and keeps the
one that best explains the real finishing orders of races the model **hadn't seen**,
using the Plackett–Luce likelihood (`plackett_luce_loglik()`). For the all-tracks model
it comes out at 1.0, which says the ranker's scores were already about the right size.

## The other helpers

- **`summarise(finish, out)`** turns a batch of simulations into the chances table.
- **`logloss(p, y)`** scores probabilities, keeping them between 0.5% and 99.5% so one
  confident miss can't dominate.
- **`calibration(bt)`** groups cars by predicted top-10 chance and compares each group
  with how many really finished top 10.

Both the app's model (`race_model.py`) and the earlier Talladega study
(`field_model.py`) use this file.
