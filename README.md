# Re-act

This is a little idea I had for consistent exploration in RL. Instead of parameter noise or evolution, we can directly optimize for "consistent actions", by using SGD during the agent's lifetime to increase the probability that it performs the same action *A* given state *S*.

Preliminary results did not look promising. I was experimenting on a long maze environment with sparse rewards. The re-act agent had a higher probability of solving the maze when the inner LR was 0, indicating that re-act is not a good way to implement consistent exploration. Oh well :'(