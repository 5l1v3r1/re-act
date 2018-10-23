# Re-act

This is an idea I had for consistent exploration in RL. Instead of parameter noise or evolution, we can directly optimize for "consistent actions", by using SGD during the agent's lifetime to increase the probability that it performs the same action a given state S.