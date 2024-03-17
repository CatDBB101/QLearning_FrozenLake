import gym
import pprint

display = False

# TODO: environment setup
if display:
    env = gym.make("FrozenLake-v1", render_mode="human")
else:
    env = gym.make("FrozenLake-v1")
env.action_space.seed(42)
state, info = env.reset(seed=42)

# TODO: agent setup
q_table = [[.0 for _2 in range(4)] for _1 in range(16)]
pprint.pprint(q_table)
print()
"""
state = 16
action = 4
"""

learning_rate = 0.8
discount_factor = 0.95

t = 0
training_round = 10_000
for _ in range(training_round):
    if t < training_round/4:
        action = env.action_space.sample()
    else:
        action = q_table[state].index(max(q_table[state]))

    new_state, reward, terminated, truncated, info = env.step(action)

    delta_q = learning_rate * (reward + discount_factor * max(q_table[new_state]) - q_table[state][action])
    q_table[state][action] += delta_q

    state = new_state

    print(f"t : {t+1}/{training_round}")
    print(f"action : {action}")
    print(f"state : {state}")
    print(f"new_state : {new_state}")
    print(f"delta_q : {delta_q}")

    if terminated or truncated:
        state, info = env.reset()
        print("terminated")
        print()
    t += 1

    print()

env.close()

print()
print("Finished train")
input("...")
print()

env = gym.make("FrozenLake-v1", render_mode="human")
env.action_space.seed(42)
state, info = env.reset(seed=42)
t = 0
while True:
    action = q_table[state].index(max(q_table[state]))
    new_state, reward, terminated, truncated, info = env.step(action)

    print(f"t : {t+1}")
    print(f"action : {action}")
    print(f"state : {state}")
    print(f"new_state : {new_state}")
    print(f"reward : {reward}")

    state = new_state
    if terminated or truncated:
        state = env.reset()[0]

        # env = gym.make("FrozenLake-v1", render_mode="human")
        # env.action_space.seed(42)
        # state, info = env.reset(seed=42)

        print("terminated")
        print()
    t += 1

    print()

env.close()