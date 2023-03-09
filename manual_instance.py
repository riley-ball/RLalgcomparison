import gym
from twenty48v1 import Twenty48

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

env = gym.make('Twenty48-v1', render_mode='human')

# env.reset()
# for i in range(1000):
#     action = env.action_space.sample()
#     obs, reward, done, info, *other_values = env.step(action)
#     env.render()
#     if done:
#         env.reset()
# env.close()

r = 0
print("reward:", r)
env.reset()
while True:
    action = input("Enter action: ")
    if action == "w":
        action = 0
    elif action == "a":
        action = 3
    elif action == "s":
        action = 1
    elif action == "d":
        action = 2
    elif action == "q":
        break

    obs, reward, done, info, *other_values = env.step(action)
    r += reward
    print("reward:", r)
    env.render()
    if done:
        env.reset()
env.close()

    