import argparse
import gym
from twenty48stoch import Twenty48stoch
from twenty48determ import Twenty48determ

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

''' 
    RENDER MODES
    'human': Render using powers of 2
    'ai': Render using values that the gym environment observes

    RUN MODES
    'manual': User inputs actions
    'auto': Randomly sample input actions
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Twenty48stoch-v0')
    parser.add_argument('--render_mode', type=str, default='human')
    parser.add_argument('--run_mode', type=str, default='manual')
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # Custom entry point for Twenty48 environments, comment out if using different environment
    env_name = args.task.split("-")[0]
    gym.envs.register(
        id=args.task,
        entry_point=f'{"".join(env_name).lower()}:{env_name}',
        max_episode_steps=1000,
        reward_threshold=float('inf'),
    )
    env = gym.make(args.task, render_mode=args.render_mode)

    if args.run_mode == 'manual':
        manual(env)
    elif args.run_mode == 'auto':
        auto(env)


def manual(env):
    r = 0
    print("Enter w, a, s, d to move up, left, down, right respectively and \
enter q to quit")
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
        else:
            print("Invalid action")
            action == 5
            continue
        if action != 5:
            obs, reward, done, info, *other_values = env.step(action)
            print("Step reward:", reward)
            r += reward
            print("reward:", r)
            env.render()
            if done:
                env.reset()
    env.close()


def auto(env):
    r = 0
    print("reward:", r)
    env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info, *other_values = env.step(action)
        print("Step reward:", reward)
        r += reward
        print("reward:", r)
        env.render()
        if done:
            r = 0
            env.reset()
    env.close()


if __name__ == '__main__':
    main(get_args())
