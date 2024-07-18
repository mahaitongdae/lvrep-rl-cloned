from repr_control.envs.custom_env import CustomEnv

def test_custom_env():
    from repr_control.scripts.define_problem import dynamics, rewards, initial_distribution, state_range, action_range, sigma
    env = CustomEnv(dynamics, rewards, initial_distribution, state_range, action_range, sigma)

    print(env.reset())
    for i in range(10):
        print(env.step(env.action_space.sample()))

if __name__ == '__main__':
    test_custom_env()