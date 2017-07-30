import gym
env = gym.make('CartPole-v0')
for i_episode in range(30):
    observation = env.reset()
    for t in range(400):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break