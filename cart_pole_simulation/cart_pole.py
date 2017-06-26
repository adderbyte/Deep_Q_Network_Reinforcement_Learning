import gym
env = gym.make('CartPole-v0')
#env.reset()
for _ in range(20):
     observation = env.reset()
     for t in range(200):
         env.render()
         action=env.action_space.sample() # take a random action
         observation, reward, done, info = env.step(action)
         if done:
             continue;
    
