import gym
import numpy as np

# env = gym.make("CartPole-v0")
env = gym.make("CartPole-v1")
# env.monitor.start("/tmp/cartpole-3")


def run_episode(env, w, test):
    obs = env.reset()
    total = 0
    for i in range(env.spec.timestep_limit):
        if test:
            env.render()
        x = obs
        p = 1.0 / (1.0 + np.exp(-np.dot(w, x)))
        if test:
            action = 0.5 < p
        else:
            action =  np.random.random() < p
        obs, reward, done, info = env.step(action)
        total += reward
        if done:
            break
    return total


(n,) = env.observation_space.shape
m = n
mu = np.zeros(m)
sigma = np.identity(m) * 10


for epoch in range(10):
    print "Epoch", epoch

    ws = np.random.multivariate_normal(mu, sigma, 1000)
    scores = [run_episode(env, w, True) for w in ws]
    best = np.argsort(scores)[-10:]
    mu = np.mean(ws[best], axis=0)
    sigma = np.cov(ws[best].T)
    print "Score", np.mean(scores)

score = run_episode(env, mu, True)
print "Weights", mu
print "Test Score", score
