# Imitation_Learning
Imitation Learning Algorithms (and IRL) based on RL library [mushroom_rl](https://github.com/MushroomRL/mushroom-rl).

- IL Implemented Algorithms: GAIL, VAIL (with PPO or TRPO policy update versions available);
- IRL Implemented Algorithms: SCIRL
    - Examples for both [MountaiCar](examples/irl/scirl_mountain_car.py) and [Pendulum-v0](examples/irl/scirl_pendulum.py) environments;
    - Feature Expectation estimator with Monte Carlo rollouts;
    - Estimation with Least-Squares Temporal difference is not debbuged (TODO);
- Option for easy states/actions selection for descriminator;
- There are multiple examples available;
- Algorithms are still being benchmarked in multiple environments;
