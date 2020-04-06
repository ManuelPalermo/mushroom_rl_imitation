# Imitation_Learning
Imitation Learning Algorithms (and IRL) based on RL library [mushroom_rl](https://github.com/MushroomRL/mushroom-rl).

- IL Implemented Algorithms: GAIL, VAIL 
    - PPO or TRPO policy update versions available;
    - Option for easy states/actions selection for descriminator;
    
- IRL Implemented Algorithms: SCIRL
    - Examples for both [MountainCar](examples/irl/scirl_mountain_car.py) and [Pendulum-v0](examples/irl/scirl_pendulum.py) environments;
    - Feature Expectation estimator with Monte Carlo rollouts;
    - Estimation with Least-Squares Temporal difference is not debbuged (TODO);

- There are multiple examples available;
- [Expert trajectories](https://github.com/ManuelPalermo/mushroom_rl_imitation/tree/master/examples/expert_data) for some evironments.
- Algorithms are still being benchmarked in multiple environments;
