# Imitation_Learning
Imitation Learning Algorithms (and IRL) based on RL library [mushroom_rl](https://github.com/MushroomRL/mushroom-rl).

- IL Implemented Algorithms: [GAIL](https://arxiv.org/pdf/1606.03476.pdf), [VAIL](https://arxiv.org/pdf/1810.00821.pdf)
    - [PPO](https://arxiv.org/pdf/1707.06347.pdf) or [TRPO](https://arxiv.org/pdf/1502.05477.pdf) policy update versions available;
    - Option for easy states/actions selection for descriminator;
    
- IRL Implemented Algorithms: SCIRL
    - Examples for both [MountainCar](examples/irl/scirl_mountain_car.py) and [Pendulum-v0](examples/irl/scirl_pendulum.py) environments;
    - Feature Expectation estimator with Monte Carlo rollouts;
    - Estimation with Least-Squares Temporal difference is not debbuged (TODO);

- There are [multiple examples](https://github.com/ManuelPalermo/mushroom_rl_imitation/tree/master/examples) available;
- [Expert trajectories](https://github.com/ManuelPalermo/mushroom_rl_imitation/tree/master/examples/expert_data) for some evironments;
- Algorithms are still being benchmarked in multiple environments;
