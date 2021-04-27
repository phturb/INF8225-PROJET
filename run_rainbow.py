import gym
import argparse
import sys
from rainbow.rainbow import Rainbow
from keras.callbacks import TensorBoard, ModelCheckpoint
# from ddqn import DDQN


def main():
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, dest="env_name", default="cart-pole",
                        choices=["mountain-car", "pong", "brick-breaker", "cart-pole", "pacman", "space-invaders", "pendulum"])
    parser.add_argument('-t', '--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.add_argument('-m', '--model-name', type=str,
                        dest="model_name", default="dqn")
    parser.add_argument('-mt', '--max-trials', type=int, dest="max_trials", default=500)
    parser.add_argument('-ns', '--n-step', type=int, dest="n_step", default=1)
    parser.add_argument('-dd','--dd-enable', dest="dd_enabled",  action='store_true')
    parser.add_argument('-du','--dueling-enable', dest="dueling_enabled", action='store_true')
    parser.add_argument('-ny','--noisy-net-enabled', dest="noisy_net_enabled", action='store_true')
    parser.add_argument('-nyt','--noisy-net-theta', type=float, dest="noisy_net_theta", default=0.5)
    parser.add_argument('-pm','--prioritized-memory-enabled',dest="prioritized_memory_enabled", action='store_true')
    parser.add_argument('-di','--categorical-enabled', dest="categorical_enabled", action='store_true')
    parser.add_argument('-wa','--warmup', dest="warmup", type=int, default=100)
    parser.add_argument('-r', '--render', dest='render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.add_argument('--no-tensorboard', dest='w_tensorboard', action='store_false')
    parser.set_defaults(train=True)
    parser.set_defaults(dd_enabled=False)
    parser.set_defaults(dueling_enabled=False)
    parser.set_defaults(noisy_net_enabled=False)
    parser.set_defaults(prioritized_memory_enabled=False)
    parser.set_defaults(categorical_enabled=False)
    parser.set_defaults(w_tensorboard=True)
    args = parser.parse_args()
    print(args)
    env_choices = {
        "mountain-car": "MountainCar-v0",
        "cart-pole": "CartPole-v1",
        "pong": "Pong-v0",
        "brick-breaker": "Breakout-v0",
        "pacman": "MsPacman-v0",
        "space-invaders": "SpaceInvaders-v0",
        "pendulum" : "Acrobot-v1"
    }
    is_atari = args.env_name == "pong" or args.env_name == "brick-breaker" or args.env_name =="pacman" or args.env_name == "space-invaders"

    model_name = args.model_name
    model_ext = ""
    if args.dd_enabled:
        model_ext += "_DD"
    if args.dueling_enabled:
        model_ext += "_dueling"
    if args.noisy_net_enabled:
        model_ext += "_noisy"
    if args.prioritized_memory_enabled:
        model_ext += "_prioritized"
    if args.categorical_enabled:
        model_ext += "_dist"
    if args.n_step > 1:
        model_ext += "_multi"
    
    env_name = env_choices[args.env_name]
    model_name += model_ext + "_" + env_name

    env = gym.make(env_name)

    print(f"Starting training for agent\n\tnamed :\t{model_name}\n\tgym :\t{env_name}")

    agent = Rainbow(env,
            model_name=model_name,
            dd_enabled=args.dd_enabled,
            dueling_enabled=args.dueling_enabled,
            noisy_net_enabled=args.noisy_net_enabled,
            prioritized_memory_enabled=args.prioritized_memory_enabled, 
            categorical_enabled=args.categorical_enabled,
            is_atari=is_atari)

    if args.w_tensorboard:
        callbacks = [TensorBoard(log_dir=f"./logs/rainbow/{model_name}", histogram_freq=1, update_freq='batch')]
    else:
        callbacks = []

    callbacks.append(ModelCheckpoint(filepath=f"./{model_name}_weights.h5"))
    # n_step > 1 activate multistep
    
    if args.train:
        history = agent.train(render=args.render, max_trials=args.max_trials, warmup=args.warmup, n_step=args.n_step, callbacks=callbacks, model_update_delay=100)
        agent.save_models()
    else:
        agent.load_models()
    agent.test(render=args.render)


if __name__ == "__main__":
    main()
