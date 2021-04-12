import gym
import argparse
from projet.models import create_atari_model, batch_states_process_atari, observation_process_atari,\
    reward_process_atari, create_mountain_cart_model, batch_states_process_mountain_cart, observation_process_mountain_cart, reward_process_mountain_cart
from projet.dqn import DQN
from projet.ddqn import DDQN
from projet.a3c import A3C
from projet.distDQN import DistDQN
from projet.nDQN import NDQN
# from ddqn import DDQN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, dest="env_name", default="mountain-car",
                        choices=["moutain-car", "pong", "brick-breaker"])
    parser.add_argument('-t', '--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.add_argument('-m', '--model-name', type=str,
                        dest="model_name", default="default")
    parser.add_argument('-a', '--agent', type=str, choices=[
                        "dqn", "ddqn", "a3c", "distdqn", "ndqn"], default="dqn", dest="agent")
    parser.add_argument('-r', '--render', dest='render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(train=True)
    parser.set_defaults(render=False)
    args = parser.parse_args()
    env_choices = {
        "mountain-car": "MountainCar-v0",
        "pong": "Pong-v0",
        "brick-breaker": "Pong-v0"
    }
    if args.env_name != "mountain-car":
        model_factory = create_atari_model
        batch_states_process = batch_states_process_atari
        observation_process = observation_process_atari
        reward_process = reward_process_atari
        input_shape = (84, 84, 1)
    else:
        model_factory = create_mountain_cart_model
        batch_states_process = batch_states_process_mountain_cart
        observation_process = observation_process_mountain_cart
        reward_process = reward_process_mountain_cart
        input_shape = None

    env_name = env_choices[args.env_name]
    env = gym.make(env_name)

    agent_choices = {
        "dqn": DQN, "ddqn": DDQN, "a3c": A3C, "distdqn": DistDQN, "ndqn": NDQN
    }

    dqn_agent = agent_choices[args.agent](env=env,
                                          model_factory=model_factory,
                                          batch_states_process=batch_states_process,
                                          observation_process=observation_process,
                                          reward_process=reward_process,
                                          input_shape=input_shape)

    if args.train:
        dqn_agent.train(render=args.render)
    else:
        dqn_agent.load_model(args.model_name)
    dqn_agent.test()


main()
