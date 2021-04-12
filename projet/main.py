import * from models_env_metadata
from dqn import DQNv2
from ddqn import DDQN


def main():
    env = gym.make("Pong-v0")
    # dqn_agent = DQNv2(env=env, model_factory=create_atari_model,
    #                   batch_states_process=batch_states_process_atari,
    #                   observation_process=observation_process_atari,
    #                   reward_process=reward_process_atari,
    #                   input_shape=(84, 84, 1))
    env = gym.make("MountainCar-v0")
    dqn_agent = DQNv2(env=env, model_factory=create_mountain_cart_model,
                      batch_states_process=batch_states_process_mountain_cart,
                      observation_process=observation_process_mountain_cart,
                      reward_process=reward_process_mountain_cart)
    if TEST:
        dqn_agent.load_model()
        dqn_agent.test()
    else:
        dqn_agent.train()


main()
