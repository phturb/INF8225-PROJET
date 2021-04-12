from rl.core import Processor


class ProcessMountainCar(Processor):
    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        return batch.reshape(-1, 2)
