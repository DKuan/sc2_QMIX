import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self._now_idx = None

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0
        self._now_idx = None

    def add(self, obs, u_last, hidden_last, state, \
                u, hidden, obs_new, state_new, r, done):
        data = (obs, u_last, hidden_last, state, \
                u, hidden, obs_new, state_new, r, done)
        self._storage[self._now_idx].append(data)

    def create_new_episode(self):
        if self._next_idx >= len(self._storage):
            self._storage.append([])
        else:
            self._storage[self._next_idx].clear()
        self._now_idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obs_t, u_last_t, hidden_last_t, state_t, u_t, hidden_t, \
            obs_new_t, state_new_t, r_t, done_t = [], [], [], [], [], [], [], [], [], []
        for i in idxes:
            data_episode = self._storage[i]
            for data in data_episode:
                obs, u_last, hidden_last, state, u, hidden, \
                    obs_new, state_new, r, done = data # episode_data
                obs_t.append(obs)
                u_last_t.append(u_last)
                hidden_last_t.append(hidden_last)
                state_t.append(state)
                u_t.append(u)
                hidden_t.append(hidden)
                obs_new_t.append(obs_new)
                state_new_t.append(state_new)
                r_t.append(r)
                done_t.append(done)
        return np.array(obs_t), np.array(u_last_t), np.array(hidden_last_t), np.array(state_t), \
            np.array(u_t), np.array(hidden_t), np.array(obs_new_t), \
            np.array(state_new_t), np.array(r_t), np.array(done_t)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
        act_batch: np.array
        rew_batch: np.array
        next_obs_batch: np.array
        done_mask: np.array
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
