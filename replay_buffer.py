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

    def add(self, data):
        #data = bs_and_u_last, state, \
                #u, new_avail_actions, obs_new, state_new, r, done
        if self._storage[self._now_idx].__len__() == 0:
            self._storage[self._now_idx] = [np.array(data_item) for data_item in data]
        else:
            for item_idx, item in enumerate(data):
                self._storage[self._now_idx][item_idx] = np.vstack((self._storage[self._now_idx][item_idx], item))

    def create_new_episode(self):
        """ add a check step, in case the game is end without done """
        if self._storage.__len__() > 0 and self._storage[self._now_idx][-1][-1] != True:
            self._storage[self._now_idx].clear()
            return # end without add new memor

        if self._next_idx >= len(self._storage):
            self._storage.append([])
        else:
            self._storage[self._next_idx].clear()
        self._now_idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        #     obs_and_last_action_t, state_t, u_t, new_avail_actions_t, \
        # obs_new_t, state_new_t, r_t, done_t = data_episode[:]
        data_encode_all = None

        # get the max episode len
        max_episode_len = 0
        for idx_id, idx in enumerate(idxes): #
            data_episode = self._storage[idx]
            max_episode_len = max_episode_len if data_episode[0].shape[0] < max_episode_len \
                else data_episode[0].shape[0]

        # get the batch data and fill zeros to small shape data
        num_diff_lens = []
        for idx_id, idx in enumerate(idxes):
            data_episode = self._storage[idx]
            num_diff_len = max_episode_len - data_episode[0].shape[0]
            num_diff_lens.append(num_diff_len)
            data_episode = [np.vstack([data, np.zeros((num_diff_len,)+data[0].shape)])[np.newaxis, :] \
                for data in data_episode]

            if idx_id == 0: 
                data_encode_all = [data for data in data_episode] 
            else: 
                for item_idx, item in enumerate(data_episode):
                    data_encode_all[item_idx] = np.vstack([data_encode_all[item_idx], item])

        return data_encode_all[:], num_diff_lens

    def make_index(self, batch_size):
        len_now = len(self._storage) - 1
        index_list = []
        for _ in range(batch_size):
            rand_idx = random.randint(0, len_now-1)
            while rand_idx == self._now_idx:
                rand_idx = random.randint(0, len_now-1) 
            index_list.append(rand_idx)
        return index_list

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
            idxes = range(0, len(self._storage)-1)
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
