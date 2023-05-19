import gym
import numpy as np



def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class ImageObservationWrapper(gym.Wrapper):
    def __init__(self, env, seed,channel_first=True, from_pixels=True,height = 84, width =84,frame_skip=1,episode_length=1000):
        gym.Wrapper.__init__(self, env)
        # self.k = k
        # self.frames = deque([], maxlen=k)
        # self.render_kwargs = render_kwargs or {}
        # shp = env.observation_space.shape
        self._channel_first = channel_first
        self._env = env
        self._seed = seed
        self._frame_skip = frame_skip
        self._max_episode_steps = (episode_length + frame_skip - 1) // frame_skip
        if from_pixels:
            shape = [3, height, width] if channel_first else [height, width, 3]
            self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._obs,self._info = self._env.reset()

    def seed(self, seed):
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)
    def reset(self):
        # image = self._env.render()
        # if self._channel_first:
        #     image = image.transpose(2, 0, 1).copy()
        # for _ in range(self.k):
        #     self.frames.append(image)
        return self._get_image_observation()

    def step(self, action):
        obs, reward, done,truncated, info = self._env.step(action)
        # image = self.env.render()
        # if self._channel_first:
        #     image = image.transpose(2, 0, 1).copy()
        # self.frames.append(image)
        return self._get_image_observation(), reward, done,truncated, info

    def _get_image_observation(self):
        image = self._env.render()
        if self._channel_first:
            image = image.transpose(2, 0, 1).copy()
        else:
            image = _flatten_obs(self._env.observation_space)
        return image