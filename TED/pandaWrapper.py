from collections import deque

import gym
import numpy as np
import cv2

# class PandaObsWrapper(gym.Wrapper):
#     def __init__(self, env, seed, frame_skip, k, img_size, greyscale):
#         gym.Wrapper.__init__(self, env)
#         self._env = env
#         self.frame_skip = frame_skip
#         self._k = k
#         self.image_size = img_size
#         self.convert_to_greyscale = greyscale
#         self._frames = deque([], maxlen=k)
#         self._obs, self._info = self._env.reset()
#         img = self._env.render()[:,120:600,:3]
#         # self.seed(seed)
#         self.
#
#         if self.image_size:
#             img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
#         # print(img.shape)
#         if self.convert_to_greyscale:
#             # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             R = img[:, :, 0]
#             G = img[:, :, 1]
#             B = img[:, :, 2]
#             img = (R * 0.2126 + G * 0.7152 + B * 0.0722).astype(np.uint8)  # Luma formula
#             img = np.expand_dims(img, -1)
#         img = img.transpose(2, 0, 1).copy()
#         shp = img.shape
#         self.observation_space = gym.spaces.Box(
#             low=0,
#             high=1,
#             shape=((shp[0] * k,) + shp[1:]),
#             dtype=img.dtype)
#         self._max_episode_steps = env._max_episode_steps
#
#     def reset(self):
#         obs, info = self._env.reset()
#         img = self._env.render()[:,120:600,:3]
#         if self.image_size:
#             img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
#         if self.convert_to_greyscale:
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             img = np.expand_dims(img, -1)
#         img = img.transpose(2, 0, 1).copy()
#         for _ in range(self._k):
#             self._frames.append(img)
#         return self._get_obs()
#
#     def step(self, action):
#         total_reward = 0
#         for i in range(self.frame_skip):
#             obs, reward, done, truncated, info = self.env.step(action)
#             total_reward += reward
#         img = self._env.render()[:,120:600,:3]
#         if self.image_size:
#             img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
#         if self.convert_to_greyscale:
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             img = np.expand_dims(img, -1)
#         img = img.transpose(2, 0, 1).copy()
#         self._frames.append(img)
#         return self._get_obs(), total_reward, done,info
#
#     def _get_obs(self):
#         assert len(self._frames) == self._k
#         return np.concatenate(list(self._frames), axis=0)
#
#     def __getattr__(self, attr):
#         if hasattr(self._env, attr):
#             return getattr(self._env, attr)
#         raise AttributeError("'{}' object has no attribute '{}'".format(
#             type(self).__name__, attr))
#
#     def render(self):
#         return self._env.render()
# class PandaObsWrapper(gym.Wrapper):
#     def __init__(self, env, seed, frame_skip, k, img_size, greyscale):
#         gym.Wrapper.__init__(self, env)
#         self._env = env
#         self.frame_skip = frame_skip
#         self._k = k
#         self.image_size = img_size
#         self.convert_to_greyscale=greyscale
#         self._frames = deque([], maxlen=k)
#         env.reset()
#         img = env.render()[:,120:600,:3]
#         if self.image_size:
#             img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
#         if self.convert_to_greyscale:
#             #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             R = img[:, :, 0]
#             G = img[:, :, 1]
#             B = img[:, :, 2]
#             img = (R * 0.2126 + G * 0.7152 + B * 0.0722).astype(np.uint8)  # Luma formula
#             img = np.expand_dims(img, -1)
#         img = img.transpose(2, 0, 1).copy()
#         shp = img.shape
#         self.observation_space = gym.spaces.Box(
#             low=0,
#             high=1,
#             shape=((shp[0] * k,) + shp[1:]),
#             dtype=img.dtype)
#         self._max_episode_steps = env._max_episode_steps
#
#     def reset(self):
#         obs = self.env.reset()
#         img = self.env.render()[:,120:600,:3]
#         if self.image_size:
#             img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
#         if self.convert_to_greyscale:
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             img = np.expand_dims(img, -1)
#         img = img.transpose(2, 0, 1).copy()
#         for _ in range(self._k):
#             self._frames.append(img)
#         return self._get_obs()
#
#     def step(self, action):
#         total_reward = 0
#         for i in range(self.frame_skip):
#             obs, reward, done, _, info = self.env.step(action)
#             total_reward += reward
#         img = self.env.render()[:,120:600,:3]
#         if self.image_size:
#             img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
#         if self.convert_to_greyscale:
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             img = np.expand_dims(img, -1)
#         img = img.transpose(2, 0, 1).copy()
#         self._frames.append(img)
#         return self._get_obs(), total_reward, done, info
#
#     def _get_obs(self):
#         assert len(self._frames) == self._k
#         return np.concatenate(list(self._frames), axis=0)
#
#     def __getattr__(self, attr):
#         if hasattr(self._env, attr):
#             return getattr(self._env, attr)
#         raise AttributeError("'{}' object has no attribute '{}'".format(
#             type(self).__name__, attr))
class PandaObsWrapper(gym.Wrapper):
    def __init__(self, env, seed, frame_skip, k, img_size, greyscale):
        gym.Wrapper.__init__(self, env)
        self._env = env
        self.frame_skip = frame_skip
        self._k = k
        self.image_size = img_size
        self.convert_to_greyscale = greyscale
        self._frames = deque([], maxlen=k)
        self._obs, self._info = self._env.reset()
        img = self._env.render()[:,120:600,:3]
        # self.seed(seed)

        if self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        # print(img.shape)
        if self.convert_to_greyscale:
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]
            img = (R * 0.2126 + G * 0.7152 + B * 0.0722).astype(np.uint8)  # Luma formula
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1).copy()
        shp = img.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=img.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs, info = self._env.reset()
        img = self._env.render()[:,120:600,:3]
        if self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.convert_to_greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1).copy()
        for _ in range(self._k):
            self._frames.append(img)
        return self._get_obs()

    def step(self, action):
        total_reward = 0
        for i in range(self.frame_skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
        img = self._env.render()[:,120:600,:3]
        if self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.convert_to_greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1).copy()
        self._frames.append(img)
        return self._get_obs(), total_reward, done,info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

    def __getattr__(self, attr):
        if hasattr(self._env, attr):
            return getattr(self._env, attr)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def render(self):
        return self._env.render()