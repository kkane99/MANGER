from torch import nn
from torch.nn import init
import torch as th
import numpy as np
from utils.exploration_utils import check

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x, valid_count=None):
        valid_count=valid_count.cpu().numpy()
        if valid_count is None:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        else:
            batch_count = np.sum(valid_count)
            batch_mean = np.sum(x.reshape((-1, x.shape[-1])), axis=0) / batch_count
            if len(x.shape)==2:
                batch_mean = np.mean(batch_mean)
            var_sum = np.zeros(shape=batch_mean.shape)
            for i, episode in enumerate(x):
                var_sum+=np.sum((episode[:int(valid_count[i])]-batch_mean)**2, axis=0)
            batch_var = var_sum/batch_count
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self, input_size, embedding_size, gamma, device, rnd_step=4e5, intri_reward_masked_rate=0.1):
        super(RNDModel, self).__init__()
        self.gamma = gamma
        self.device = device

        self.rnd_step = rnd_step
        self.rnd_t = 0
        self.intri_reward_masked_rate = intri_reward_masked_rate

        self.input_size = input_size
        self.embedding_size = embedding_size

        # self.tpdv = tpdv

        feature_output = 1024
        self.predictor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, feature_output),
            nn.LeakyReLU(),
            # Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_size)
        ).to(self.device)

        self.target = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, feature_output),
            nn.LeakyReLU(),
            # Flatten(),
            nn.Linear(feature_output, self.embedding_size)
        ).to(self.device)

        self.obs_rms = RunningMeanStd(shape=(1, input_size))
        self.return_rms = RunningMeanStd()

        # Initialize weights    
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        # Set target parameters as untrainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs.to(th.float))
        predict_feature = self.predictor(next_obs.to(th.float))

        return predict_feature, target_feature

    def calculate_intrinsic_reward(self, batch_states, masks_batch, space_mapping=None, intri_reward_masked_rate=None):    # states.shape = 32 * 61 * 48
        if intri_reward_masked_rate is None:
            intri_reward_masked_rate = self.intri_reward_masked_rate

        # debug:
        # space_mapping = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

        # 每一个episode的有效state数量：
        valid_count = th.sum(masks_batch, axis=1)

        self.obs_rms.update(batch_states.cpu().numpy(), valid_count)

        obs_mean = th.tensor(self.obs_rms.mean).to(batch_states.device)
        obs_std = th.tensor(np.sqrt(self.obs_rms.var)).to(batch_states.device)

        batch_states = ((batch_states - obs_mean) / obs_std).clip(-5, 5).to(th.float)

        # map state to sub state
        if space_mapping is not None:
            batch_states = batch_states * check(space_mapping).to(batch_states.device)

        # debug
        if (batch_states[:, :, -2:]!=5).sum() > 0:
            1==1


        # 用next obs计算intrinsic reward
        predict_feature, target_feature = self.forward(batch_states * masks_batch)
        intrinsic_reward_batch=(target_feature - predict_feature).pow(2).sum(-1) / 2

        intrinsic_returns_batch = self._cal_return(reward=intrinsic_reward_batch, mask=masks_batch)
        
        # 更新 intrinsic return 的 mean, var
        self.return_rms.update(intrinsic_returns_batch.detach().cpu().numpy(), valid_count)

        return_mean = self.return_rms.mean
        return_std = np.sqrt(self.return_rms.var)
        '''
        normlization:
            reward: x/std
            return: (x-mean)/std
        '''
        intrinsic_reward_batch = th.unsqueeze(intrinsic_reward_batch, -1) * masks_batch
        intrinsic_reward_batch_mean = (masks_batch.reshape(intrinsic_reward_batch.shape) * intrinsic_reward_batch).sum()/(masks_batch.sum())
        # intrinsic_reward_batch_norm = intrinsic_reward_batch/np.sqrt(self.return_rms.var) if shared_rnd else (intrinsic_reward_batch - th.mean(intrinsic_reward_batch))/np.sqrt(self.return_rms.var)
        intrinsic_reward_batch_norm = (intrinsic_reward_batch - intrinsic_reward_batch_mean)/return_std
        intrinsic_reward_batch_norm = intrinsic_reward_batch_norm * masks_batch
        # intrinsic_return_norm = (intrinsic_returns_batch-return_mean)/return_std

        # masked loss
        if (batch_states[:, :, np.where(space_mapping==1)[0]]!=5).sum() != 0:
            1==1
        mask = th.rand(size=intrinsic_reward_batch.shape).to(self.device)
        mask = mask<intri_reward_masked_rate
        # mask = check(mask)
        rnd_loss = th.sum(mask * intrinsic_reward_batch * masks_batch)

        # if self.rnd_t <= self.rnd_step:
        #     self.rnd_t+=batch_states.shape[0]
        #     corr_f = np.cos((np.pi*self.rnd_t)/(2*self.rnd_step))
        # else: corr_f = 0

        # intrinsic_reward_batch_norm, intrinsic_reward_batch = th.unsqueeze(intrinsic_reward_batch_norm, -1) * masks_batch, th.unsqueeze(intrinsic_reward_batch, -1) * masks_batch

        return intrinsic_reward_batch_norm, intrinsic_reward_batch, rnd_loss
    
    def _cal_return(self, reward, mask):
        intrinsic_returns_batch = th.zeros(size=reward.shape, device=self.device)
        # masks_batch = check(masks_batch)

        for step in reversed(range(reward.shape[1])):
            if step==reward.shape[1]-1:
                intrinsic_returns_batch[:, step]=reward[:, step] * (mask[:, step].reshape((-1)))
            else:
                intrinsic_returns_batch[:, step] = intrinsic_returns_batch[:, step + 1] * self.gamma * (mask[:, step+1].reshape((-1))) + reward[:, step] * (mask[:, step].reshape((-1)))
        return intrinsic_returns_batch