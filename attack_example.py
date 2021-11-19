from advertorch.attacks import GradientSignAttack
from atari_wrapper import wrap_deepmind
import copy
import torch
from drl_attacks.uniform_attack import uniform_attack_collector
from drl_attacks.critical_point_attack import critical_point_attack_collector
from utils import A2CPPONetAdapter

from ale_py import ALEInterface


def make_atari_env_watch(env_name):
    return wrap_deepmind(env_name, frame_stack=4,
                         episode_life=False, clip_rewards=False)

# define Pong Atari environment
env = make_atari_env_watch("PongNoFrameskip-v4")
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.env.action_space.shape or env.env.action_space.n
print("Action shape", action_shape)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device", device)
# load pretrained Pong-PPO policy
ppo_pong_path = "log/PongNoFrameskip-v4/ppo/policy.pth"
ppo_policy, _ = torch.load(ppo_pong_path, map_location=torch.device('cpu'))
ppo_policy.to(device).init(device)
print("Original policy loaded")

# adapt PPO policy to Advertorch library
print("ppo policy", ppo_policy)
ppo_adv_net = A2CPPONetAdapter(copy.deepcopy(ppo_policy)).to(device)
ppo_adv_net.eval()
print("ppo adv net", ppo_adv_net)

# define image adversarial attack
eps = 0.1
obs_adv_atk = GradientSignAttack(ppo_adv_net, eps=eps*255,
                                 clip_min=0, clip_max=255, targeted=False)

# define RL adversarial attack
collector = uniform_attack_collector(ppo_policy, env, obs_adv_atk,
                                     perfect_attack=False,
                                     atk_frequency=0.5,
                                     device=device)
'''collector = critical_point_attack_collector(ppo_policy, env, obs_adv_atk,
                                     perfect_attack=False,
                                     device=device)'''


# perform uniform attack with attack frequency of 0.5
collector.atk_frequency = 0.5
print("Here0")
test_adversarial_policy = collector.collect(n_episode=10)
print("Here")
avg_atk_rate = test_adversarial_policy['atk_rate(%)']
print("Here2")
avg_rew = test_adversarial_policy['rew']
avg_num_atks = test_adversarial_policy['n_atks']
avg_succ_atks_rate = test_adversarial_policy['succ_atks(%)']
print("attack frequency (%) =", avg_atk_rate)
print("number of attacks =", avg_num_atks)
print("number of successful attacks (%) =", avg_succ_atks_rate)
print("reward =", avg_rew)