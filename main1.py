import torch
from MSAC_with_3agent import SAC_Agent
from ReplayBuffer_SAC import RandomBuffer
from RIS_UAV_env import MiniSystem
from Res_plot_up import Res_plot
import numpy as np
np.set_printoptions(threshold=np.inf)
import time
import math
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()

    #env_parameter
    parser.add_argument('--step', type=int, required=False, default=300, help="how many step in each episode")
    parser.add_argument("--power_t", default=30, type=float, metavar='N', help="Transmission power for the constrained optimization in dB")
    parser.add_argument("--num_users", default=3, type=int, metavar='N', help='Number of users')
    parser.add_argument("--num_attacker", default=1, type=int, metavar='N', help='Number of attacker_num')
    parser.add_argument("--num_BS_antennas", default=4, type=int, metavar='N', help='Number of antennas in the BS')
    parser.add_argument("--num_RIS_elements", default=25, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--awgn_power", default=-114, type=float,  help='power of the additive white Gaussian noise at users (dBm)')
    parser.add_argument("--max_ARIS_power", default=-20, type=float, metavar='N', help="ARIS power for the constrained optimization in dBm")
    parser.add_argument("--hot_noise_power", default=-114, type=float, help='power of the hot_noise_power at ARIS (dBm)')

    #algorithm-parameters
    parser.add_argument("--discount", help="set the discount", type=float, default=0.95)
    parser.add_argument("--tau", default=0.005, help='Target network update rate')
    parser.add_argument("--actor_lr", default=2e-4, type=float, metavar='G', help='Learning rate for the actor (and explorer) network (default: 0.001)')
    parser.add_argument("--critic_lr", default=2e-4, type=float, metavar='G', help='Learning rate for the critic network (default: 0.001)')
    parser.add_argument("--alpha_lr", default=2e-4, type=float, metavar='G', help='Learning rate for the critic network (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Entropy coefficient')
    parser.add_argument('--adaptive_alpha', default=True, help='Use adaptive_alpha or Not(SAC)')

    #training-parameters
    parser.add_argument('--ep_upgrade', default=1, type=int, help='Discount factor')
    parser.add_argument('--upgrade_step', default=80, type=float, help='Discount factor')
    parser.add_argument("--seed", default=1, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--buffer", help="set the buffer", type=int, default=40000)
    parser.add_argument('--ep_num', type=int, required=False, default=10000, help="how many episodes do you want to train your DRL")
    parser.add_argument("--batch_size", default=130, metavar='N', help='Mini-batch size (default: 100)')
    parser.add_argument("--dn", default=0.05, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--tn", default=2, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--an", default=2, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--policy", default="SAC", help='Algorithm (default: Beta-Space Exploration)')

    return parser
parser = create_parser()
args = parser.parse_args()

# Set path
# S_path = 'S_user'+str(args.num_users)+'_RIS_num_'+str(args.num_RIS_elements)+ '_attacker' + str(args.num_attacker)+'/'
# mkdir(S_path)
# E_path = 'E_user'+str(args.num_users)+'_RIS_num_'+str(args.num_RIS_elements)+ '_attacker' + str(args.num_attacker)+'/'
# mkdir(E_path)
SEE_path = 'SEE_user' + str(args.num_users) + '_RIS_num_' + str(args.num_RIS_elements) + '_attacker' + str(args.num_attacker)+'/'
mkdir(SEE_path)
amp_num_path = 'amp_num_user' + str(args.num_users) + '_RIS_num_' + str(args.num_RIS_elements) + '_attacker' + str(args.num_attacker)+'/'
mkdir(amp_num_path)
result_path = ('1_GResult_alpha_'+str(args.alpha)+'_seed_'+str(args.seed)+'_upgrade_'+str(args.ep_upgrade)+'batch_size'+str(args.batch_size)+
               'lrAC'+str(args.actor_lr)+str(args.critic_lr)+'tau_'+str(args.tau)+'user'+str(args.num_users)+'RIS_num'+str(args.num_RIS_elements)+'/')
mkdir(result_path[:-1])
# model_path = 'Model_'+str(args.num_users)+'/'
# mkdir(model_path[:-1])
# outdir = "checkpoints"
# mkdir("{}/models".format(outdir))

# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
t1 = time.time()

res_plot_instance = Res_plot(args.num_users, args.num_attacker)

system = MiniSystem(
    user_num=args.num_users,
    attacker_num=args.num_attacker,
    p_total=args.power_t,
    RIS_ant_num=args.num_RIS_elements,
    BS_ant_num=args.num_BS_antennas,
    awgn_power=args.awgn_power,
    seed=args.seed,
    step=args.step,
    max_ARIS_power=args.max_ARIS_power,
    hot_noise_power=args.hot_noise_power,
    )

# SD3_initial
# max_action
dn_max = args.dn
an_max = args.an
top_ = np.array([math.pi/2, math.pi/2, dn_max])
amp_array = np.full(args.num_RIS_elements, an_max/2)
phi_array = np.full(args.num_RIS_elements, math.pi)
beamforming_array1 = np.ones(2*args.num_users*args.num_BS_antennas)
max_action = np.concatenate((top_, amp_array, phi_array, beamforming_array1))  # 连接几个数组，构建最大动作的组合区域
#print('max_action',max_action)
s_dim = system.get_system_state_dim()
a_dim_UAV, a_dim_RIS, a_dim_BS, a_dim = system.get_system_action_dim()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kwargs = {
    "state_dim": s_dim,
    "action_dim": a_dim,
    "action_dim_ARISUAV": a_dim_RIS+3,
    # "action_dim_UAV": a_dim_UAV,
    # "action_dim_RIS": a_dim_RIS,
    # "action_dim_BS": a_dim_BS,
    "tau": args.tau,
    "gamma": args.discount,
    "hid_shape": [256, 512],
    "a_lr": args.actor_lr,
    "c_lr": args.critic_lr,
    "alpha_lr": args.alpha_lr,
    "batch_size": args.batch_size,
    "alpha": args.alpha,
    "adaptive_alpha": args.adaptive_alpha,
    "device": device
    }
model = SAC_Agent(**kwargs)

# train
replay_buffer = RandomBuffer(s_dim, a_dim,  Env_with_dead=False, max_size=int(args.buffer))
episode_sum_reward = []
secrecy_rate = []
energy_consumption = []
SEE = []

total_episode = args.ep_num
step = args.step
for episode in range(1, total_episode + 1):
    t3 = time.time()
    t = system.reset()  # t=0,位置等变量初始化
    sum_reward = 0
    sum_secrecy = 0
    sum_energy_consumption = 0
    amp_num_set = []
    see = 0
    done = False
    # 每回合UAV的初始位置都一样
    x0_uav = []
    y0_uav = []
    z0_uav = []
    location = np.zeros([3, 1])
    location[0][0] = system.ARISUAV.coordinate[0]
    location[1][0] = system.ARISUAV.coordinate[1]
    location[2][0] = system.ARISUAV.coordinate[2]
    x0_uav = np.append(x0_uav, location[0][0])
    y0_uav = np.append(y0_uav, location[1][0])
    z0_uav = np.append(z0_uav, location[2][0])
    # get the initial state
    s = system.observe()
    while not done:
        amp_num = 0
        a_low = model.select_action(s, deterministic=False, with_logprob=False)
        a = a_low * max_action
        for i in range(args.num_RIS_elements):
            a[3+i] += an_max / 2
            if a[3+i] > 1:
                amp_num += 1
        amp_num_set = np.append(amp_num_set, amp_num)
        secrecy, energy, r, s_, done, t = system.step(a, t)  # 执行动作，t=t+1
        sum_reward += r
        sum_secrecy += secrecy
        sum_energy_consumption += energy
        see += secrecy / energy
        replay_buffer.add(s, a_low, r, s_, done)  # put a transition in buffer
        location[0][0] = system.ARISUAV.coordinate[0]
        location[1][0] = system.ARISUAV.coordinate[1]
        location[2][0] = system.ARISUAV.coordinate[2]
        x0_uav = np.append(x0_uav, location[0][0])
        y0_uav = np.append(y0_uav, location[1][0])
        z0_uav = np.append(z0_uav, location[2][0])
        s = s_
        # if replay_buffer.size >= 3000 and t % args.ep_upgrade == 0:
        #     for step in range(args.upgrade_step):
        #         model.train(replay_buffer)
    if replay_buffer.size >= 3000 and episode % args.ep_upgrade == 0:
        for step in range(args.upgrade_step):
            model.train(replay_buffer)
        #print('t',t)
    episode_sum_reward = np.append(episode_sum_reward, sum_reward)
    secrecy_rate = np.append(secrecy_rate, sum_secrecy)
    energy_consumption = np.append(energy_consumption, sum_energy_consumption)
    SEE = np.append(SEE, see)

    # print("energy",sum_energy_consumption,"secrecy",sum_secrecy,"SEE",see)
    print('Episode:', episode, 'sum_Reward: %.16f' % sum_reward)
    # plot_path_picture
    if episode % 50 == 0 and episode >= 2000:
        np.save(os.path.join(amp_num_path, f"secrecy_energy_user_{args.num_users}_RIS_num_{args.num_RIS_elements}_episode_{episode}.npy"), amp_num_set)
        plt.figure()
        plt.plot(amp_num_set, label='amp_num_set', color='red')
        plt.xlabel('Iterations')
        plt.ylabel('amp_num')
        plt.title('amp_num')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(amp_num_path, f'amp_episode_{episode}.png'))
        # res_plot_instance.draw_location(x0_uav, y0_uav, z0_uav,
        #                                    result_path + 'UAVPath_Users_3D_%s.png' % str(episode).zfill(2))
        res_plot_instance.draw_location_2d_xoy(x0_uav, y0_uav,
                                           result_path + 'UAVPath_Users_2D_xoy%s.png' % str(episode).zfill(2))
    #     res_plot_instance.draw_location_2d_yoz(y0_uav, z0_uav,
    #                                        result_path + 'UAVPath_Users_2D_yoz%s.png' % str(episode).zfill(2))
    #     if episode % 50 == 0 and episode >= 4000:
    #         np.save(os.path.join(result_path, f"4_path_x.npy_{episode}"), x0_uav)
    #         np.save(os.path.join(result_path, f"4_path_y.npy_{episode}"), y0_uav)
    #         np.save(os.path.join(result_path, f"4_path_z.npy_{episode}"), z0_uav)
    plt.figure()
    plt.plot(episode_sum_reward, label='reward', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('reward')
    plt.title('episode_sum_reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f'1G_alpha_{args.alpha}_seed_{args.seed}_upgrade_{args.ep_upgrade}_batch_size_{args.batch_size}_lrAC_{args.actor_lr}_{args.critic_lr}_tau_{args.tau}.png')
    print(time.time() - t3)
# model.save('{}/models'.format(outdir))
plt.figure()
plt.plot(episode_sum_reward, label='reward', color='red')
plt.xlabel('Iterations')
plt.ylabel('reward')
plt.title('episode_sum_reward')
plt.legend()
plt.grid(True)
plt.savefig(f'1G_alpha_{args.alpha}_seed_{args.seed}_upgrade_{args.ep_upgrade}_batch_size_{args.batch_size}_lrAC_{args.actor_lr}_{args.critic_lr}_tau_{args.tau}.png')
# plt.show()
# filename = f"1Result_alpha_{args.alpha}_seed_{args.seed}_upgrade_{args.ep_upgrade}_batch_size_{args.batch_size}_lrAC_{args.actor_lr}_{args.critic_lr}"
# np.save(filename + "reward.npy", episode_sum_reward)
#
# np.save(os.path.join(S_path, f"secrecy_rate_user_{args.num_users}_RIS_num_{args.num_RIS_elements}.npy"), secrecy_rate)
# np.save(os.path.join(E_path, f"energy_consumption_user_{args.num_users}_RIS_num_{args.num_RIS_elements}.npy"), energy_consumption)
# np.save(os.path.join(amp_num_path, f"secrecy_energy_user_{args.num_users}_RIS_num_{args.num_RIS_elements}.npy"),
#         amp_num_set)
# np.save(os.path.join(SEE_path, f"secrecy_energy_user_{args.num_users}_RIS_num_{args.num_RIS_elements}.npy"),
#         SEE)
# plt.figure()
# plt.plot(secrecy_rate, label='secrecy_rate', color='blue')
# plt.xlabel('Episodes')
# plt.ylabel('Secrecy Rate')
# plt.title('Secrecy Rate per Episode')
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(S_path, f'secrecy_rate_progress_user_{args.num_users}_RIS_num_{args.num_RIS_elements}.png'))
# #plt.close()
#
# plt.figure()
# plt.plot(energy_consumption, label='energy_consumption', color='green')
# plt.xlabel('Episodes')
# plt.ylabel('Energy Consumption')
# plt.title('Energy Consumption per Episode')
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(E_path, f'energy_consumption_progress_user_{args.num_users}_RIS_num_{args.num_RIS_elements}.png'))
# plt.close()

# plt.figure()
# plt.plot(amp_num_set, label='amp_num', color='green')
# plt.xlabel('Episodes')
# plt.ylabel('amp_num')
# # plt.title('Secrecy Energy per Episode')
# plt.legend()
# plt.grid(True)
# plt.savefig(
#     os.path.join(amp_num_path, f'amp_num_progress_user_{args.num_users}_RIS_num_{args.num_RIS_elements}.png'))

# plt.figure()
# plt.plot(SEE, label='secrecy_energy', color='green')
# plt.xlabel('Episodes')
# plt.ylabel('Secrecy Energy')
# plt.title('Secrecy Energy per Episode')
# plt.legend()
# plt.grid(True)
# plt.savefig(
#     os.path.join(SEE_path, f'secrecy_energy_progress_user_{args.num_users}_RIS_num_{args.num_RIS_elements}.png'))

print('Running time:', time.time() - t1)