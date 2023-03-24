import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os
import pandas as pd
import csv

if __name__ == "__main__":

    args = get_args()
    eval_file_path_1 = args.save_dir + '/DQN_Grid3by3/DiscreteRL_episode_return_queue_reward_flowprob0.5.npy'
    eval_file_path_2 = args.save_dir + '/DQN_Grid3by3/DiscreteRL_prevstep_episode_return_queue_reward_flowprob0.5.npy'
    eval_file_path_3 = args.save_dir + '/DQN_Grid3by3/DiscreteRL_LSTM_episode_return_queue_reward_flowprob0.5.npy'
    eval_file_path_4 = args.save_dir + '/DQN_Grid3by3/DiscreteRL_RNN_episode_return_queue_reward_flowprob0.5.npy'
    # eval_file_path_5 = args.save_dir + '/DQN_Grid3by3/DiscreteRL_episode_return_queue_reward_flow_timevariant.npy'
    #
    # eval_file_path_11 = args.save_dir + '/DQN_Grid3by3/DiscreteRL_avg_traffic_load_queue_reward_flow_timevariant.npy'
    # eval_file_path_12 = args.save_dir + '/DQN_Grid3by3/DiscreteRL_OneModel_avg_traffic_load_queue_reward_flow_timevariant.npy'

    # data = np.load(eval_file_path)
    data1 = np.load(eval_file_path_1)
    data2 = np.load(eval_file_path_2)
    data3 = np.load(eval_file_path_3)
    data4 = np.load(eval_file_path_4)
    # data5 = np.load(eval_file_path_5)

    # data11 = np.load(eval_file_path_11)
    # data12 = np.load(eval_file_path_12)

    # print(data1.shape)

    data1_all  = data1.mean(axis=1)
    data2_all  = data2.mean(axis=1)
    data3_all  = data3.mean(axis=1)
    data4_all  = data4.mean(axis=1)
    # data5_all  = data5.mean(axis=1)
    # print(data1_all.shape)

    # np.save(args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_All_rewards_jointlearning_maxvehicle30.npy', data_all1)
    # np.save(args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_All_rewards_independentlearning_maxvehicle30.npy', data_all2)
    # np.save(args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_All_rewards_jointlearning_deepernets_maxvehicle30.npy', data_all3)

    # x = np.linspace(0, len(data), len(data))
    # x1 = np.linspace(0, len(data1), len(data1))

    x1 = np.linspace(0, len(data1), len(data1))
    x2 = np.linspace(0, len(data2), len(data2))
    x3 = np.linspace(0, len(data3), len(data3))
    x4 = np.linspace(0, len(data4), len(data4))
    # x5 = np.linspace(0, len(data5), len(data5))

    # x11 = np.linspace(0, len(data11), len(data11))
    # x12 = np.linspace(0, len(data12), len(data12))

    mpl.style.use('ggplot')
    fig = plt.figure(1)
    fig.patch.set_facecolor('white')
    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Cumulative Return', fontsize=16)
    plt.title('DQN', fontsize=20)
    plt.plot(x1, data1_all, color='purple', linewidth=2, label='DiscreteRL, flow0.5')
    plt.plot(x2, data2_all, color='red', linewidth=2, label='DiscreteRL+prevstep, flow0.5')
    plt.plot(x3, data3_all, color='blue', linewidth=2, label='DiscreteRL+LSTM, flow0.5')
    plt.plot(x4, data4_all, color='green', linewidth=2, label='DiscreteRL+RNN, flow0.5')
    # plt.plot(x4, data4_all, color='orange', linewidth=2, label='MARL+prevstep+extraNBlayer, close nbs, time variant flow')

    plt.legend(loc='lower left')
    plt.show()

    # fig1 = plt.figure(1)
    # fig1.patch.set_facecolor('white')
    # plt.xlabel('Episodes', fontsize=16)
    # plt.ylabel('Average traffic load', fontsize=16)
    # plt.title('DQN', fontsize=20)
    # plt.plot(x11, data11, color='red', linewidth=2, label='discrete learning')
    # plt.plot(x12, data12, color='blue', linewidth=2, label='onemodel learning')
    # plt.legend(loc='center right')
    # plt.show()
