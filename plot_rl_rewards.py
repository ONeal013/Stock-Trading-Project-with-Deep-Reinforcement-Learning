# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
args = parser.parse_args()

# Charger le tableau des rewards depuis 'rl_trader_rewards'
a = np.load(f'rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

plt.hist(a, bins=30)
plt.title(args.mode)
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.show()
