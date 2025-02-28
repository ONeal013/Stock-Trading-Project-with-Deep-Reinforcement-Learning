# RL Trader: A Reinforcement Learning Stock Trading Agent

RL Trader is a deep reinforcement learning project designed to train an AI agent to trade stocks using historical price data. The project features a custom multi-stock trading environment, a Deep Q-Network (DQN) agent, and utility functions for data scaling and replay memory management. This document serves as an overview of the project, explaining its structure, installation, and usage.

---

## Overview

The goal of RL Trader is to develop an agent that can learn optimal trading strategies by interacting with a simulated stock market environment. The agent is trained on historical data for three stocks:
- **APPL**
- **MSI**
- **SBUX**

Key components of the project include:
- **Data Processing:** Loading and preprocessing stock price data.
- **Environment:** A custom multi-stock trading environment (`MultiStockEnv`) that simulates buying, selling, and holding actions.
- **Agent:** A DQN-based agent (`DQNAgent`) that uses a replay buffer to store experiences and learns from them to maximize future rewards.
- **Model Building:** A simple multilayer perceptron (MLP) that predicts Q-values for available actions.
- **Scaling:** A standard scaler is used to normalize state data for better learning performance.
- **Evaluation:** After training, the agent's performance is measured by the portfolio value achieved over episodes.
- **Visualization:** A separate script plots the rewards (portfolio value increases) over training/testing episodes.

---

## Project Structure

- **rl_trader.py**  
  Contains the main logic for the reinforcement learning agent, including the environment, replay buffer, agent implementation, and training/testing loop.

- **plot_rl_rewards.py**  
  A utility script that loads the saved portfolio values and generates a histogram to visualize the distribution of rewards (portfolio value increases) achieved during training or testing.

- **aapl_msi_sbux.csv**  
  The CSV file containing historical stock price data for APPL, MSI, and SBUX. This data is used to simulate the trading environment.

---

## Features

- **Custom Trading Environment:**  
  Simulates a stock market for three stocks with a state representation that includes the number of shares owned, current stock prices, and available cash.

- **Deep Q-Network Agent:**  
  Utilizes an MLP model to predict Q-values for each possible action. The agent learns to balance exploration and exploitation using an epsilon-greedy strategy.

- **Replay Buffer:**  
  Stores past experiences to sample mini-batches for training the neural network, thereby stabilizing learning.

- **State Scaling:**  
  Normalizes state data using a `StandardScaler` to improve the efficiency and performance of the learning process.

- **Model Saving and Loading:**  
  Trained model weights and scalers are saved for later testing and evaluation.

- **Reward Visualization:**  
  Provides a script to plot a histogram of portfolio value increases, enabling performance analysis of the agent over multiple episodes.

---

## Installation and Setup

1. **Install Dependencies:**  
   Ensure that Python 3.x is installed along with the required libraries:
   - numpy
   - pandas
   - tensorflow (and keras)
   - scikit-learn
   - matplotlib
   - argparse

2. **Prepare the Data:**  
   Place the `aapl_msi_sbux.csv` file in the project directory. This file should contain the historical stock price data in a T x 3 format.

3. **Project Files:**  
   Ensure all project files (`rl_trader.py`, `plot_rl_rewards.py`, and others) are in the same working directory.

---

## Usage

### Training the Agent

To train the RL agent, run the main script with the training mode enabled:
python rl_trader.py --mode train

This command will:
- Load and preprocess the historical stock data.
- Initialize the custom trading environment.
- Train the DQN agent over a specified number of episodes (default is 2000).
- Save the trained model weights and scaler for future use.
- Record the portfolio value at the end of each episode.

### Testing the Agent

After training, test the agent by running:
python rl_trader.py --mode test

This mode:
- Loads the saved model weights and scaler.
- Uses the test portion of the data to evaluate the agent’s performance.
- Sets a low exploration rate to prioritize learned behavior over random actions.

### Visualizing Rewards

To analyze the training/testing results, plot the reward distribution using:
python plot_rl_rewards.py --mode train






python plot_rl_rewards.py --mode test
The script will display a histogram showing the frequency of portfolio value increases, along with basic statistics such as average, minimum, and maximum rewards.

---

## Code Structure Details

- **Environment (`MultiStockEnv`):**  
  - **State:** Consists of the number of shares owned for each stock, current stock prices, and available cash.
  - **Action Space:** Encoded as a discrete set of actions for buying, holding, or selling each stock.
  - **Reward:** Defined as the change in portfolio value after each action.

- **Replay Buffer:**  
  Stores transitions (state, action, reward, next state, done flag) and allows sampling of mini-batches for training.

- **DQN Agent (`DQNAgent`):**  
  - **Act:** Chooses actions using an epsilon-greedy strategy.
  - **Replay:** Samples experiences from the replay buffer to perform a gradient descent step on the network.
  - **Training:** Updates the neural network to minimize the difference between predicted and target Q-values.

- **Model Builder (MLP):**  
  Constructs a neural network with a configurable number of hidden layers and neurons to predict Q-values for each possible action.

---

## Contributing

Contributions and suggestions for improvement are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

---

## License

This project is distributed under an open-source license. Please refer to the LICENSE file for further details.

---

Enjoy exploring and enhancing the RL Trader project!
