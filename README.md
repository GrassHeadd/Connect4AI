# Connect4AI
### General Information

This project features an AI for playing Connect 4, utilizing the Minimax algorithm and Reinforcement Learning. You can play against the AI and see how it performs using different strategies.

### Disclaimers

This project is based on and edited for a submission for a contest for the NUS Intro to Artificial Intelligence module. I do not own the implementations apart from the different AI agents and the `tune-weights.py` file. While Connect 4 is a solved game, there are limitations in place by the school, and hence the AI does not play perfectly.

### Files and Structure

- `agents.py`: Contains the implementation of different AI agents.
- `aiFinal.py`: Contains the final AI agent using advanced strategies.
- `pygame_simulator.py`: The main file to run the game using Pygame.
- `tune-weights.py`: Contains a reinforced learning algorithm that alters the weights of the aiAgents

### How to Play

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Open `pygame_simulator.py`.
4. Choose the desired AI agent, you can view all the available agents in agents.py and aiFinal.py and change it in line 102 of pygame_simulator.py
5. Run the script to start playing.

### Requirements

- Python 3.x
- Pygame

### Installation

Install the required dependencies using pip:

```sh
pip install pygame
```

### Running the Game

To run the game, execute the following command:

```sh
python pygame_simulator.py
```

Enjoy playing against the AI and see how it performs using different strategies!
