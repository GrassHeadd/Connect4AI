import numpy as np
import random
import pickle
from aiFinal import AIAgentDef, AIAgent, AIAgentAttack, AIAgentNonOp
from connect_four import ConnectFour
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from copy import deepcopy
from functools import partial

class WeightOptimizer:
    def __init__(self, population_size=50, generations=100):
        self.population_size = population_size
        self.generations = generations
        self.stagnation_limit = 30  # Early stopping if no improvement over this many generations
        
        # Initial weight ranges are broader for greaterß diversity
        self.initial_ranges = {
            'horizontal_immediate_threat_score': (30, 200),
            'horizontal_potential_threat_score': (5, 40),
            'vertical_immediate_threat_score': (30, 200),
            'vertical_potential_threat_score': (5, 40),
            'diagonal_immediate_threat_score': (30, 200),
            'diagonal_potential_threat_score': (5, 40),
            'aggressive_defensive_balance_score': (0.2, 2.4),
            'fork_score': (100, 400),
            'vertical_bridge_score': (10, 10),
            'horizontal_bridge_score': (10, 100),
            'center_control_score': (30, 100),
            'height_penalty': (1, 10),
            'mobility_bonus': (0, 50)
        }

    def create_individual(self):
        """Create a random set of weights within specified ranges for initial diversity"""
        return {key: random.uniform(*range) for key, range in self.initial_ranges.items()}

    def evaluate_fitness(self, args):
        """Evaluate fitness by playing against itself"""
        weights, generation, player2_class = args  # Unpack the tuple
        agent1 = AIAgent(1)  # Always use AIAgentDef for the first player
        total_wins = 0
        
        # Apply weights to the first agent
        agent1.weights = weights
        
        num_games = 4 + (generation // 10)  # Adjust number of games based on generation
        
        for _ in range(num_games):
            result = self.simulate_game(AIAgent, player2_class)  # Play against the specified second agent
            if result == 1:
                total_wins += 1
        
        return total_wins / num_games

    def simulate_game(self, player1_class, player2_class):
        """Simulate a game between two agents and return the winner (1 or 2) or 0 for draw"""
        game = ConnectFour()
        player1 = player1_class(1)  # Create an instance of the first player
        player2 = player2_class(2)  # Create an instance of the second player
        current_player = player1
        
        while not game.is_end():
            move = current_player.make_move(game.get_state())
            if move is None or not isinstance(move, int):
                return 2 if current_player == player1 else 1
            
            game.step((move, current_player.player_id))
            
            if game.is_win():
                return current_player.player_id
            
            current_player = player2 if current_player == player1 else player1
            
        return 0  # Draw

    def optimize(self):
        """Main optimization loop with improved selection"""
        population = [self.create_individual() for _ in range(self.population_size)]
        best_weights = None
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in tqdm(range(self.generations), desc="Optimizing", unit="generation"):
            # Prepare arguments for fitness evaluation
            fitness_args = [(weights, generation, AIAgentAttack) for weights in population]  # Change AIAgent to the desired opponent class
            
            # Evaluate fitness for all individuals
            with ProcessPoolExecutor() as executor:
                fitness_scores = list(executor.map(self.evaluate_fitness, fitness_args))
            
            # Track best individual
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_weights = deepcopy(population[max_fitness_idx])
                generations_without_improvement = 0
                print(f"\nGeneration {generation}: New best fitness = {best_fitness:.3f}")
                for key, value in best_weights.items():
                    print(f"{key}: {value:.3f}")
                
                # Save best weights
                with open('best_weights.pkl', 'wb') as f:
                    pickle.dump(best_weights, f)
            else:
                generations_without_improvement += 1
            
            # Early stopping if no improvement for many generations
            if generations_without_improvement >= 30:
                print("\nStopping early due to lack of improvement")
                break
            
            # Selection with elitism
            elite_size = max(2, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite = [deepcopy(population[i]) for i in elite_indices]
            
            # Tournament selection for rest of population
            parents = []
            while len(parents) < self.population_size - elite_size:
                tournament = random.sample(list(enumerate(fitness_scores)), 3)
                winner_idx = max(tournament, key=lambda x: x[1])[0]
                parents.append(population[winner_idx])
            
            # Create next generation
            next_population = elite  # Keep elite individuals
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)
            
            population = next_population
        
        return best_weights

    def crossover(self, parent1, parent2):
        """Weighted average crossover for creating a child with values between parents"""
        child = {}
        for key in self.initial_ranges.keys():
            weight = random.uniform(0.4, 0.6)
            child[key] = parent1[key] * weight + parent2[key] * (1 - weight)
        return child

    def mutate(self, individual):
        """Mutation with controlled changes and occasional larger mutations"""
        mutated = deepcopy(individual)
        for key in self.initial_ranges.keys():
            if random.random() < 0.2:  # 20% mutation rate for slight changes
                mutated[key] *= random.uniform(0.9, 1.1)
            elif random.random() < 0.05:  # 5% chance for a larger change
                mutated[key] *= random.uniform(0.7, 1.3)
        return mutated

if __name__ == "__main__":
    optimizer = WeightOptimizer(population_size=50, generations=100)
    best_weights = optimizer.optimize()
    
    print("\nOptimization complete! Best weights found:")
    for key, value in best_weights.items():
        print(f"{key}: {value:.3f}")

