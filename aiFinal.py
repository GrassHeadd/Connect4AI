from game_utils import initialize, step, get_valid_col_id, is_win
import numpy as np

class AIAgent(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id

        if self.player_id == 1:
            self.OPTIMAL_LIMIT = 2
            self.OPTIMAL_MOVES = [3, # depth 0 
                                  3, 5, 5, 3, 1, 1, 3, # depth 1
                                  3, 3, 3, 3, 3, 3, 3, # depth 2 col 0
                                  4, 4, 3, 4, 5, 4, 3, # depth 2 col 1
                                  6, 3, 6, 3, 4, 6, 5, # depth 2 col 2
                                  4, 3, 3, 3, 3, 3, 2, # depth 2 col 3
                                  1, 0, 2, 3, 0, 3, 0, # depth 2 col 4
                                  1, 2, 1, 2, 3, 2, 2, # depth 2 col 5
                                  3, 3, 3, 3, 3, 3, 3] # depth 2 col 6
            # Going First Weights
            self.weights = {
                'horizontal_immediate_threat_score': 160.0,
                'horizontal_potential_threat_score': 50.0,
                'vertical_immediate_threat_score': 170.0,
                'vertical_potential_threat_score': 55.0,
                'diagonal_immediate_threat_score': 175.0,
                'diagonal_potential_threat_score': 60.0,
                'aggressive_defensive_balance_score': 1.0,
                'fork_score': 425.0,
                'vertical_bridge_score': 40.0,
                'horizontal_bridge_score': 50.0,
                'center_control_score': 100.0,
                'height_penalty': 1.5,
                'mobility_bonus': 45.0
            }
        else:
            self.OPTIMAL_LIMIT = 1
            self.OPTIMAL_MOVES = [3, 2, 3, 3, 3, 4, 3,
                                  3, 3, 3, 3, 3, 3, 2,
                                  2, 1, 2, 3, 2, 2, 2,
                                  3, 3, 2, 3, 3, 3, 3,
                                  3, 2, 4, 3, 2, 4, 3,
                                  3, 3, 3, 3, 4, 3, 3,
                                  4, 4, 4, 3, 4, 5, 4,
                                  4, 3, 3, 3, 3, 3, 2]
            # Going Second Weights
            self.weights = {
                'horizontal_immediate_threat_score': 170.0,
                'horizontal_potential_threat_score': 35.0,
                'vertical_immediate_threat_score': 190.0,
                'vertical_potential_threat_score': 50.0,
                'diagonal_immediate_threat_score': 185.0,
                'diagonal_potential_threat_score': 55.0,
                'aggressive_defensive_balance_score': 1.8,
                'fork_score': 375.0,
                'vertical_bridge_score': 45.0,
                'horizontal_bridge_score': 40.0,
                'center_control_score': 80.0,
                'height_penalty': 2.0,
                'mobility_bonus': 50.0
            }

        self.MAX_DEPTH = 4
        self.MAX_VALUE = 999999
        
        # Transposition table for memoization
        self.transposition_table = {}
        self.max_table_size = 50000
        self.clear_threshold = 40000

    def evaluate_position(self, state):
        if is_win(state):
            return self.MAX_VALUE
            
        score = 0

        def count_threats(player):
            threat_score = 0
            immediate_threats = 0
            potential_threats = 0

            # Horizontal threats
            for row in range(6):
                for col in range(4):
                    window = state[row, col:col+4]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['horizontal_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['horizontal_potential_threat_score']
            
            # Vertical threats
            for row in range(3):
                for col in range(7):
                    window = state[row:row+4, col]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['vertical_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['vertical_potential_threat_score']
            
            # Diagonal threats
            for row in range(3):
                for col in range(4):
                    window_positive = [state[row+i][col+i] for i in range(4)]
                    player_count_positive = sum(1 for x in window_positive if x == player)
                    empty_count_positive = sum(1 for x in window_positive if x == 0)
                    
                    if player_count_positive == 3 and empty_count_positive == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_positive == 2 and empty_count_positive == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
                    
                    window_negative = [state[row+3-i][col+i] for i in range(4)]
                    player_count_negative = sum(1 for x in window_negative if x == player)
                    empty_count_negative = sum(1 for x in window_negative if x == 0)
                    
                    if player_count_negative == 3 and empty_count_negative == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_negative == 2 and empty_count_negative == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
            
            return threat_score, immediate_threats, potential_threats

        player_threats, player_immediate, _ = count_threats(self.player_id)
        opponent_threats, opponent_immediate, _ = count_threats(self.opponent_id)

        # Aggressive-defensive balance
        score += player_threats - opponent_threats * self.weights['aggressive_defensive_balance_score']
        
        # Fork detection
        if player_immediate > 1:
            score += self.weights['fork_score']
        if opponent_immediate > 1:
            score -= self.weights['fork_score']

        # Vertical bridge detection
        for col in range(7):
            if np.count_nonzero(state[:, col] == self.player_id) == 2:
                empty_below = np.count_nonzero(state[5:, col] == 0)
                if empty_below > 0:
                    score += self.weights['vertical_bridge_score']

        # Horizontal bridge detection
        for row in range(6):
            for col in range(5):
                if state[row, col] == self.player_id and state[row, col + 1] == self.player_id:
                    if (col - 1 >= 0 and state[row, col - 1] == 0) or (col + 2 < 7 and state[row, col + 2] == 0):
                        score += self.weights['horizontal_bridge_score']
        
        # Center control bonus
        center_array = state[:, 3]
        center_score = np.count_nonzero(center_array == self.player_id) * self.weights['center_control_score']
        score += center_score

        # Height penalty to prefer lower positions
        for col in range(7):
            col_height = np.count_nonzero(state[:, col])
            if col_height > 0:
                score -= col_height * self.weights['height_penalty']
        
        # Mobility bonus
        valid_moves = get_valid_col_id(state)
        score += len(valid_moves) * self.weights['mobility_bonus']
        
        return score


    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
            # Transposition table lookup
            state_hash = hash(state.tobytes())
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    return stored_value, stored_move

            # Base cases with checks
            if is_win(state):
                return (-self.MAX_VALUE if maximizing_player else self.MAX_VALUE), None
                
            if depth == 0:
                return self.evaluate_position(state), None

            valid_moves = get_valid_col_id(state)
            
            # Handle edge cases
            if len(valid_moves) == 0:
                return 0, None
                
            if len(valid_moves) == 1:
                return 0, valid_moves[0]
            
            # Main search logic
            if maximizing_player:
                value = float('-inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.player_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, False)
                    
                    if eval_score > value:
                        value = eval_score
                        best_move = move
                        
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            else:
                value = float('inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.opponent_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, True)
                    
                    if eval_score < value:
                        value = eval_score
                        best_move = move
                        
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

            # Update transposition table
            self.transposition_table[state_hash] = (depth, value, best_move)
            if len(self.transposition_table) > self.max_table_size:
                self.transposition_table.clear()

            return value, best_move

    def make_move(self, state):
        valid_moves = get_valid_col_id(state)
        
        # Handle single move case
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Opening book with index calculation
        depth = int((np.count_nonzero(state) + 1 - self.player_id) / 2)
        if depth < self.OPTIMAL_LIMIT:
            ind = 0
            for i in range(depth):
                ind += 7 ** i
            if ind < len(self.OPTIMAL_MOVES):
                book_move = self.OPTIMAL_MOVES[ind]
                if book_move in valid_moves:  # Validate book move
                    return book_move
        
        # Quick win/block check
        for move in valid_moves:
            # Check winning move
            new_state = step(state, move, self.player_id, False)
            if is_win(new_state):
                return move
                
        for move in valid_moves:
            # Check blocking move
            new_state = step(state, move, self.opponent_id, False)
            if is_win(new_state):
                return move
        
        # Regular search with validation
        _, best_move = self.alpha_beta_pruning(state, self.MAX_DEPTH, float('-inf'), float('inf'), True)
        
        if best_move is not None and best_move in valid_moves:
            return best_move
            
        # Fallback to center-focused strategy
        return min(valid_moves, key=lambda x: abs(x-3))


class AIAgentAttack(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id

        # Weights initialization
        self.weights = {
            'horizontal_immediate_threat_score': 60.783,
            'horizontal_potential_threat_score': 21.172,
            'vertical_immediate_threat_score': 95.224,
            'vertical_potential_threat_score': 30.180,
            'diagonal_immediate_threat_score': 129.034,
            'diagonal_potential_threat_score': 30.928,
            'aggressive_defensive_balance_score': 0.776,
            'fork_score': 120.399,
            'vertical_bridge_score': 10.000,
            'horizontal_bridge_score': 42.814,
            'center_control_score': 94.523,
            'height_penalty':  5.792,
            'mobility_bonus': 27.395
        }

        if self.player_id == 1:
            self.OPTIMAL_LIMIT = 2
            self.OPTIMAL_MOVES = [3, # depth 0 
                                  3, 5, 5, 3, 1, 1, 3, # depth 1
                                  3, 3, 3, 3, 3, 3, 3, # depth 2 col 0
                                  4, 4, 3, 4, 5, 4, 3, # depth 2 col 1
                                  6, 3, 6, 3, 4, 6, 5, # depth 2 col 2
                                  4, 3, 3, 3, 3, 3, 2, # depth 2 col 3
                                  1, 0, 2, 3, 0, 3, 0, # depth 2 col 4
                                  1, 2, 1, 2, 3, 2, 2, # depth 2 col 5
                                  3, 3, 3, 3, 3, 3, 3] # depth 2 col 6
        else:
            self.OPTIMAL_LIMIT = 1
            self.OPTIMAL_MOVES = [3, 2, 3, 3, 3, 4, 3,
                                  3, 3, 3, 3, 3, 3, 2,
                                  2, 1, 2, 3, 2, 2, 2,
                                  3, 3, 2, 3, 3, 3, 3,
                                  3, 2, 4, 3, 2, 4, 3,
                                  3, 3, 3, 3, 4, 3, 3,
                                  4, 4, 4, 3, 4, 5, 4,
                                  4, 3, 3, 3, 3, 3, 2]

        self.MAX_DEPTH = 4
        self.MAX_VALUE = 999999
        
        # Transposition table for memoization
        self.transposition_table = {}
        self.max_table_size = 50000
        self.clear_threshold = 40000

    def evaluate_position(self, state):
        """Enhanced position evaluation with advanced threat detection and custom weights."""
        if is_win(state):
            return self.MAX_VALUE
            
        score = 0

        def count_threats(player):
            threat_score = 0
            immediate_threats = 0
            potential_threats = 0

            # Horizontal threats
            for row in range(6):
                for col in range(4):
                    window = state[row, col:col+4]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['horizontal_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['horizontal_potential_threat_score']
            
            # Vertical threats
            for row in range(3):
                for col in range(7):
                    window = state[row:row+4, col]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['vertical_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['vertical_potential_threat_score']
            
            # Diagonal threats
            for row in range(3):
                for col in range(4):
                    window_positive = [state[row+i][col+i] for i in range(4)]
                    player_count_positive = sum(1 for x in window_positive if x == player)
                    empty_count_positive = sum(1 for x in window_positive if x == 0)
                    
                    if player_count_positive == 3 and empty_count_positive == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_positive == 2 and empty_count_positive == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
                    
                    window_negative = [state[row+3-i][col+i] for i in range(4)]
                    player_count_negative = sum(1 for x in window_negative if x == player)
                    empty_count_negative = sum(1 for x in window_negative if x == 0)
                    
                    if player_count_negative == 3 and empty_count_negative == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_negative == 2 and empty_count_negative == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
            
            return threat_score, immediate_threats, potential_threats

        player_threats, player_immediate, player_potential = count_threats(self.player_id)
        opponent_threats, opponent_immediate, opponent_potential = count_threats(self.opponent_id)

        # Aggressive-defensive balance
        score += player_threats - opponent_threats * self.weights['aggressive_defensive_balance_score']
        
        # Fork detection
        if player_immediate > 1:
            score += self.weights['fork_score']
        if opponent_immediate > 1:
            score -= self.weights['fork_score']

        # Vertical bridge detection
        for col in range(7):
            if np.count_nonzero(state[:, col] == self.player_id) == 2:
                empty_below = np.count_nonzero(state[5:, col] == 0)
                if empty_below > 0:
                    score += self.weights['vertical_bridge_score']

        # Horizontal bridge detection
        for row in range(6):
            for col in range(5):
                if state[row, col] == self.player_id and state[row, col + 1] == self.player_id:
                    if (col - 1 >= 0 and state[row, col - 1] == 0) or (col + 2 < 7 and state[row, col + 2] == 0):
                        score += self.weights['horizontal_bridge_score']
        
        # Center control bonus
        center_array = state[:, 3]
        center_score = np.count_nonzero(center_array == self.player_id) * self.weights['center_control_score']
        score += center_score

        # Height penalty to prefer lower positions
        for col in range(7):
            col_height = np.count_nonzero(state[:, col])
            if col_height > 0:
                score -= col_height * self.weights['height_penalty']
        
        # Mobility bonus
        valid_moves = get_valid_col_id(state)
        score += len(valid_moves) * self.weights['mobility_bonus']
        
        return score


    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
            """Fixed alpha-beta pruning with your current features"""
            # Transposition table lookup
            state_hash = hash(state.tobytes())
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    return stored_value, stored_move

            # Base cases with proper checks
            if is_win(state):
                return (-self.MAX_VALUE if maximizing_player else self.MAX_VALUE), None
                
            if depth == 0:
                return self.evaluate_position(state), None

            valid_moves = get_valid_col_id(state)
            
            # Handle edge cases properly
            if len(valid_moves) == 0:
                return 0, None
                
            if len(valid_moves) == 1:
                return 0, valid_moves[0]
            
            # Main search logic
            if maximizing_player:
                value = float('-inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.player_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, False)
                    
                    if eval_score > value:
                        value = eval_score
                        best_move = move
                        
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            else:
                value = float('inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.opponent_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, True)
                    
                    if eval_score < value:
                        value = eval_score
                        best_move = move
                        
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

            # Update transposition table
            self.transposition_table[state_hash] = (depth, value, best_move)
            if len(self.transposition_table) > self.max_table_size:
                self.transposition_table.clear()

            return value, best_move

    def make_move(self, state):
        """Fixed move selection with your current features"""
        valid_moves = get_valid_col_id(state)
        
        # Handle single move case
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Opening book with proper index calculation
        depth = int((np.count_nonzero(state) + 1 - self.player_id) / 2)
        if depth < self.OPTIMAL_LIMIT:
            ind = 0
            for i in range(depth):
                ind += 7 ** i
            if ind < len(self.OPTIMAL_MOVES):
                book_move = self.OPTIMAL_MOVES[ind]
                if book_move in valid_moves:  # Validate book move
                    return book_move
        
        # Quick win/block check
        for move in valid_moves:
            # Check winning move
            new_state = step(state, move, self.player_id, False)
            if is_win(new_state):
                return move
                
        for move in valid_moves:
            # Check blocking move
            new_state = step(state, move, self.opponent_id, False)
            if is_win(new_state):
                return move
        
        # Regular search with validation
        _, best_move = self.alpha_beta_pruning(state, self.MAX_DEPTH, float('-inf'), float('inf'), True)
        
        if best_move is not None and best_move in valid_moves:
            return best_move
            
        # Fallback to center-focused strategy
        return min(valid_moves, key=lambda x: abs(x-3))
    

class AIAgentDef(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id

        # Weights initialization
        self.weights = {
            'horizontal_immediate_threat_score': 127.861,
            'horizontal_potential_threat_score': 18.919,
            'vertical_immediate_threat_score': 128.404,
            'vertical_potential_threat_score': 7.347,
            'diagonal_immediate_threat_score': 84.072,
            'diagonal_potential_threat_score': 24.853,
            'aggressive_defensive_balance_score': 1.280,
            'fork_score': 318.984,
            'vertical_bridge_score': 61.454,
            'horizontal_bridge_score': 28.192,
            'center_control_score': 70.161,
            'height_penalty':  1.347,
            'mobility_bonus': 26.920

        }

        if self.player_id == 1:
            self.OPTIMAL_LIMIT = 2
            self.OPTIMAL_MOVES = [3, # depth 0 
                                  3, 5, 5, 3, 1, 1, 3, # depth 1
                                  3, 3, 3, 3, 3, 3, 3, # depth 2 col 0
                                  4, 4, 3, 4, 5, 4, 3, # depth 2 col 1
                                  6, 3, 6, 3, 4, 6, 5, # depth 2 col 2
                                  4, 3, 3, 3, 3, 3, 2, # depth 2 col 3
                                  1, 0, 2, 3, 0, 3, 0, # depth 2 col 4
                                  1, 2, 1, 2, 3, 2, 2, # depth 2 col 5
                                  3, 3, 3, 3, 3, 3, 3] # depth 2 col 6
        else:
            self.OPTIMAL_LIMIT = 1
            self.OPTIMAL_MOVES = [3, 2, 3, 3, 3, 4, 3,
                                  3, 3, 3, 3, 3, 3, 2,
                                  2, 1, 2, 3, 2, 2, 2,
                                  3, 3, 2, 3, 3, 3, 3,
                                  3, 2, 4, 3, 2, 4, 3,
                                  3, 3, 3, 3, 4, 3, 3,
                                  4, 4, 4, 3, 4, 5, 4,
                                  4, 3, 3, 3, 3, 3, 2]

        self.MAX_DEPTH = 4
        self.MAX_VALUE = 999999
        
        # Transposition table for memoization
        self.transposition_table = {}
        self.max_table_size = 50000
        self.clear_threshold = 40000

    def evaluate_position(self, state):
        """Enhanced position evaluation with advanced threat detection and custom weights."""
        if is_win(state):
            return self.MAX_VALUE
            
        score = 0

        def count_threats(player):
            threat_score = 0
            immediate_threats = 0
            potential_threats = 0

            # Horizontal threats
            for row in range(6):
                for col in range(4):
                    window = state[row, col:col+4]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['horizontal_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['horizontal_potential_threat_score']
            
            # Vertical threats
            for row in range(3):
                for col in range(7):
                    window = state[row:row+4, col]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['vertical_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['vertical_potential_threat_score']
            
            # Diagonal threats
            for row in range(3):
                for col in range(4):
                    window_positive = [state[row+i][col+i] for i in range(4)]
                    player_count_positive = sum(1 for x in window_positive if x == player)
                    empty_count_positive = sum(1 for x in window_positive if x == 0)
                    
                    if player_count_positive == 3 and empty_count_positive == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_positive == 2 and empty_count_positive == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
                    
                    window_negative = [state[row+3-i][col+i] for i in range(4)]
                    player_count_negative = sum(1 for x in window_negative if x == player)
                    empty_count_negative = sum(1 for x in window_negative if x == 0)
                    
                    if player_count_negative == 3 and empty_count_negative == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_negative == 2 and empty_count_negative == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
            
            return threat_score, immediate_threats, potential_threats

        player_threats, player_immediate, player_potential = count_threats(self.player_id)
        opponent_threats, opponent_immediate, opponent_potential = count_threats(self.opponent_id)

        # Aggressive-defensive balance
        score += player_threats - opponent_threats * self.weights['aggressive_defensive_balance_score']
        
        # Fork detection
        if player_immediate > 1:
            score += self.weights['fork_score']
        if opponent_immediate > 1:
            score -= self.weights['fork_score']

        # Vertical bridge detection
        for col in range(7):
            if np.count_nonzero(state[:, col] == self.player_id) == 2:
                empty_below = np.count_nonzero(state[5:, col] == 0)
                if empty_below > 0:
                    score += self.weights['vertical_bridge_score']

        # Horizontal bridge detection
        for row in range(6):
            for col in range(5):
                if state[row, col] == self.player_id and state[row, col + 1] == self.player_id:
                    if (col - 1 >= 0 and state[row, col - 1] == 0) or (col + 2 < 7 and state[row, col + 2] == 0):
                        score += self.weights['horizontal_bridge_score']
        
        # Center control bonus
        center_array = state[:, 3]
        center_score = np.count_nonzero(center_array == self.player_id) * self.weights['center_control_score']
        score += center_score

        # Height penalty to prefer lower positions
        for col in range(7):
            col_height = np.count_nonzero(state[:, col])
            if col_height > 0:
                score -= col_height * self.weights['height_penalty']
        
        # Mobility bonus
        valid_moves = get_valid_col_id(state)
        score += len(valid_moves) * self.weights['mobility_bonus']
        
        return score


    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
            """Fixed alpha-beta pruning with your current features"""
            # Transposition table lookup
            state_hash = hash(state.tobytes())
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    return stored_value, stored_move

            # Base cases with proper checks
            if is_win(state):
                return (-self.MAX_VALUE if maximizing_player else self.MAX_VALUE), None
                
            if depth == 0:
                return self.evaluate_position(state), None

            valid_moves = get_valid_col_id(state)
            
            # Handle edge cases properly
            if len(valid_moves) == 0:
                return 0, None
                
            if len(valid_moves) == 1:
                return 0, valid_moves[0]
            
            # Main search logic
            if maximizing_player:
                value = float('-inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.player_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, False)
                    
                    if eval_score > value:
                        value = eval_score
                        best_move = move
                        
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            else:
                value = float('inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.opponent_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, True)
                    
                    if eval_score < value:
                        value = eval_score
                        best_move = move
                        
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

            # Update transposition table
            self.transposition_table[state_hash] = (depth, value, best_move)
            if len(self.transposition_table) > self.max_table_size:
                self.transposition_table.clear()

            return value, best_move

    def make_move(self, state):
        """Fixed move selection with your current features"""
        valid_moves = get_valid_col_id(state)
        
        # Handle single move case
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Opening book with proper index calculation
        depth = int((np.count_nonzero(state) + 1 - self.player_id) / 2)
        if depth < self.OPTIMAL_LIMIT:
            ind = 0
            for i in range(depth):
                ind += 7 ** i
            if ind < len(self.OPTIMAL_MOVES):
                book_move = self.OPTIMAL_MOVES[ind]
                if book_move in valid_moves:  # Validate book move
                    return book_move
        
        # Quick win/block check
        for move in valid_moves:
            # Check winning move
            new_state = step(state, move, self.player_id, False)
            if is_win(new_state):
                return move
                
        for move in valid_moves:
            # Check blocking move
            new_state = step(state, move, self.opponent_id, False)
            if is_win(new_state):
                return move
        
        # Regular search with validation
        _, best_move = self.alpha_beta_pruning(state, self.MAX_DEPTH, float('-inf'), float('inf'), True)
        
        if best_move is not None and best_move in valid_moves:
            return best_move
            
        # Fallback to center-focused strategy
        return min(valid_moves, key=lambda x: abs(x-3))
    
from game_utils import initialize, step, get_valid_col_id, is_win
import numpy as np

class AIAgentBalance(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id

        # Weights initialization
        self.weights = {
            'horizontal_immediate_threat_score': 120.0,  # Slightly reduced to balance with other factors
            'horizontal_potential_threat_score': 30.0,   # Increased to enhance setup potential
            'vertical_immediate_threat_score': 140.0,    # Slightly reduced, still prioritizing critical vertical blocking
            'vertical_potential_threat_score': 40.0,     # Increased for more proactive vertical setups
            'diagonal_immediate_threat_score': 135.0,    # Balanced for diagonal threats
            'diagonal_potential_threat_score': 40.0,     # Increased to encourage diagonal setups
            'aggressive_defensive_balance_score': 1.4,   # Balanced, slightly defensive
            'fork_score': 325.0,                         # Retained high value for double threats
            'vertical_bridge_score': 30.0,               # Balanced vertical bridge for structural support
            'horizontal_bridge_score': 35.0,             # Balanced horizontal bridge for lateral strategies
            'center_control_score': 75.0,                # Slightly reduced for adaptable play
            'height_penalty': 2.5,                       # Kept moderate to avoid excessive stacking
            'mobility_bonus': 35.0                       # Slightly reduced to focus on board control
        }

        if self.player_id == 1:
            self.OPTIMAL_LIMIT = 2
            self.OPTIMAL_MOVES = [3, # depth 0 
                                  3, 5, 5, 3, 1, 1, 3, # depth 1
                                  3, 3, 3, 3, 3, 3, 3, # depth 2 col 0
                                  4, 4, 3, 4, 5, 4, 3, # depth 2 col 1
                                  6, 3, 6, 3, 4, 6, 5, # depth 2 col 2
                                  4, 3, 3, 3, 3, 3, 2, # depth 2 col 3
                                  1, 0, 2, 3, 0, 3, 0, # depth 2 col 4
                                  1, 2, 1, 2, 3, 2, 2, # depth 2 col 5
                                  3, 3, 3, 3, 3, 3, 3] # depth 2 col 6
        else:
            self.OPTIMAL_LIMIT = 1
            self.OPTIMAL_MOVES = [3, 2, 3, 3, 3, 4, 3,
                                  3, 3, 3, 3, 3, 3, 2,
                                  2, 1, 2, 3, 2, 2, 2,
                                  3, 3, 2, 3, 3, 3, 3,
                                  3, 2, 4, 3, 2, 4, 3,
                                  3, 3, 3, 3, 4, 3, 3,
                                  4, 4, 4, 3, 4, 5, 4,
                                  4, 3, 3, 3, 3, 3, 2]

        self.MAX_DEPTH = 4
        self.MAX_VALUE = 999999
        
        # Transposition table for memoization
        self.transposition_table = {}
        self.max_table_size = 50000
        self.clear_threshold = 40000

    def evaluate_position(self, state):
        """Enhanced position evaluation with advanced threat detection, two-step threat recognition, and positional advantage."""
        if is_win(state):
            return self.MAX_VALUE
            
        score = 0

        def count_threats(player):
            threat_score = 0
            immediate_threats = 0
            potential_threats = 0
            two_step_threats = 0

            # Horizontal threats
            for row in range(6):
                for col in range(4):
                    window = state[row, col:col+4]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['horizontal_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['horizontal_potential_threat_score']
                    elif player_count == 1 and empty_count == 3:
                        two_step_threats += 1
                        threat_score += self.weights['horizontal_potential_threat_score'] / 2  # Lower weight for two-step threats

            # Vertical threats
            for row in range(3):
                for col in range(7):
                    window = state[row:row+4, col]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['vertical_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['vertical_potential_threat_score']
                    elif player_count == 1 and empty_count == 3:
                        two_step_threats += 1
                        threat_score += self.weights['vertical_potential_threat_score'] / 2  # Two-step threat weight

            # Diagonal threats
            for row in range(3):
                for col in range(4):
                    # Positive slope
                    window_positive = [state[row+i][col+i] for i in range(4)]
                    player_count_positive = sum(1 for x in window_positive if x == player)
                    empty_count_positive = sum(1 for x in window_positive if x == 0)
                    
                    if player_count_positive == 3 and empty_count_positive == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_positive == 2 and empty_count_positive == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
                    elif player_count_positive == 1 and empty_count_positive == 3:
                        two_step_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score'] / 2

                    # Negative slope
                    window_negative = [state[row+3-i][col+i] for i in range(4)]
                    player_count_negative = sum(1 for x in window_negative if x == player)
                    empty_count_negative = sum(1 for x in window_negative if x == 0)
                    
                    if player_count_negative == 3 and empty_count_negative == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_negative == 2 and empty_count_negative == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
                    elif player_count_negative == 1 and empty_count_negative == 3:
                        two_step_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score'] / 2
                
            return threat_score, immediate_threats, potential_threats, two_step_threats

        # Calculate threats for both players
        player_threats, player_immediate, player_potential, player_two_step = count_threats(self.player_id)
        opponent_threats, opponent_immediate, opponent_potential, opponent_two_step = count_threats(self.opponent_id)

        # Aggressive-defensive balance
        score += player_threats - opponent_threats * self.weights['aggressive_defensive_balance_score']
        
        # Fork detection
        if player_immediate > 1:
            score += self.weights['fork_score']
        if opponent_immediate > 1:
            score -= self.weights['fork_score']

        # Vertical bridge detection
        for col in range(7):
            if np.count_nonzero(state[:, col] == self.player_id) == 2:
                empty_below = np.count_nonzero(state[5:, col] == 0)
                if empty_below > 0:
                    score += self.weights['vertical_bridge_score']

        # Horizontal bridge detection
        for row in range(6):
            for col in range(5):
                if state[row, col] == self.player_id and state[row, col + 1] == self.player_id:
                    if (col - 1 >= 0 and state[row, col - 1] == 0) or (col + 2 < 7 and state[row, col + 2] == 0):
                        score += self.weights['horizontal_bridge_score']
        
        # Center control bonus
        center_array = state[:, 3]
        center_score = np.count_nonzero(center_array == self.player_id) * self.weights['center_control_score']
        score += center_score

        # Height penalty to prefer lower positions
        for col in range(7):
            col_height = np.count_nonzero(state[:, col])
            if col_height > 0:
                score -= col_height * self.weights['height_penalty']
        
        # Mobility bonus
        valid_moves = get_valid_col_id(state)
        score += len(valid_moves) * self.weights['mobility_bonus']
        
        # Positional advantage: reward for creating two-step threats
        score += player_two_step * (self.weights['horizontal_potential_threat_score'] + self.weights['vertical_potential_threat_score']) / 2
        score -= opponent_two_step * (self.weights['horizontal_potential_threat_score'] + self.weights['vertical_potential_threat_score']) / 2
        
        return score



    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
            """Fixed alpha-beta pruning with your current features"""
            # Transposition table lookup
            state_hash = hash(state.tobytes())
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    return stored_value, stored_move

            # Base cases with proper checks
            if is_win(state):
                return (-self.MAX_VALUE if maximizing_player else self.MAX_VALUE), None
                
            if depth == 0:
                return self.evaluate_position(state), None

            valid_moves = get_valid_col_id(state)
            
            # Handle edge cases properly
            if len(valid_moves) == 0:
                return 0, None
                
            if len(valid_moves) == 1:
                return 0, valid_moves[0]
            
            # Main search logic
            if maximizing_player:
                value = float('-inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.player_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, False)
                    
                    if eval_score > value:
                        value = eval_score
                        best_move = move
                        
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            else:
                value = float('inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.opponent_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, True)
                    
                    if eval_score < value:
                        value = eval_score
                        best_move = move
                        
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

            # Update transposition table
            self.transposition_table[state_hash] = (depth, value, best_move)
            if len(self.transposition_table) > self.max_table_size:
                self.transposition_table.clear()

            return value, best_move

    def make_move(self, state):
        """Improved move selection with optimized ordering and quick checks."""
        valid_moves = get_valid_col_id(state)
        
        # Prioritize single moves with win/block checks
        for move in valid_moves:
            new_state = step(state, move, self.player_id, False)
            if is_win(new_state):
                return move
        for move in valid_moves:
            new_state = step(state, move, self.opponent_id, False)
            if is_win(new_state):
                return move
        
        # Use the opening book if within depth limit
        depth = int((np.count_nonzero(state) + 1 - self.player_id) / 2)
        if depth < self.OPTIMAL_LIMIT:
            ind = 0
            for i in range(depth):
                ind += 7 ** i
            if ind < len(self.OPTIMAL_MOVES):
                book_move = self.OPTIMAL_MOVES[ind]
                if book_move in valid_moves:
                    return book_move
        
        # Pre-sort moves by heuristic value for alpha-beta efficiency
        sorted_moves = sorted(valid_moves, key=lambda move: -self.evaluate_position(step(state, move, self.player_id, False)))
        _, best_move = self.alpha_beta_pruning(state, self.MAX_DEPTH, float('-inf'), float('inf'), True)
        
        return best_move if best_move in valid_moves else min(valid_moves, key=lambda x: abs(x - 3))



from game_utils import initialize, step, get_valid_col_id, is_win
import numpy as np

class AIAgent(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id

        if self.player_id == 1:
            self.OPTIMAL_LIMIT = 2
            self.OPTIMAL_MOVES = [3, # depth 0 
                                  3, 5, 5, 3, 1, 1, 3, # depth 1
                                  3, 3, 3, 3, 3, 3, 3, # depth 2 col 0
                                  4, 4, 3, 4, 5, 4, 3, # depth 2 col 1
                                  6, 3, 6, 3, 4, 6, 5, # depth 2 col 2
                                  4, 3, 3, 3, 3, 3, 2, # depth 2 col 3
                                  1, 0, 2, 3, 0, 3, 0, # depth 2 col 4
                                  1, 2, 1, 2, 3, 2, 2, # depth 2 col 5
                                  3, 3, 3, 3, 3, 3, 3] # depth 2 col 6
            # Going First Weights
            self.weights = {
                'horizontal_immediate_threat_score': 130.0,   # Slightly lower for first move, allows offensive focus elsewhere
                'horizontal_potential_threat_score': 35.0,    # Encourages setup for potential horizontal threats
                'vertical_immediate_threat_score': 150.0,     # Strong blocking for immediate vertical threats
                'vertical_potential_threat_score': 40.0,      # Balanced, encourages building vertical structures for offense
                'diagonal_immediate_threat_score': 155.0,     # Higher for catching critical diagonal moves
                'diagonal_potential_threat_score': 45.0,      # Encourages diagonal setups, which are powerful offensively
                'aggressive_defensive_balance_score': 1.3,    # Slight offensive skew for going first, to dominate the board early
                'fork_score': 375.0,                          # High fork score for creating double threats, essential for offense
                'vertical_bridge_score': 30.0,                # Balanced vertical bridge for structural flexibility
                'horizontal_bridge_score': 45.0,              # Slightly higher for horizontal bridge to create lateral threats
                'center_control_score': 85.0,                 # Emphasizes early control of the center for board dominance
                'height_penalty': 2.0,                        # Lower penalty allows more freedom for offensive stacking
                'mobility_bonus': 35.0                        # Rewards open options to maintain flexibility in strategy
            }
        else:
            self.OPTIMAL_LIMIT = 1
            self.OPTIMAL_MOVES = [3, 2, 3, 3, 3, 4, 3,
                                  3, 3, 3, 3, 3, 3, 2,
                                  2, 1, 2, 3, 2, 2, 2,
                                  3, 3, 2, 3, 3, 3, 3,
                                  3, 2, 4, 3, 2, 4, 3,
                                  3, 3, 3, 3, 4, 3, 3,
                                  4, 4, 4, 3, 4, 5, 4,
                                  4, 3, 3, 3, 3, 3, 2]
            # Going Second Weights
            self.weights = {
                'horizontal_immediate_threat_score': 150.0,   # High to quickly block any immediate horizontal threats
                'horizontal_potential_threat_score': 25.0,    # Moderate to maintain awareness of potential horizontal threats
                'vertical_immediate_threat_score': 170.0,     # Strong focus on blocking vertical threats, which are quick to develop
                'vertical_potential_threat_score': 30.0,      # Slightly lower to focus on immediate responses
                'diagonal_immediate_threat_score': 160.0,     # Strong diagonal blocking, as diagonals are tricky to defend against
                'diagonal_potential_threat_score': 35.0,      # Balanced to maintain an awareness of diagonal setups
                'aggressive_defensive_balance_score': 1.6,    # Higher defensive skew to focus on reacting to the opponent's moves
                'fork_score': 300.0,                          # Still prioritizes fork prevention, essential when going second
                'vertical_bridge_score': 35.0,                # Emphasis on defensive structures for blocking
                'horizontal_bridge_score': 30.0,              # Lower horizontal bridge as defense is prioritized over lateral setup
                'center_control_score': 70.0,                 # Center control is still valuable but less critical than immediate threats
                'height_penalty': 2.75,                       # Higher penalty to avoid stacking too early without good setup
                'mobility_bonus': 40.0                        # Rewards flexibility, essential for counterplay
            }

        self.MAX_DEPTH = 4
        self.MAX_VALUE = 999999
        
        # Transposition table for memoization
        self.transposition_table = {}
        self.max_table_size = 50000
        self.clear_threshold = 40000

    def evaluate_position(self, state):
        if is_win(state):
            return self.MAX_VALUE
            
        score = 0

        def count_threats(player):
            threat_score = 0
            immediate_threats = 0
            potential_threats = 0

            # Horizontal threats
            for row in range(6):
                for col in range(4):
                    window = state[row, col:col+4]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['horizontal_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['horizontal_potential_threat_score']
            
            # Vertical threats
            for row in range(3):
                for col in range(7):
                    window = state[row:row+4, col]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['vertical_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['vertical_potential_threat_score']
            
            # Diagonal threats
            for row in range(3):
                for col in range(4):
                    window_positive = [state[row+i][col+i] for i in range(4)]
                    player_count_positive = sum(1 for x in window_positive if x == player)
                    empty_count_positive = sum(1 for x in window_positive if x == 0)
                    
                    if player_count_positive == 3 and empty_count_positive == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_positive == 2 and empty_count_positive == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
                    
                    window_negative = [state[row+3-i][col+i] for i in range(4)]
                    player_count_negative = sum(1 for x in window_negative if x == player)
                    empty_count_negative = sum(1 for x in window_negative if x == 0)
                    
                    if player_count_negative == 3 and empty_count_negative == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_negative == 2 and empty_count_negative == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
            
            return threat_score, immediate_threats, potential_threats

        player_threats, player_immediate, _ = count_threats(self.player_id)
        opponent_threats, opponent_immediate, _ = count_threats(self.opponent_id)

        # Aggressive-defensive balance
        score += player_threats - opponent_threats * self.weights['aggressive_defensive_balance_score']
        
        # Fork detection
        if player_immediate > 1:
            score += self.weights['fork_score']
        if opponent_immediate > 1:
            score -= self.weights['fork_score']

        # Vertical bridge detection
        for col in range(7):
            if np.count_nonzero(state[:, col] == self.player_id) == 2:
                empty_below = np.count_nonzero(state[5:, col] == 0)
                if empty_below > 0:
                    score += self.weights['vertical_bridge_score']

        # Horizontal bridge detection
        for row in range(6):
            for col in range(5):
                if state[row, col] == self.player_id and state[row, col + 1] == self.player_id:
                    if (col - 1 >= 0 and state[row, col - 1] == 0) or (col + 2 < 7 and state[row, col + 2] == 0):
                        score += self.weights['horizontal_bridge_score']
        
        # Center control bonus
        center_array = state[:, 3]
        center_score = np.count_nonzero(center_array == self.player_id) * self.weights['center_control_score']
        score += center_score

        # Height penalty to prefer lower positions
        for col in range(7):
            col_height = np.count_nonzero(state[:, col])
            if col_height > 0:
                score -= col_height * self.weights['height_penalty']
        
        # Mobility bonus
        valid_moves = get_valid_col_id(state)
        score += len(valid_moves) * self.weights['mobility_bonus']
        
        return score


    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
            # Transposition table lookup
            state_hash = hash(state.tobytes())
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    return stored_value, stored_move

            # Base cases with checks
            if is_win(state):
                return (-self.MAX_VALUE if maximizing_player else self.MAX_VALUE), None
                
            if depth == 0:
                return self.evaluate_position(state), None

            valid_moves = get_valid_col_id(state)
            
            # Handle edge cases
            if len(valid_moves) == 0:
                return 0, None
                
            if len(valid_moves) == 1:
                return 0, valid_moves[0]
            
            # Main search logic
            if maximizing_player:
                value = float('-inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.player_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, False)
                    
                    if eval_score > value:
                        value = eval_score
                        best_move = move
                        
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            else:
                value = float('inf')
                best_move = valid_moves[0]
                
                for move in valid_moves:
                    new_state = step(state, move, self.opponent_id, False)
                    eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, True)
                    
                    if eval_score < value:
                        value = eval_score
                        best_move = move
                        
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

            # Update transposition table
            self.transposition_table[state_hash] = (depth, value, best_move)
            if len(self.transposition_table) > self.max_table_size:
                self.transposition_table.clear()

            return value, best_move

    def make_move(self, state):
        valid_moves = get_valid_col_id(state)
        
        # Handle single move case
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Opening book with index calculation
        depth = int((np.count_nonzero(state) + 1 - self.player_id) / 2)
        if depth < self.OPTIMAL_LIMIT:
            ind = 0
            for i in range(depth):
                ind += 7 ** i
            if ind < len(self.OPTIMAL_MOVES):
                book_move = self.OPTIMAL_MOVES[ind]
                if book_move in valid_moves:  # Validate book move
                    return book_move
        
        # Quick win/block check
        for move in valid_moves:
            # Check winning move
            new_state = step(state, move, self.player_id, False)
            if is_win(new_state):
                return move
                
        for move in valid_moves:
            # Check blocking move
            new_state = step(state, move, self.opponent_id, False)
            if is_win(new_state):
                return move
        
        # Regular search with validation
        _, best_move = self.alpha_beta_pruning(state, self.MAX_DEPTH, float('-inf'), float('inf'), True)
        
        if best_move is not None and best_move in valid_moves:
            return best_move
            
        # Fallback to center-focused strategy
        return min(valid_moves, key=lambda x: abs(x-3))

from itertools import product
class AIAgentGab(object):
    """
    A class representing an agent that plays Connect Four.
    """
    def __init__(self, player_id=1):
        """Initializes the agent with the specified player ID.

        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        """
        self.player_id = player_id
        self.OPTIMAL_MOVES = [[3, 3, 1, 5, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 2, 1, 3, 0, 1, 3, 3, 3, 2, 3, 2, 3, 4, 6, 5, 4, 3, 2, 3, 3, 3, 2, 1, 0, 2, 3, 0, 3, 0, 3, 2, 1, 2, 3, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 4, 3, 3, 4, 3, 1, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 4, 2, 4, 4, 3, 2, 4, 1, 4, 1, 1, 1, 1, 1, 4, 3, 1, 4, 1, 1, 2, 3, 3, 2, 1, 2, 3, 3, 1, 3, 0, 1, 1, 1, 0, 3, 1, 1, 3, 3, 5, 0, 1, 1, 5, 1, 5, 0, 0, 1, 2, 3, 1, 0, 0, 4, 6, 3, 6, 3, 4, 2, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 3, 3, 4, 4, 5, 2, 3, 5, 6, 3, 2, 2, 2, 3, 4, 3, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 1, 2, 3, 3, 3, 5, 2, 5, 3, 2, 2, 3, 2, 0, 3, 2, 1, 1, 1, 1, 3, 3, 3, 2, 1, 3, 2, 2, 1, 2, -1, 1, 5, 2, 1, 1, 1, 4, 0, 3, 3, 0, 1, 0, 5, 3, 0, 0, 4, 3, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 4, 3, 2, 3, 4, 1, 0, 2, 0, 1, 3, 0, 2, 2, 1, 2, 2, 2, 2, 3, 3, 2, 4, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 3, 2, 2, 4, 0, 0, 0, 0, 0, 0, 1, 3, 1, 1, 1, 3, 3, 4, 0, 0, 0, 0, 0, 0, 3, 3, 2, 4, 2, 2, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 2, 4, 4, 3, 2, 4, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 2, 1, 1, 2, 3, 3, 3, 0, 3, 3, 3, 2, 3, 1, 1, 2, 1, 1, 4, 4, 1, 1, 0, 1, 6, -1, 0, 0, -1, 0, 0, 0, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 4, 3, 3, 4, 3, 1, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 4, 2, 4, 4, 3, 2, 4, 1, 4, 1, 1, 1, 1, 1, 4, 3, 1, 4, 1, 1, 2, 3, 3, 2, 1, 2, 3, 3, 1, 3, 0, 1, 1, 1, 0, 3, 1, 1, 3, 3, 5, 0, 1, 1, 5, 1, 5, 0, 0, 1, 2, 3, 1, 0, 0, 4, 6, 3, 6, 3, 4, 2, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 3, 3, 4, 4, 5, 2, 3, 5, 6, 3, 2, 2, 2, 3, 4, 3, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 1, 2, 3, 3, 3, 5, 2, 5, 3, 2, 2, 3, 2, 0, 3, 2, 1, 1, 1, 1, 3, 3, 3, 2, 1, 3, 2, 2, 1, 2, 2, 1, 5, 2, 1, 1, 1, 4, 0, 3, 3, 0, 1, 0, 5, 3, 0, 0, 4, 3, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 4, 3, 2, 3, 4, 1, 0, 2, 0, 1, 3, 0, 2, 2, 1, 2, 2, 2, 2, 3, 3, 2, 4, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 3, 2, 2, 4, 0, 0, 0, 0, 0, 0, 1, 3, 1, 1, 1, 3, 3, 4, 0, 0, 0, 0, 0, 0, 3, 3, 2, 4, 2, 2, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 2, 4, 4, 3, 2, 4, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 2, 1, 1, 2, 3, 3, 3, 0, 3, 3, 3, 2, 3, 1, 1, 2, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 5, 5, 2, 5, 5, 5, 2, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 5, 5, 2, 5, 5, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 3, 4, 1, 1, 5, 5, 0, 5, 5, 4, 4, 0, 1, 4, 0, 3, 5, 2, 4, 1, 1, 3, 5, 4, 3, 2, 3, 2, 2, 3, 2, 4, 0, 1, 2, 3, 4, 0, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 3, 2, 1, 1, 1, 0, 3, 1, 1, 1, 1, 1, 4, 3, 3, 2, 0, 3, 3, 0, 1, 2, 1, 2, 0, 0, 1, 1, 2, 1, 2, 0, 0, 1, 3, 1, 1, 1, 1, 1, 4, 3, 3, 3, 3, 3, 3, 0, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 3, 3, 3, 1, 3, 3, 3, 0, 1, 4, 0, 3, 5, 2, 2, 2, 1, 2, 1, 2, 2, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 4, 3, 3, 3, 3, 1, 2, 3, 3, 4, 3, 1, 0, 2, 3, 3, 5, 0, 1, 3, 0, 1, 1, 0, 0, 4, 1, 1, 3, 5, 4, 3, 2, 2, 2, 3, 2, 1, 1, 1, 3, 0, 1, 1, 0, 0, 3, 1, 2, 3, 3, 4, 3, 1, 0, 0, 1, 0, 5, 0, 0, 0, 0, 1, 5, 0, 0, 0, 2, 0, 1, 1, 0, 0, 2, 3, 2, 2, 3, 2, 4, 3, 3, 3, 6, 3, 3, 3, 1, 3, 0, 1, 1, 0, 2, 1, 0, 2, 3, 3, 5, 0, 0, 0, 0, 1, 5, 0, 0, 2, 2, 3, 0, 2, 1, 6, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1, 1, 1, 1, 0, 4, 4, 4, 4, 4, 4, 4, 3, 6, 6, 2, 3, 6, 3, 3, 3, 3, 3, 3, 4, 3, 2, 1, 1, 2, 1, 2, 1, 3, 0, 0, 3, 3, 3, 3, 4, 3, 1, 1, 1, 1, 0, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 1, 3, 4, 4, 4, 1, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 2, 2, 5, 2, 2, 1, 2, 5, 4, 3, 2, 2, 0, 1, 3, 3, 3, 3, 3, 3, 3, 2, 4, 3, 4, 4, 6, 5, 2, 1, 4, 3, 2, 4, 4, 6, 6, 6, 6, 6, 6, 4, 2, 1, 2, 5, 2, 3, 2, 3, 6, 6, 2, 3, 6, 3, 3, 2, 1, 2, 2, 2, 1, 5, 1, 2, 5, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 2, 3, 2, 2, 2, 3, 6, 3, 4, 5, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 3, 2, 2, 3, 2, 2, 1, 2, 4, 4, 2, 3, 1, 2, 3, 2, 3, 4, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 1, 4, 3, 4, 3, 2, 4, 4, 5, 2, 3, 5, 5, 3, 3, -1, -1, 4, 4, -1, 4, 4, 3, 3, 3, 5, 6, 2, 4, 2, 2, 2, 3, 5, 3, 2, 2, 3, 3, 3, 6, 3, 4, 3, 3, 5, 3, 4, 2, 2, 2, 2, 3, 6, 6, 2, 4, 3, 3, 3, 3, 5, 3, 3, 4, 2, 2, 2, 3, 4, 2, 3, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 2, 3, 3, 1, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 2, 3, 4, 4, 4, 2, 3, 3, 2, 3, 3, 6, 0, 3, 2, 3, 2, 3, 2, 3, 3, 2, 1, 2, 2, 0, 2, 2, 3, 2, 0, 6, 0, 2, 1, 0, 0, 2, 5, 4, 2, 1, 4, 2, 2, 6, 5, 3, 2, 6, 2, 6, 2, 4, 2, 1, 3, 4, 5, 4, 2, 3, 3, 2, 1, 2, 2, 0, 3, 1, 2, 0, 1, 3, 1, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 3, 6, 5, 2, 3, 1, 1, 6, 2, 4, 1, 2, 1, 1, 5, 4, 2, 1, 2, 3, 1, 5, 4, 2, 4, 3, 2, 3, 5, 4, 4, 3, 2, 2, 2, 5, 4, 2, 0, 1, 2, 1, 5, 4, 1, 2, 3, 3, 3, 3, 1, 4, 2, 3, 2, 3, 3, 3, 4, 0, 3, 2, 1, 5, 4, 1, 0, 1, 0, 4, 5, 4, 0, 1, 1, 4, 1, 2, 6, 2, 0, 1, 2, 1, 5, 4, 1, 5, 1, 1, 5, 5, 4, 5, 2, 2, 2, 2, 4, 6, 5, 4, 4, 4, 4, 4, 5, 4, 0, 1, 1, 1, 5, 4, 1, 2, 1, 4, 2, 2, 6, 5, 2, 1, 2, 2, 3, 6, 5, 2, 3, 5, 3, 4, 6, 5, 2, 1, 4, 2, 4, 6, 5, 3, 3, 3, 2, 3, 6, 5, 5, 5, 2, 2, 4, 3, 3, 2, 3, 2, 2, 4, 3, 2, 2, 2, 1, 1, 1, 1, 0, 2, 3, 0, 2, 0, 3, 2, 1, 0, 2, 4, 3, 2, 2, 5, 5, 5, 4, 5, 4, 3, 1, 0, 3, 4, 3, 3, 3, 1, 3, 2, 6, 3, 5, 3, 0, 2, 2, 5, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 0, 1, 2, 5, 5, 4, 1, 0, 3, 2, 1, 5, 4, 1, 2, 0, 1, 0, 4, 5, 2, 0, 2, 1, 3, 3, 3, 3, 0, 2, 2, 5, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 0, 4, 2, 3, 1, 4, 4, 4, 5, 1, 3, 1, 5, 1, 4, 2, 4, 4, 2, 1, 4, 3, 3, 2, 3, 4, 1, 3, 4, 4, 1, 3, 2, 4, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 2, 3, 1, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 3, 4, 4, 4, 2, 5, 1, 3, 1, 3, 3, 3, 3, 5, 4, 3, 0, 1, 1, 6, 1, 3, 3, 3, 3, 4, 3, 4, 4, 3, 4, 4, 1, 5, 5, 2, 2, 6, 3, 6, 3, 4, 2, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 3, 3, 4, 4, 5, 2, 3, 5, 6, 3, 2, 2, 2, 3, 4, 3, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 1, 2, 3, 3, 3, 5, 2, 5, 3, 2, 2, 3, 2, 0, 3, 2, 1, 1, 1, 1, 3, 3, 3, 2, 1, 3, 2, 2, 1, 2, 2, 1, 5, 2, 1, 1, 1, 4, 0, 3, 3, 0, 1, 0, 5, 3, 0, 0, 4, 3, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 4, 3, 2, 3, 4, 1, 0, 2, 0, 1, 3, 0, 2, 2, 1, 2, 2, 2, 2, 3, 3, 2, 4, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 3, 2, 2, 4, 0, 0, 0, 0, 0, 0, 1, 3, 1, 1, 1, 3, 3, 4, 0, 0, 0, 0, 0, 0, 3, 3, 2, 4, 2, 2, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 2, 4, 4, 3, 2, 4, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 2, 1, 1, 2, 3, 3, 3, 0, 3, 3, 3, 2, 3, 1, 1, 2, 1, 1, 4, 4, 1, 1, 0, 1, 6, 3, 3, 3, 3, 3, 3, 3, 1, 4, 1, 3, 1, 5, 4, 1, 1, 2, 2, 2, 1, 0, 4, 5, 2, 1, 3, 3, 3, 1, 0, 1, 0, 0, 0, 0, 4, 0, 1, 1, 0, 5, 3, 3, 4, 2, 3, 1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 5, 1, 1, 1, 2, 3, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 1, 3, 4, 4, 3, 1, 4, 3, 3, 2, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1, 0, 1, 1, 5, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 1, 2, 1, 1, 1, 5, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 4, 4, 2, 4, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 4, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 2, 2, 4, 1, 2, 1, 2, 0, 0, 0, 0, 0, 2, 1, 2, 0, 4, 2, 2, 4, 2, 1, 1, 0, 1, 1, 4, 3, 4, 0, 1, 0, 0, 2, 2, 1, 2, 0, 2, 2, 4, 2, 1, 1, 0, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 0, 2, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 1, 1, 2, 1, 1, 3, 3, 3, 2, 3, 3, 3, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 5, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 4, 5, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 4, 0, 0, 0, 0, 0, 0, 4, 4, 1, 1, 4, 1, 1], [1, 2, 2, 3, 1, 4, 3, 0, 1, 2, 3, 5, 4, 1, 2, 1, 2, 3, 2, 2, 1, 3, 0, 3, 1, 3, 2, 2, 2, 2, 1, 3, 2, 4, 3, 5, 4, 1, 5, 4, 3, 4, 1, 0, 4, 3, 4, 5, 4, 2, 3, 3, 1, 3, 3, 2, 3, 1, 2, 3, 1, 4, 0, 4, 4, 4, 3, 5, 4, 4, 2, 2, 1, 1, 1, 2, 2, 3, 3, 1, 3, 5, 4, 3, 4, 1, 2, 3, 4, 1, 4, 2, 1, 4, 3, 1, 1, 0, 1, 0, 1, 1, 3, 3, 1, 2, 1, 1, 2, 0, 0, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 3, 3, 3, 2, 3, 2, 2, 0, 2, 2, 1, 4, 2, 2, 0, 2, 2, 3, 2, 1, 2, 1, 2, 1, 2, 4, 4, 1, 2, 2, 3, 3, 3, 2, 3, 2, 2, 2, 4, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 1, 1, 2, 3, 5, 4, 2, 3, 2, 3, 2, 4, 3, 3, 2, 0, 3, 4, 3, 2, 1, 2, 0, 1, 1, 2, 1, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 3, 2, 3, 2, 2, 1, 1, 3, 3, 5, 4, 3, 1, 0, 1, 0, 2, 4, 1, 2, 3, 2, 2, 4, 6, 5, 3, 3, 3, 4, 3, 3, 3, 3, 2, 1, 4, 5, 4, 3, 4, 1, 2, 3, 4, 1, 4, 4, 1, 4, 5, 4, 3, 4, 2, 1, 1, 5, 4, 3, 2, 3, 3, 3, 3, 4, 1, 3, 4, 4, 1, 5, 2, 3, 4, 3, 4, 2, 3, 4, 3, 3, 4, 4, 4, 5, 3, 3, 3, 2, 1, 4, 3, 1, 1, 0, 4, 1, 4, 3, 4, 1, 4, 1, 0, 2, 3, 2, 4, 0, 3, 3, 3, 4, 3, 3, 3, 4, 0, 4, 1, 5, 4, 4, 0, 0, 4, 3, 4, 4, 0, 2, 0, 0, 4, 4, 4, 4, 4, 3, 4, 4, 2, 2, 4, 3, 3, 1, 1, 1, 0, 2, 0, 1, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 1, 2, 4, 3, 0, 2, 3, 0, 2, 5, 0, 2, 1, 4, 3, 1, 1, 2, 1, 6, 3, 3, 3, 3, 3, 3, 1, 0, 3, 1, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 1, 1, 5, 3, 5, 4, 1, 4, 1, 2, 5, 4, 1, 1, 4, 4, 2, 3, 2, 5, 4, 3, 0, 2, 0, 0, 0, 0, 0, 3, 1, 3, 1, 1, 1, 3, 3, 4, 3, 3, 4, 4, 1, 4, 2, 3, 1, 1, 1, 3, 3, 4, 3, 2, 4, 3, 4, 5, 1, 3, 1, 5, 1, 1, 4, 1, 3, 1, 5, 4, 1, 4, 1, 3, 1, 4, 1, 0, 1, 3, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3, 1, 3, 5, 4, 2, 1, 5, 1, 5, 1, 3, 1, 2, 2, 3, 4, 3, 2, 1, 2, 1, 3, 1, 2, 1, 1, 0, 3, 2, 4, 3, 3, 3, 3, 1, 2, 4, 3, 3, 3, 1, 2, 3, 1, 5, 4, 2, 0, 1, 0, 4, 5, 4, 0, 3, 3, 2, 3, 4, 1, 3, 4, 5, 2, 1, 3, 3, 3, 3, 3, 0, 1, 5, 4, 3, 0, 4, 2, 3, 1, 4, 4, 4, 5, 1, 3, 1, 5, 1, 4, 2, 4, 4, 2, 1, 4, 3, 3, 2, 3, 4, 1, 3, 4, 4, 1, 3, 2, 4, 4, 1, 0, 1, 1, 1, 0, 0, 4, 4, 2, 3, 1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1, 4, 1, 3, 1, 5, 4, 1, 1, 2, 2, 2, 1, 0, 4, 5, 2, 1, 3, 3, 3, 1, 0, 1, 0, 0, 0, 0, 4, 0, 1, 1, 0, 5, 3, 3, 4, 2, 3, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 0, 0, 2, 3, 5, 4, 1, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 3, 1, 3, 4, 1, 1, 4, 3, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 2, 4, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 0, 0, 4, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 0, 1, 2, 2, 4, 2, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1, 2, 2, 1, 2, 2, 2, 2, 0, 2, 2, 3, 2, 1, 2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 1, 3, 4, 2, 2, 0, 4, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 1, 0, 1, 3, 2, 2, 2, 2, 1, 0, 0, 0, 1, 0, 1, 1, 5, 0, 1, 2, 3, 5, 2, 2, 4, 1, 1, 1, 2, 1, 5, 0, 3, 2, 0, 3, 1, 2, 0, 5, 2, 2, 2, 3, 3, 2, 1, 2, 2, 0, 3, 1, 2, 0, 1, 3, 1, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 3, 6, 5, 2, 3, 1, 1, 6, 2, 4, 1, 2, 1, 1, 5, 4, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 4, 2, 2, 2, 2, 1, 1, 1, 3, 4, 6, 5, 3, 3, 3, 2, 4, 6, 5, 2, 1, 2, 1, 1, 6, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 1, 2, 1, 2, 1, 0, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 3, 6, 1, 4, 2, 3, 1, 1, 6, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 4, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 1, 2, 1, 1, 1, 1, 1, 2, 2, 3, 2, 5, 4, 2, 2, 5, 1, 5, 2, 3, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 3, 1, 1, 6, 0, 3, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 1, 5, 0, 0, 0, 3, 0, 0, 2, 2, 3, 2, 2, 2, 2, 3, 1, 3, 4, 3, 1, 1, 2, 2, 1, 2, 2, 2, 2, 3, 1, 0, 0, 1, 3, 3, 2, 2, 4, 4, 3, 2, 1, 2, 1, 1, 4, 3, 2, 2, 3, 1, 3, 4, 3, 4, 3, 2, 2, 2, 2, 2, 3, 1, 2, 1, 2, 2, 1, 3, 4, 2, 4, 4, 4, 3, 1, 1, 2, 2, 4, 4, 3, 1, 1, 5, 0, 0, 0, 3, 0, 0, 0, 1, 1, 1, 4, 1, 1, 0, 1, 2, 2, 3, 2, 2, 3, 2, 1, 3, 3, 3, 3, 3, 4, 3, 4, 3, 3, 3, 0, 1, 2, 5, 3, 5, 3, 0, 1, 2, 2, 3, 3, 2, 1, 3, 1, 3, 5, 4, 2, 1, 2, 2, 3, 5, 4, 1, 1, 1, 1, 3, 5, 4, 1, 2, 2, 3, 2, 5, 4, 2, 4, 2, 2, 3, 2, 1, 4, 3, 2, 2, 3, 4, 1, 3, 1, 2, 3, 2, 5, 4, 1, 3, 1, 3, 4, 3, 1, 1, 3, 2, 3, 2, 1, 1, 1, 3, 4, 3, 4, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 0, 1, 3, 2, 2, 4, 3, 1, 4, 3, 4, 3, 3, 4, 1, 3, 3, 4, 3, 4, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 1, 4, 3, 2, 1, 1, 3, 3, 3, 3, 5, 3, 1, 0, 3, 2, 2, 2, 1, 2, 1, 3, 2, 2, 5, 2, 2, 2, 5, 2, 2, 2, 2, 1, 2, 3, 4, 3, 1, 1, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 4, 3, 1, 1, 3, 1, 3, 3, 4, 3, 5, 1, 2, 3, 2, 5, 4, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 3, 4, 3, 1, 1, 2, 2, 6, 2, 2, 2, 2, 0, 3, 2, 3, 2, 3, 2, 3, 3, 2, 1, 2, 2, 0, 2, 2, 3, 2, 0, 6, 0, 2, 1, 0, 0, 2, 5, 4, 2, 1, 4, 2, 2, 6, 5, 3, 2, 6, 2, 6, 2, 4, 2, 1, 3, 4, 5, 4, 2, 3, 3, 2, 1, 2, 2, 0, 3, 1, 2, 0, 1, 3, 1, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 3, 6, 5, 2, 3, 1, 1, 6, 2, 4, 1, 2, 1, 1, 5, 4, 2, 1, 2, 3, 1, 5, 4, 2, 4, 3, 2, 3, 5, 4, 4, 3, 2, 2, 2, 5, 4, 2, 0, 1, 2, 1, 5, 4, 1, 2, 3, 3, 3, 3, 1, 4, 2, 3, 2, 3, 3, 3, 4, 0, 3, 2, 1, 5, 4, 1, 0, 1, 0, 4, 5, 4, 0, 1, 1, 4, -1, 4, 4, 2, 3, 3, 3, 1, 4, 1, 4, 0, 1, 2, 0, 4, 6, 2, 3, 3, 4, 5, 2, 6, 5, 3, 3, 1, 2, 6, 2, 4, 3, 3, 4, 4, 5, 4, 3, 3, 3, 2, 3, 4, 1, 3, 3, 5, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 1, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 1, 3, 3, 4, 3, 1, 3, 2, 3, 4, 3, 1, 3, 3, 4, 4, 3, 1, 1, 4, 5, 2, 1, 3, 3, 3, 5, 3, 3, 4, 3, 3, 3, 2, 3, 1, 3, 1, 2, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 0, 1, 5, 4, 3, 3, 1, 3, 4, 5, 4, 3, 0, 3, 2, 1, 5, 4, 1, 0, 1, 1, 1, 5, 4, 1, 3, 3, 4, 4, 3, 1, 1, 3, 3, 4, 3, 3, 3, 3, 1, 0, 1, 1, 5, 4, 6, 0, 4, 2, 3, 1, 4, 4, 4, 5, 1, 3, 1, 5, 1, 4, 2, 4, 4, 2, 1, 4, 3, 3, 2, 3, 4, 1, 3, 4, 4, 1, 3, 2, 4, 4, 1, 0, 1, 1, 1]]
        self.HORIZONTAL_MULTIPLIER = 2.5
        self.VERTICAL_MULTIPLIER = 1.5
        self.DIAGONAL_MULTIPLIER = 2
        self.CENTER_MULTIPLIER = 0.25
        self.MAX_DEPTH = 3
        self.MAX_VALUE = 999999
        self.dp = {}
        self.prev_state = None if self.player_id == 1 else initialize()
        self.enemy_moves = ""
        self.permutations = []
        self.optimal = True
        self.MAX_SIZE = 60000
        for i in range(0, 4):
            self.permutations.extend([''.join(p) for p in product('0123456', repeat=i)])
        self.permutations.sort(key=lambda x: (len(x), x))
        if self.player_id == 2:
            self.permutations = self.permutations[1:]
        
    def get_optimal_move(self):
        cur = self.enemy_moves
        print("cur: ", cur)
        return self.OPTIMAL_MOVES[(self.player_id + 1) % 2][self.permutations.index(cur)]
    
    def heuristic(self,state):
        score = 0
        HORIZONTAL_MULTIPLIER = 1
        VERTICAL_MULTIPLIER = 0.5
        DIAGONAL_MULTIPLIER = 0.8
        FORK_MULTIPLIER = 3
        ROW_MULTIPLIER = 0.1
        POTENTIAL_MULTIPLIER = 0.1
        BRIDGE_MULTIPLIER = 0.5
        MOBILITY_MULTIPLIER = 0.05
        def count_threats(player):
            score = 0
            imm = 0
            pot = 0
            for row in range(3):
                for col in range(4):
                    window = [state[row+3-i][col+i] for i in range(4)]
                    count = sum(1 for x in window if x == player)
                    nonzero_count = np.count_nonzero(window)                   
                    if count == 3 and nonzero_count == 3:
                        imm += 1
                        score += DIAGONAL_MULTIPLIER
                    elif count == 2 and nonzero_count == 2:
                        pot += 1
                        score += DIAGONAL_MULTIPLIER * POTENTIAL_MULTIPLIER
                    window = [state[row+i][col+i] for i in range(4)]
                    count = sum(1 for x in window if x == player)
                    nonzero_count = np.count_nonzero(window) 
                    if count == 3 and nonzero_count == 3:
                        imm += 1
                        score += DIAGONAL_MULTIPLIER
                    elif count == 2 and nonzero_count == 2:
                        pot += 1
                        score += DIAGONAL_MULTIPLIER * POTENTIAL_MULTIPLIER        
            for row in range(6):
                for col in range(4):
                    window = state[row, col:col+4]
                    count = np.count_nonzero(window == player)
                    nonzero_count = np.count_nonzero(window)
                    if count == 3 and nonzero_count == 3:
                        imm += 1
                        score += HORIZONTAL_MULTIPLIER
                    elif count == 2 and nonzero_count == 2:
                        pot += 1
                        score += HORIZONTAL_MULTIPLIER * POTENTIAL_MULTIPLIER
            for row in range(3):
                for col in range(7):
                    window = state[row:row+4, col]
                    count = np.count_nonzero(window == player)
                    nonzero_count = np.count_nonzero(window)
                    
                    if count == 3 and nonzero_count == 3:
                        imm += 1
                        score += VERTICAL_MULTIPLIER
                    elif count == 2 and nonzero_count == 2:
                        pot = 1
                        score += VERTICAL_MULTIPLIER * POTENTIAL_MULTIPLIER
                
            return score, imm, pot
            
        score1, imm1, pot1 = count_threats(self.player_id)
        score2, imm2, pot2 = count_threats([1, 2][self.player_id % 2])
        if imm1 > 1:
            score1 += FORK_MULTIPLIER
        if imm2 > 1:
            score2 += FORK_MULTIPLIER
        
        for row in range(6):
            for col in range(5):  # Check only up to the 5th column
                if state[row, col] == 1 and state[row, col + 1] == 1:
                    if (col - 1 >= 0 and state[row, col - 1] == 0) or (col + 2 < 7 and state[row, col + 2] == 0):  # Check for empty spaces on either side
                        score1 += BRIDGE_MULTIPLIER
        for row in range(6):
            for col in range(5):  # Check only up to the 5th column
                if state[row, col] == 2 and state[row, col + 1] == 1:
                    if (col - 1 >= 0 and state[row, col - 1] == 0) or (col + 2 < 7 and state[row, col + 2] == 0):  # Check for empty spaces on either side
                        score2 += BRIDGE_MULTIPLIER

        # Mobility bonus
        valid_moves = get_valid_col_id(state)
        score += len(valid_moves) * MOBILITY_MULTIPLIER  # Bonus for keeping options open
        score = score1 - score2
        return score

    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
        hashed_state = hash(state.tobytes())
        if hashed_state in self.dp:
                stored_depth, stored_value, stored_move = self.dp[hashed_state]
                if stored_depth >= depth:
                    return stored_value, stored_move
        if depth == 0:
            return self.heuristic(state), None
        if is_win(state):
            return -self.MAX_VALUE if maximizing_player else self.MAX_VALUE, None
        valid_moves = get_valid_col_id(state)
        player = self.player_id if maximizing_player else [2, 1][self.player_id - 1]
        best_move = None
        if maximizing_player:
            max_eval = -self.MAX_VALUE
            for move in valid_moves:
                new_state = step(state, move, player, False)
                eval, _ = self.alpha_beta_pruning(new_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.dp[hashed_state] = (depth, max_eval, best_move)
            if len(self.dp) > self.MAX_SIZE:
                self.dp.clear()
            return max_eval, best_move
        else:
            min_eval = self.MAX_VALUE
            for move in valid_moves:
                new_state = step(state, move, player, False)
                # Apply the move to the new_state
                eval, _ = self.alpha_beta_pruning(new_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.dp[hashed_state] = (depth, min_eval, best_move)
            if len(self.dp) > self.MAX_SIZE:
                self.dp.clear()
            return min_eval, best_move
    
    def get_opponent_move(self, state):
        difference = state - self.prev_state
        row, col = np.argwhere(difference != 0)[0]
        self.enemy_moves += str(col)

    def make_move(self, state):
        """
        Determines and returns the next move for the agent based on the current game state.

        Parameters:
        -----------
        state : np.ndarray
            A 2D numpy array representing the current, read-only state of the game board. 
            The board contains:
            - 0 for an empty cell,
            - 1 for Player 1's piece,
            - 2 for Player 2's piece.

        Returns:
        --------
        int
            The valid action, ie. a valid column index (col_id) where this agent chooses to drop its piece.
        """
        """ YOUR CODE HERE """
        if self.optimal and self.prev_state is not None:
            self.get_opponent_move(state)
        if self.optimal and self.enemy_moves in self.permutations:
            best_move = self.get_optimal_move()
            next_state = step(state, best_move, self.player_id, False)
            self.prev_state = next_state
            return best_move
        else:
            self.optimal = False
        possible_moves = get_valid_col_id(state)
        best_score = -2 * self.MAX_VALUE
        best_move = None
        for move in possible_moves:
            new_state = step(state, move, self.player_id, False)
            score, _ = self.alpha_beta_pruning(new_state, self.MAX_DEPTH, -self.MAX_VALUE, self.MAX_VALUE, False)
            score += self.CENTER_MULTIPLIER if move == 3 else 0
            if score > best_score:
                best_score = score
                best_move = move
        next_state = step(state, best_move, self.player_id, False)
        self.prev_state = next_state
        return best_move
    
    from game_utils import initialize, step, get_valid_col_id, is_win
    import numpy as np

class AIAgentNonOp(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id

        # Weights for evaluation function
        if self.player_id == 1:
            # Going First Weights
            self.weights = {
                'horizontal_immediate_threat_score': 160.0,
                'horizontal_potential_threat_score': 50.0,
                'vertical_immediate_threat_score': 170.0,
                'vertical_potential_threat_score': 55.0,
                'diagonal_immediate_threat_score': 175.0,
                'diagonal_potential_threat_score': 60.0,
                'aggressive_defensive_balance_score': 1.0,
                'fork_score': 425.0,
                'vertical_bridge_score': 40.0,
                'horizontal_bridge_score': 50.0,
                'center_control_score': 100.0,
                'height_penalty': 1.5,
                'mobility_bonus': 45.0
            }
        else:
            # Going Second Weights
            self.weights = {
                'horizontal_immediate_threat_score': 170.0,
                'horizontal_potential_threat_score': 35.0,
                'vertical_immediate_threat_score': 190.0,
                'vertical_potential_threat_score': 50.0,
                'diagonal_immediate_threat_score': 185.0,
                'diagonal_potential_threat_score': 55.0,
                'aggressive_defensive_balance_score': 1.8,
                'fork_score': 375.0,
                'vertical_bridge_score': 45.0,
                'horizontal_bridge_score': 40.0,
                'center_control_score': 80.0,
                'height_penalty': 2.0,
                'mobility_bonus': 50.0
            }

        self.MAX_DEPTH = 4
        self.MAX_VALUE = 999999
        
        # Transposition table for memoization
        self.transposition_table = {}
        self.max_table_size = 50000
        self.clear_threshold = 40000

    def evaluate_position(self, state):
        if is_win(state):
            return self.MAX_VALUE
            
        score = 0

        def count_threats(player):
            threat_score = 0
            immediate_threats = 0
            potential_threats = 0

            # Horizontal threats
            for row in range(6):
                for col in range(4):
                    window = state[row, col:col+4]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['horizontal_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['horizontal_potential_threat_score']
            
            # Vertical threats
            for row in range(3):
                for col in range(7):
                    window = state[row:row+4, col]
                    player_count = np.count_nonzero(window == player)
                    empty_count = np.count_nonzero(window == 0)
                    
                    if player_count == 3 and empty_count == 1:
                        immediate_threats += 1
                        threat_score += self.weights['vertical_immediate_threat_score']
                    elif player_count == 2 and empty_count == 2:
                        potential_threats += 1
                        threat_score += self.weights['vertical_potential_threat_score']
            
            # Diagonal threats
            for row in range(3):
                for col in range(4):
                    window_positive = [state[row+i][col+i] for i in range(4)]
                    player_count_positive = sum(1 for x in window_positive if x == player)
                    empty_count_positive = sum(1 for x in window_positive if x == 0)
                    
                    if player_count_positive == 3 and empty_count_positive == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_positive == 2 and empty_count_positive == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
                    
                    window_negative = [state[row+3-i][col+i] for i in range(4)]
                    player_count_negative = sum(1 for x in window_negative if x == player)
                    empty_count_negative = sum(1 for x in window_negative if x == 0)
                    
                    if player_count_negative == 3 and empty_count_negative == 1:
                        immediate_threats += 1
                        threat_score += self.weights['diagonal_immediate_threat_score']
                    elif player_count_negative == 2 and empty_count_negative == 2:
                        potential_threats += 1
                        threat_score += self.weights['diagonal_potential_threat_score']
            
            return threat_score, immediate_threats, potential_threats

        player_threats, player_immediate, _ = count_threats(self.player_id)
        opponent_threats, opponent_immediate, _ = count_threats(self.opponent_id)

        # Aggressive-defensive balance
        score += player_threats - opponent_threats * self.weights['aggressive_defensive_balance_score']
        
        # Fork detection
        if player_immediate > 1:
            score += self.weights['fork_score']
        if opponent_immediate > 1:
            score -= self.weights['fork_score']

        # Vertical bridge detection
        for col in range(7):
            if np.count_nonzero(state[:, col] == self.player_id) == 2:
                empty_below = np.count_nonzero(state[5:, col] == 0)
                if empty_below > 0:
                    score += self.weights['vertical_bridge_score']

        # Horizontal bridge detection
        for row in range(6):
            for col in range(5):
                if state[row, col] == self.player_id and state[row, col + 1] == self.player_id:
                    if (col - 1 >= 0 and state[row, col - 1] == 0) or (col + 2 < 7 and state[row, col + 2] == 0):
                        score += self.weights['horizontal_bridge_score']
        
        # Center control bonus
        center_array = state[:, 3]
        center_score = np.count_nonzero(center_array == self.player_id) * self.weights['center_control_score']
        score += center_score

        # Height penalty to prefer lower positions
        for col in range(7):
            col_height = np.count_nonzero(state[:, col])
            if col_height > 0:
                score -= col_height * self.weights['height_penalty']
        
        # Mobility bonus
        valid_moves = get_valid_col_id(state)
        score += len(valid_moves) * self.weights['mobility_bonus']
        
        return score

    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
        # Transposition table lookup
        state_hash = hash(state.tobytes())
        if state_hash in self.transposition_table:
            stored_depth, stored_value, stored_move = self.transposition_table[state_hash]
            if stored_depth >= depth:
                return stored_value, stored_move

        # Base cases with checks
        if is_win(state):
            return (-self.MAX_VALUE if maximizing_player else self.MAX_VALUE), None
            
        if depth == 0:
            return self.evaluate_position(state), None

        valid_moves = get_valid_col_id(state)
        
        # Handle edge cases
        if len(valid_moves) == 0:
            return 0, None
            
        if len(valid_moves) == 1:
            return 0, valid_moves[0]
        
        # Move ordering (optional for performance)
        # You can implement move ordering here based on heuristic evaluations.

        # Main search logic
        if maximizing_player:
            value = float('-inf')
            best_move = valid_moves[0]
            
            for move in valid_moves:
                new_state = step(state, move, self.player_id, False)
                eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, False)
                
                if eval_score > value:
                    value = eval_score
                    best_move = move
                    
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = float('inf')
            best_move = valid_moves[0]
            
            for move in valid_moves:
                new_state = step(state, move, self.opponent_id, False)
                eval_score, _ = self.alpha_beta_pruning(new_state, depth-1, alpha, beta, True)
                
                if eval_score < value:
                    value = eval_score
                    best_move = move
                    
                beta = min(beta, value)
                if beta <= alpha:
                    break

        # Update transposition table
        self.transposition_table[state_hash] = (depth, value, best_move)
        if len(self.transposition_table) > self.max_table_size:
            self.transposition_table.clear()

        return value, best_move

    def make_move(self, state):
        valid_moves = get_valid_col_id(state)
        
        # Handle single move case
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Quick win/block check
        for move in valid_moves:
            # Check winning move
            new_state = step(state, move, self.player_id, False)
            if is_win(new_state):
                return move
                
        for move in valid_moves:
            # Check blocking move
            new_state = step(state, move, self.opponent_id, False)
            if is_win(new_state):
                return move
        
        # Regular search with validation
        _, best_move = self.alpha_beta_pruning(state, self.MAX_DEPTH, float('-inf'), float('inf'), True)
        
        if best_move is not None and best_move in valid_moves:
            return best_move
            
        # Fallback to center-focused strategy
        return min(valid_moves, key=lambda x: abs(x - 3))
