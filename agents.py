from game_utils import initialize, step, get_valid_col_id, is_win
import numpy as np
import random

class AIAgent(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.depthMax = 4
        
        # Initialize transposition table with optimized size
        self.table_size = 2097152  # Optimized size
        self.trans_table = {}
        
        # Pre-compute column order (center-focused)
        self.column_order = [3, 4, 2, 5, 1, 6, 0]
        
        # Core evaluation weights (optimized)
        self.horiFactor = 1.5990909781768534
        self.vertFactor = 1.9494985131455613
        self.slantedFactor = 2.7583460186691884
        self.midFactor = 0.5827992706213135
        self.three_in_row_weight = 6.2063446108401426
        self.two_in_row_weight = 1.5968756202658536
        self.blocking_weight = 5.2528839165062156
      
        # Position weights matrix (optimized)
        self.position_weights = np.array([
            [ 4.70359286,  0.99559081,  4.38309755, 12.64871894,  2.39195936,  0.69010755, 7.6074756 ],
            [ 8.22975751,  5.06248076,  9.52007571,  4.65878   ,  3.25727208,  2.25353991, 4.63191304],
            [ 4.48767285,  9.04200705, 12.92681254,  1.77837709,  7.98920157, 10.71530826, 1.53310598],
            [ 6.54432238,  5.50378947,  8.47773185, 10.29124116, 10.98548771,  2.00135297, 4.05457742],
            [ 1.37334287,  2.6304275 , 10.77721742,  2.0083834 , 13.57188319, 13.06650791, 4.2197079 ],
            [ 8.6987031 ,  7.58753176, 13.97102872,  5.31164886,  4.86272716,  1.92013464, 12.50690872]
        ], dtype=np.float32)

        # Enhanced evaluation parameters (optimized)
        self.winning_score = 1000000  # Keep original
        self.position_weight_multiplier = 9.503926767750782
        self.center_column_bias = 1.8492536892779754
        
        # Game phase weights (optimized)
        self.early_game_weight = 1.3912858554038834
        self.mid_game_weight = 0.8425932811305796
        self.late_game_weight = 0.785563790610896
        
        # Search parameters (optimized)
        self.alpha_beta_window = 103.65587642613303

    def make_move(self, state):
        valid_moves = [col for col in range(7) if state[0][col] == 0]
        
        # Quick check for winning moves and blocks
        for col in valid_moves:
            # Check for win
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.player_id
                if self._is_winning_move(temp_state, row, col, self.player_id):
                    return col
                temp_state[row][col] = 0
                
            # Check for block
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.opponent_id
                if self._is_winning_move(temp_state, row, col, self.opponent_id):
                    return col

        # Use minimax with alpha-beta pruning
        best_score = float('-inf')
        best_move = valid_moves[0]
        
        # Sort moves by column preference
        moves = sorted(valid_moves, key=lambda x: -abs(x-3))
        
        for col in moves:
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.player_id
                score = self.minimax(temp_state, self.depthMax-1, float('-inf'), float('inf'), False)
                if score > best_score:
                    best_score = score
                    best_move = col
                    
        return best_move

    def minimax(self, state, depth, alpha, beta, maximizing):
        # Use alpha-beta window for pruning
        if maximizing:
            alpha = max(alpha, -self.winning_score - self.alpha_beta_window)
        else:
            beta = min(beta, self.winning_score + self.alpha_beta_window)
            
        # Check transposition table
        key = hash(state.tobytes())
        if key in self.trans_table and self.trans_table[key][0] >= depth:
            return self.trans_table[key][1]
            
        if depth == 0:
            return self.evaluate_board(state)
            
        valid_moves = [col for col in range(7) if state[0][col] == 0]
        if not valid_moves:
            return 0
            
        if maximizing:
            max_eval = float('-inf')
            for col in sorted(valid_moves, key=lambda x: -abs(x-3)):
                row = self._get_row(state, col)
                if row != -1:
                    state[row][col] = self.player_id
                    if self._is_winning_move(state, row, col, self.player_id):
                        eval = 1000000 + depth
                    else:
                        eval = self.minimax(state, depth-1, alpha, beta, False)
                    state[row][col] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            
            # Store in transposition table
            if len(self.trans_table) >= self.table_size:
                self.trans_table.clear()
            self.trans_table[key] = (depth, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            for col in sorted(valid_moves, key=lambda x: -abs(x-3)):
                row = self._get_row(state, col)
                if row != -1:
                    state[row][col] = self.opponent_id
                    if self._is_winning_move(state, row, col, self.opponent_id):
                        eval = -1000000 - depth
                    else:
                        eval = self.minimax(state, depth-1, alpha, beta, True)
                    state[row][col] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                        
            # Store in transposition table
            if len(self.trans_table) >= self.table_size:
                self.trans_table.clear()
            self.trans_table[key] = (depth, min_eval)
            return min_eval

    def _get_row(self, state, col):
        """Fast method to get the next available row in a column"""
        for row in range(5, -1, -1):
            if state[row][col] == 0:
                return row
        return -1

    def _is_winning_move(self, state, row, col, player):
        """Optimized win check only around the last move"""
        # Horizontal
        count = 0
        for c in range(max(0, col-3), min(7, col+4)):
            if state[row][c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
                
        # Vertical
        if row <= 2:
            if all(state[row+i][col] == player for i in range(4)):
                return True
                
        # Diagonal checks
        for i in range(-3, 1):
            # Positive slope
            count = 0
            for j in range(4):
                if (0 <= row+i+j < 6 and 
                    0 <= col+j < 7 and 
                    state[row+i+j][col+j] == player):
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
                    
            # Negative slope
            count = 0
            for j in range(4):
                if (0 <= row+i+j < 6 and 
                    0 <= col-j < 7 and 
                    state[row+i+j][col-j] == player):
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
        return False

    def evaluate_board(self, state):
        # Determine game phase
        moves_played = np.count_nonzero(state)
        if moves_played < 12:
            phase_weight = self.early_game_weight
        elif moves_played < 24:
            phase_weight = self.mid_game_weight
        else:
            phase_weight = self.late_game_weight
            
        score = 0
        
        # Position weights with multiplier
        for i in range(6):
            for j in range(7):
                if state[i][j] == self.player_id:
                    score += self.position_weights[i][j] * self.position_weight_multiplier
                elif state[i][j] == self.opponent_id:
                    score -= self.position_weights[i][j] * self.position_weight_multiplier
        
        # Apply center column bias
        center_pieces = sum(1 for i in range(6) if state[i][3] == self.player_id)
        score += center_pieces * self.center_column_bias
        
        # Evaluate sequences with phase weight
        score += self._evaluate_sequences(state) * phase_weight
        
        return score

    def _evaluate_sequences(self, state):
        score = 0
        
        # Horizontal sequences
        for row in range(6):
            for col in range(4):
                window = list(state[row, col:col+4])
                score += self._evaluate_window(window) * self.horiFactor
                
        # Vertical sequences
        for row in range(3):
            for col in range(7):
                window = [state[row+i][col] for i in range(4)]
                score += self._evaluate_window(window) * self.vertFactor
                
        # Diagonal sequences
        for row in range(3):
            for col in range(4):
                # Positive slope
                window = [state[row+i][col+i] for i in range(4)]
                score += self._evaluate_window(window) * self.slantedFactor
                
                # Negative slope
                window = [state[row+3-i][col+i] for i in range(4)]
                score += self._evaluate_window(window) * self.slantedFactor
                
        return score

    def _evaluate_window(self, window):
        score = 0
        player_count = window.count(self.player_id)
        opponent_count = window.count(self.opponent_id)
        empty_count = window.count(0)
        
        if player_count == 4:
            score += 1000
        elif player_count == 3 and empty_count == 1:
            score += self.three_in_row_weight
        elif player_count == 2 and empty_count == 2:
            score += self.two_in_row_weight
            
        if opponent_count == 3 and empty_count == 1:
            score -= self.blocking_weight
            
        return score

class AIAgent2(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.depthMax = 4
        self.opponent_id = 3 - player_id
        
        # Initialize transposition table and its size limit
        self.transposition_table = {}
        self.max_table_size = 2**21
        
        self.horiFactor = 5.775408907109855      
        self.vertFactor = 2.6430478452207185    
        self.slantedFactor = 4.295768794326157 
        self.midFactor = 1.7259132063416474    

        # Updated pattern weights
        self.three_in_row_weight = 14.104559183824938  
        self.two_in_row_weight = 8.268010700665716    
        self.blocking_weight = 7.159522221999543      
        
        # Updated position weights from search
        self.position_weights = np.array([
            [ 1.23860216,  2.73850437, 14.0071334 ,  6.12342321,  3.23991672,  7.59300562,  8.04716999],
            [ 9.16905538,  3.81054018,  9.69596729, 12.32434684,  2.41487186,  8.72575823, 10.14652539],
            [11.04181352,  2.88602554,  6.07071602,  7.44794302,  2.36995105, 11.98727083,  1.90939228],
            [14.11700105,  1.47386146, 11.67853867, 12.07477527,  3.86484734,  9.3666596 , 12.76659639],
            [10.04788352,  6.82449552,  6.2776096 , 10.22638499,  4.53091542, 11.94955921,  7.48897421],
            [ 5.75554054,  4.75812942,  1.63330726, 11.40432902,  5.75804151,  8.61651056,  6.18893644]
        ], dtype=np.float32)

    def make_move(self, state):
        valid_moves = get_valid_col_id(state)
        moves_played = np.count_nonzero(state)
        
        # Center first strategy for opening moves
        if moves_played <= 1 and 3 in valid_moves:
            return 3
            
        # Quick check for winning moves and blocks
        for col in valid_moves:
            temp_state = state.copy()
            
            # Check for win
            step(temp_state, col, self.player_id, in_place=True)
            if is_win(temp_state):
                return col
                
            # Check for block
            temp_state = state.copy()
            step(temp_state, col, self.opponent_id, in_place=True)
            if is_win(temp_state):
                return col
        
        # Clear transposition table if too large
        if len(self.transposition_table) > self.max_table_size:
            self.transposition_table.clear()
        
        best_col, _ = self.minimax(state, self.depthMax, float('-inf'), float('inf'), True)
        return best_col

    def evaluate_position(self, state, player_id):
        score = 0
        opponent_id = 3 - player_id
        
        # Fast position-based scoring using numpy operations
        player_positions = (state == player_id)
        opponent_positions = (state == opponent_id)
        score += np.sum(self.position_weights * player_positions) * 10
        score -= np.sum(self.position_weights * opponent_positions) * 10
        
        # Evaluate horizontal windows (optimized)
        for row in range(6):
            for col in range(4):
                window = state[row, col:col+4]
                score += self.evaluate_window(window, player_id) * self.horiFactor
        
        # Evaluate vertical windows (optimized)
        for row in range(3):
            for col in range(7):
                window = state[row:row+4, col]
                score += self.evaluate_window(window, player_id) * self.vertFactor
        
        # Evaluate diagonal windows
        for row in range(3):
            for col in range(4):
                # Positive slope diagonal
                window = [state[row+i, col+i] for i in range(4)]
                score += self.evaluate_window(window, player_id) * self.slantedFactor
                
                # Negative slope diagonal
                window = [state[row+3-i, col+i] for i in range(4)]
                score += self.evaluate_window(window, player_id) * self.slantedFactor
        
        # Quick center control evaluation
        center_array = state[:, 3]
        center_count = np.count_nonzero(center_array == player_id)
        score += center_count * self.midFactor * 6
        
        return score

    def evaluate_window(self, window, player_id):
        """Optimized window evaluation with enhanced pattern detection"""
        score = 0
        player_count = np.count_nonzero(window == player_id)
        empty_count = np.count_nonzero(window == 0)
        opponent_count = 4 - player_count - empty_count
        
        # Fast pattern matching with new weights
        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += self.three_in_row_weight
        elif player_count == 2 and empty_count == 2:
            score += self.two_in_row_weight
        
        if opponent_count == 3 and empty_count == 1:
            score -= self.blocking_weight
        elif opponent_count == 2 and empty_count == 2:
            score -= self.two_in_row_weight * 0.5
        
        return score

    def minimax(self, state, depth, alpha, beta, maximizing_player):
        state_hash = hash(state.tobytes())
        if state_hash in self.transposition_table and self.transposition_table[state_hash][0] >= depth:
            return self.transposition_table[state_hash][1:]
            
        valid_moves = get_valid_col_id(state)
        
        if depth == 0 or len(valid_moves) == 0:
            return (None, self.evaluate_position(state, self.player_id))
        
        # Quick move ordering for better pruning
        if maximizing_player:
            value = float('-inf')
            column = valid_moves[0]
            
            # Prioritize center and near-center columns
            ordered_moves = sorted(valid_moves, key=lambda x: -abs(x-3))
            
            for col in ordered_moves:
                temp_state = state.copy()
                step(temp_state, col, self.player_id, in_place=True)
                
                new_score = self.minimax(temp_state, depth-1, alpha, beta, False)[1]
                
                if new_score > value:
                    value = new_score
                    column = col
                
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            
            self.transposition_table[state_hash] = (depth, column, value)
            return column, value
        
        else:
            value = float('inf')
            column = valid_moves[0]
            
            ordered_moves = sorted(valid_moves, key=lambda x: -abs(x-3))
            
            for col in ordered_moves:
                temp_state = state.copy()
                step(temp_state, col, 3-self.player_id, in_place=True)
                
                new_score = self.minimax(temp_state, depth-1, alpha, beta, True)[1]
                
                if new_score < value:
                    value = new_score
                    column = col
                
                beta = min(beta, value)
                if alpha >= beta:
                    break
            
            self.transposition_table[state_hash] = (depth, column, value)
            return column, value

class AIAgent3(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.depthMax = 4
        
        # Initialize transposition table with optimized size
        self.table_size = 4194304 
        self.trans_table = {}
        
        # Pre-compute column order (center-focused)
        self.column_order = [3, 4, 2, 5, 1, 6, 0]
        
        # Core evaluation weights (optimized)
        self.horiFactor = 3.8600311277385058
        self.vertFactor = 3.1591795025727345
        self.slantedFactor = 3.9692139750003896
        self.midFactor = 0.35518420559559066
        self.three_in_row_weight = 4.731966853994749
        self.two_in_row_weight = 2.1156102802282652
        self.blocking_weight = 7.76377766764153
      
        # Position weights matrix (optimized)
        self.position_weights = np.array([
            [11.23365676,  9.6018381,  14.76228616, 13.98606701,  8.62677406, 13.95816206,  2.98520278],
            [11.21382001,  2.54020529,  2.54363826,  5.14126,  5.81140495,  3.3385982,  4.52016671],
            [12.8955076,  12.02591045,  4.02357085,  3.42308578,  4.91078337,  9.94105508,  9.68249969],
            [ 4.75146767,  1.22162071, 10.73620986, 11.41561618,  4.873666,  8.75913189,  4.81081042],
            [13.06580423,  6.68769955, 16.09619108,  7.73154306,  7.59529664,  4.01965681,  1.45897385],
            [ 2.98935032,  2.80578649, 13.8185484,  11.82347541,  2.25112894, 14.20285944,  5.47644603]], dtype=np.float32)

        # Enhanced evaluation parameters (optimized)
        self.winning_score = 1000000 
        self.position_weight_multiplier = 10.501517859730082
        self.center_column_bias = 1.6767659847192964
        
        # Game phase weights (optimized)
        self.early_game_weight = 1.6493893606766283
        self.mid_game_weight = 0.8086804118689379
        self.late_game_weight = 1.0147020543611225
        # Search parameters (optimized)
        self.alpha_beta_window = 106.27432861262218

    def make_move(self, state):
        valid_moves = [col for col in range(7) if state[0][col] == 0]
        
        # Quick check for winning moves and blocks
        for col in valid_moves:
            # Check for win
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.player_id
                if self._is_winning_move(temp_state, row, col, self.player_id):
                    return col
                temp_state[row][col] = 0
                
            # Check for block
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.opponent_id
                if self._is_winning_move(temp_state, row, col, self.opponent_id):
                    return col

        # Use minimax with alpha-beta pruning
        best_score = float('-inf')
        best_move = valid_moves[0]
        
        # Sort moves by column preference
        moves = sorted(valid_moves, key=lambda x: -abs(x-3))
        
        for col in moves:
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.player_id
                score = self.minimax(temp_state, self.depthMax-1, float('-inf'), float('inf'), False)
                if score > best_score:
                    best_score = score
                    best_move = col
                    
        return best_move

    def minimax(self, state, depth, alpha, beta, maximizing):
        # Use alpha-beta window for pruning
        if maximizing:
            alpha = max(alpha, -self.winning_score - self.alpha_beta_window)
        else:
            beta = min(beta, self.winning_score + self.alpha_beta_window)
            
        # Check transposition table
        key = hash(state.tobytes())
        if key in self.trans_table and self.trans_table[key][0] >= depth:
            return self.trans_table[key][1]
            
        if depth == 0:
            return self.evaluate_board(state)
            
        valid_moves = [col for col in range(7) if state[0][col] == 0]
        if not valid_moves:
            return 0
            
        if maximizing:
            max_eval = float('-inf')
            for col in sorted(valid_moves, key=lambda x: -abs(x-3)):
                row = self._get_row(state, col)
                if row != -1:
                    state[row][col] = self.player_id
                    if self._is_winning_move(state, row, col, self.player_id):
                        eval = 1000000 + depth
                    else:
                        eval = self.minimax(state, depth-1, alpha, beta, False)
                    state[row][col] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            
            # Store in transposition table
            if len(self.trans_table) >= self.table_size:
                self.trans_table.clear()
            self.trans_table[key] = (depth, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            for col in sorted(valid_moves, key=lambda x: -abs(x-3)):
                row = self._get_row(state, col)
                if row != -1:
                    state[row][col] = self.opponent_id
                    if self._is_winning_move(state, row, col, self.opponent_id):
                        eval = -1000000 - depth
                    else:
                        eval = self.minimax(state, depth-1, alpha, beta, True)
                    state[row][col] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                        
            # Store in transposition table
            if len(self.trans_table) >= self.table_size:
                self.trans_table.clear()
            self.trans_table[key] = (depth, min_eval)
            return min_eval

    def _get_row(self, state, col):
        """Fast method to get the next available row in a column"""
        for row in range(5, -1, -1):
            if state[row][col] == 0:
                return row
        return -1

    def _is_winning_move(self, state, row, col, player):
        """Optimized win check only around the last move"""
        # Horizontal
        count = 0
        for c in range(max(0, col-3), min(7, col+4)):
            if state[row][c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
                
        # Vertical
        if row <= 2:
            if all(state[row+i][col] == player for i in range(4)):
                return True
                
        # Diagonal checks
        for i in range(-3, 1):
            # Positive slope
            count = 0
            for j in range(4):
                if (0 <= row+i+j < 6 and 
                    0 <= col+j < 7 and 
                    state[row+i+j][col+j] == player):
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
                    
            # Negative slope
            count = 0
            for j in range(4):
                if (0 <= row+i+j < 6 and 
                    0 <= col-j < 7 and 
                    state[row+i+j][col-j] == player):
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
        return False

    def evaluate_board(self, state):
        # Determine game phase
        moves_played = np.count_nonzero(state)
        if moves_played < 12:
            phase_weight = self.early_game_weight
        elif moves_played < 24:
            phase_weight = self.mid_game_weight
        else:
            phase_weight = self.late_game_weight
            
        score = 0
        
        # Position weights with multiplier
        for i in range(6):
            for j in range(7):
                if state[i][j] == self.player_id:
                    score += self.position_weights[i][j] * self.position_weight_multiplier
                elif state[i][j] == self.opponent_id:
                    score -= self.position_weights[i][j] * self.position_weight_multiplier
        
        # Apply center column bias
        center_pieces = sum(1 for i in range(6) if state[i][3] == self.player_id)
        score += center_pieces * self.center_column_bias
        
        # Evaluate sequences with phase weight
        score += self._evaluate_sequences(state) * phase_weight
        
        return score

    def _evaluate_sequences(self, state):
        score = 0
        
        # Horizontal sequences
        for row in range(6):
            for col in range(4):
                window = list(state[row, col:col+4])
                score += self._evaluate_window(window) * self.horiFactor
                
        # Vertical sequences
        for row in range(3):
            for col in range(7):
                window = [state[row+i][col] for i in range(4)]
                score += self._evaluate_window(window) * self.vertFactor
                
        # Diagonal sequences
        for row in range(3):
            for col in range(4):
                # Positive slope
                window = [state[row+i][col+i] for i in range(4)]
                score += self._evaluate_window(window) * self.slantedFactor
                
                # Negative slope
                window = [state[row+3-i][col+i] for i in range(4)]
                score += self._evaluate_window(window) * self.slantedFactor
                
        return score

    def _evaluate_window(self, window):
        score = 0
        player_count = window.count(self.player_id)
        opponent_count = window.count(self.opponent_id)
        empty_count = window.count(0)
        
        if player_count == 4:
            score += 1000
        elif player_count == 3 and empty_count == 1:
            score += self.three_in_row_weight
        elif player_count == 2 and empty_count == 2:
            score += self.two_in_row_weight
            
        if opponent_count == 3 and empty_count == 1:
            score -= self.blocking_weight
            
        return score    
       
class AIAgent4(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.depthMax = 4
        
        # Initialize transposition table
        self.table_size = 2097152  # 2^21
        self.trans_table = {}
        
        # Core evaluation weights (from training)
        self.horiFactor = 4.9486535187305405
        self.vertFactor = 4.1757368687520735
        self.slantedFactor = 3.8749728594883255
        self.midFactor = 1.9125292117667416
        self.three_in_row_weight = 4.738284125286771
        self.two_in_row_weight = 3.276609006390295
        self.blocking_weight = 7.593380991830439
        
        # Position weights matrix (from training)
        self.position_weights = np.array([
            [ 3.25062201,  8.83850068, 10.17640399, 11.18262157, 11.36849489,  7.89687808, 4.29433897],
            [ 5.35118787,  7.38100207, 13.90026157, 10.70237208, 10.14818013, 10.58140278, 8.41015798],
            [10.58723994, 11.56236307, 14.07009324, 18.50440583, 13.63420154, 10.06405292, 10.10325427],
            [10.19756691,  8.53758915, 10.06566286, 14.38320764, 13.34962925, 12.90857527,  8.21822703],
            [ 5.2846437,   9.53027023, 12.51207457, 13.0227996,  10.51345871, 11.30517451,  6.84209545],
            [ 4.45275163,  7.91967906, 10.57032771, 12.44049109, 11.43235569,  8.36358703,  5.93425282]
        ], dtype=np.float32)
        
        # Enhanced evaluation parameters
        self.winning_score = 1000000
        self.position_weight_multiplier = 12.953615243372532
        self.center_column_bias = 1.7843278888652345
        
        # Game phase weights
        self.early_game_weight = 1.4535607049856767
        self.mid_game_weight = 1.5157791196205057
        self.late_game_weight = 1.6483383051722968
        
        # Search parameters
        self.alpha_beta_window = 136.6175557040026

    def make_move(self, state):
        valid_moves = [col for col in range(7) if state[0][col] == 0]
        moves_played = np.count_nonzero(state)
        
        # Center first strategy for opening moves
        if moves_played <= 1 and 3 in valid_moves:
            return 3
            
        # Quick check for winning moves and blocks
        for col in valid_moves:
            # Check for win
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.player_id
                if self._is_winning_move(temp_state, row, col, self.player_id):
                    return col
                temp_state[row][col] = 0
                
            # Check for block
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.opponent_id
                if self._is_winning_move(temp_state, row, col, self.opponent_id):
                    return col

        # Clear transposition table if too large
        if len(self.trans_table) >= self.table_size:
            self.trans_table.clear()

        # Use minimax with alpha-beta pruning
        best_score = float('-inf')
        best_move = valid_moves[0]
        
        # Sort moves by column preference
        moves = sorted(valid_moves, key=lambda x: -abs(x-3))
        
        for col in moves:
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = self.player_id
                score = self.minimax(temp_state, self.depthMax-1, float('-inf'), float('inf'), False)
                if score > best_score:
                    best_score = score
                    best_move = col
                    
        return best_move

    def minimax(self, state, depth, alpha, beta, maximizing):
        # Use alpha-beta window for pruning
        if maximizing:
            alpha = max(alpha, -self.winning_score - self.alpha_beta_window)
        else:
            beta = min(beta, self.winning_score + self.alpha_beta_window)
            
        # Check transposition table
        key = hash(state.tobytes())
        if key in self.trans_table:
            return self.trans_table[key]
            
        if depth == 0:
            return self.evaluate_position(state, self.player_id)
            
        valid_moves = [col for col in range(7) if state[0][col] == 0]
        if not valid_moves:
            return 0
            
        if maximizing:
            max_eval = float('-inf')
            for col in sorted(valid_moves, key=lambda x: -abs(x-3)):
                row = self._get_row(state, col)
                if row != -1:
                    state[row][col] = self.player_id
                    if self._is_winning_move(state, row, col, self.player_id):
                        eval = self.winning_score + depth
                    else:
                        eval = self.minimax(state, depth-1, alpha, beta, False)
                    state[row][col] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            
            self.trans_table[key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for col in sorted(valid_moves, key=lambda x: -abs(x-3)):
                row = self._get_row(state, col)
                if row != -1:
                    state[row][col] = self.opponent_id
                    if self._is_winning_move(state, row, col, self.opponent_id):
                        eval = -self.winning_score - depth
                    else:
                        eval = self.minimax(state, depth-1, alpha, beta, True)
                    state[row][col] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                        
            self.trans_table[key] = min_eval
            return min_eval

    def evaluate_position(self, state, player_id):
        score = 0
        opponent_id = 3 - player_id
        moves_played = np.count_nonzero(state)
        
        # Determine game phase weight
        if moves_played <= 12:
            phase_weight = self.early_game_weight
        elif moves_played <= 24:
            phase_weight = self.mid_game_weight
        else:
            phase_weight = self.late_game_weight
        
        # Fast position-based scoring using numpy operations
        player_positions = (state == player_id)
        opponent_positions = (state == opponent_id)
        position_score = (np.sum(self.position_weights * player_positions) -
                         np.sum(self.position_weights * opponent_positions))
        score += position_score * self.position_weight_multiplier * phase_weight
        
        # Directional sequence evaluation
        sequence_score = 0
        
        # Horizontal
        for row in range(6):
            for col in range(4):
                window = state[row, col:col+4]
                sequence_score += self._evaluate_window(window) * self.horiFactor
        
        # Vertical
        for row in range(3):
            for col in range(7):
                window = state[row:row+4, col]
                sequence_score += self._evaluate_window(window) * self.vertFactor
        
        # Diagonal
        for row in range(3):
            for col in range(4):
                # Positive slope
                window = np.array([state[row+i, col+i] for i in range(4)])
                sequence_score += self._evaluate_window(window) * self.slantedFactor
                
                # Negative slope
                window = np.array([state[row+3-i, col+i] for i in range(4)])
                sequence_score += self._evaluate_window(window) * self.slantedFactor
        
        score += sequence_score * phase_weight
        
        # Enhanced center control evaluation
        center_array = state[:, 3]
        center_count = np.count_nonzero(center_array == player_id)
        score += center_count * self.midFactor * self.center_column_bias
        
        return score

    def _evaluate_window(self, window):
        score = 0
        player_count = np.count_nonzero(window == self.player_id)
        empty_count = np.count_nonzero(window == 0)
        opponent_count = 4 - player_count - empty_count
        
        if player_count == 4:
            score += self.winning_score
        elif player_count == 3 and empty_count == 1:
            score += self.three_in_row_weight
        elif player_count == 2 and empty_count == 2:
            score += self.two_in_row_weight
        
        if opponent_count == 3 and empty_count == 1:
            score -= self.blocking_weight
        elif opponent_count == 2 and empty_count == 2:
            score -= self.two_in_row_weight * 0.5
            
        return score

    def _get_row(self, state, col):
        for row in range(5, -1, -1):
            if state[row][col] == 0:
                return row
        return -1

    def _is_winning_move(self, state, row, col, player):
        # Horizontal
        count = 0
        for c in range(max(0, col-3), min(7, col+4)):
            if state[row][c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
                
        # Vertical
        if row <= 2:
            count = 0
            for r in range(row, min(6, row+4)):
                if state[r][col] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    break
                
        # Diagonal (positive slope)
        count = 0
        for i in range(-3, 4):
            r = row - i
            c = col + i
            if 0 <= r < 6 and 0 <= c < 7:
                if state[r][c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
                    
        # Diagonal (negative slope)
        count = 0
        for i in range(-3, 4):
            r = row + i
            c = col + i
            if 0 <= r < 6 and 0 <= c < 7:
                if state[r][c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
                    
        return False
    


    def minimax(self, state, depth, alpha, beta, maximizing):
        # Use alpha-beta window for pruning
        if maximizing:
            alpha = max(alpha, -self.winning_score - self.alpha_beta_window)
        else:
            beta = min(beta, self.winning_score + self.alpha_beta_window)
            
        # Check transposition table
        key = hash(state.tobytes())
        if key in self.trans_table:
            return self.trans_table[key]
            
        if depth == 0:
            return self.evaluate_position(state, self.player_id)
            
        valid_moves = [col for col in range(7) if state[0][col] == 0]
        if not valid_moves:
            return 0
            
        if maximizing:
            max_eval = float('-inf')
            for col in sorted(valid_moves, key=lambda x: -abs(x-3)):
                row = self._get_row(state, col)
                if row != -1:
                    state[row][col] = self.player_id
                    if self._is_winning_move(state, row, col, self.player_id):
                        eval = self.winning_score + depth
                    else:
                        eval = self.minimax(state, depth-1, alpha, beta, False)
                    state[row][col] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            
            self.trans_table[key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for col in sorted(valid_moves, key=lambda x: -abs(x-3)):
                row = self._get_row(state, col)
                if row != -1:
                    state[row][col] = self.opponent_id
                    if self._is_winning_move(state, row, col, self.opponent_id):
                        eval = -self.winning_score - depth
                    else:
                        eval = self.minimax(state, depth-1, alpha, beta, True)
                    state[row][col] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                        
            self.trans_table[key] = min_eval
            return min_eval

    def evaluate_position(self, state, player_id):
        """Enhanced position evaluation using optimized weights"""
        score = 0
        opponent_id = 3 - player_id
        moves_played = np.count_nonzero(state)
        
        # Determine game phase weight
        if moves_played <= 12:
            phase_weight = self.early_game_weight
        elif moves_played <= 24:
            phase_weight = self.mid_game_weight
        else:
            phase_weight = self.late_game_weight
        
        # Fast position-based scoring using numpy operations
        player_positions = (state == player_id)
        opponent_positions = (state == opponent_id)
        position_score = (np.sum(self.position_weights * player_positions) -
                         np.sum(self.position_weights * opponent_positions))
        score += position_score * self.position_weight_multiplier * phase_weight
        
        # Directional sequence evaluation
        sequence_score = 0
        
        # Horizontal
        for row in range(6):
            for col in range(4):
                window = state[row, col:col+4]
                sequence_score += self._evaluate_window(window) * self.horiFactor
        
        # Vertical
        for row in range(3):
            for col in range(7):
                window = state[row:row+4, col]
                sequence_score += self._evaluate_window(window) * self.vertFactor
        
        # Diagonal
        for row in range(3):
            for col in range(4):
                # Positive slope
                window = np.array([state[row+i, col+i] for i in range(4)])
                sequence_score += self._evaluate_window(window) * self.slantedFactor
                
                # Negative slope
                window = np.array([state[row+3-i, col+i] for i in range(4)])
                sequence_score += self._evaluate_window(window) * self.slantedFactor
        
        score += sequence_score * phase_weight
        
        # Enhanced center control evaluation
        center_array = state[:, 3]
        center_count = np.count_nonzero(center_array == player_id)
        score += center_count * self.midFactor * self.center_column_bias
        
        return score

    def _evaluate_window(self, window):
        """Enhanced window evaluation with numpy operations"""
        score = 0
        player_count = np.count_nonzero(window == self.player_id)
        empty_count = np.count_nonzero(window == 0)
        opponent_count = 4 - player_count - empty_count
        
        # Pattern matching with optimized weights
        if player_count == 4:
            score += self.winning_score
        elif player_count == 3 and empty_count == 1:
            score += self.three_in_row_weight
        elif player_count == 2 and empty_count == 2:
            score += self.two_in_row_weight
        
        # Enhanced blocking evaluation
        if opponent_count == 3 and empty_count == 1:
            score -= self.blocking_weight
        elif opponent_count == 2 and empty_count == 2:
            score -= self.two_in_row_weight * 0.5  # Partial blocking
            
        return score

    def _get_row(self, state, col):
        """Fast method to get the next available row in a column"""
        for row in range(5, -1, -1):
            if state[row][col] == 0:
                return row
        return -1

    def _is_winning_move(self, state, row, col, player):
        """Optimized win check only around the last move"""
        # Horizontal
        count = 0
        for c in range(max(0, col-3), min(7, col+4)):
            if state[row][c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
                
        # Vertical
        if row <= 2:
            count = 0
            for r in range(row, min(6, row+4)):
                if state[r][col] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    break
                
        # Diagonal (positive slope)
        count = 0
        for i in range(-3, 4):
            r = row - i
            c = col + i
            if 0 <= r < 6 and 0 <= c < 7:
                if state[r][c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
                    
        # Diagonal (negative slope)
        count = 0
        for i in range(-3, 4):
            r = row + i
            c = col + i
            if 0 <= r < 6 and 0 <= c < 7:
                if state[r][c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
                    
        return False

    def _evaluate_threats(self, state, player_id):
        """New method for detecting potential threats and opportunities"""
        threat_score = 0
        opponent_id = 3 - player_id
        
        # Check for double threats (two winning moves)
        valid_moves = [col for col in range(7) if state[0][col] == 0]
        winning_moves = 0
        
        for col in valid_moves:
            temp_state = state.copy()
            row = self._get_row(temp_state, col)
            if row != -1:
                temp_state[row][col] = player_id
                if self._is_winning_move(temp_state, row, col, player_id):
                    winning_moves += 1
        
        # Bonus for having multiple winning moves
        if winning_moves >= 2:
            threat_score += self.winning_score * 0.5
            
        return threat_score
    
class AIAgentBest(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.depthMax = 4
        self.opponent_id = 3 - player_id
        
        # Initialize transposition table and its size limit
        self.transposition_table = {}
        self.max_table_size = 1000000
        
        # Optimized weights from search
        self.horiFactor = 2.0371364039383266
        self.vertFactor = 2.7563175087980665
        self.slantedFactor = 3.9140122872297605
        self.midFactor = 0.6118851749256868
        
        # Optimized position weights from search
        self.position_weights = np.array([
            [ 1.23860216,  2.73850437, 13.62546509,  6.12342321,  3.25478926,  7.59300562,  8.04716999],
            [ 9.16905538,  3.81054018,  9.69596729, 12.32434684,  2.41487186,  8.72575823, 10.14652539],
            [11.39534062,  2.85866986,  6.07071602,  7.77056776,  2.36995105, 11.98727083,  1.90939228],
            [14.11700105,  1.47386146, 11.67853867, 12.07477527,  3.86484734,  9.3666596,  12.76659639],
            [10.04788352,  6.82449552,  6.25616973, 10.22638499,  4.53091542, 11.45803135,  7.84748538],
            [ 5.75554054,  4.63648117,  1.63330726, 11.40432902,  5.75804151,  8.61651056,  6.31939364]
        ], dtype=np.float32)

    def make_move(self, state):
        valid_moves = get_valid_col_id(state)
        moves_played = np.count_nonzero(state)
        
        # Center first strategy for opening moves
        if moves_played <= 1 and 3 in valid_moves:
            return 3
            
        # Quick check for winning moves and blocks
        for col in valid_moves:
            temp_state = state.copy()
            
            # Check for win
            step(temp_state, col, self.player_id, in_place=True)
            if is_win(temp_state):
                return col
                
            # Check for block
            temp_state = state.copy()
            step(temp_state, col, self.opponent_id, in_place=True)
            if is_win(temp_state):
                return col
        
        # Clear transposition table if too large
        if len(self.transposition_table) > self.max_table_size:
            self.transposition_table.clear()
        
        best_col, _ = self.minimax(state, self.depthMax, float('-inf'), float('inf'), True)
        return best_col

    def evaluate_position(self, state, player_id):
        score = 0
        opponent_id = 3 - player_id
        
        # Fast position-based scoring using numpy operations
        player_positions = (state == player_id)
        opponent_positions = (state == opponent_id)
        score += np.sum(self.position_weights * player_positions) * 10
        score -= np.sum(self.position_weights * opponent_positions) * 10
        
        # Evaluate horizontal windows (optimized)
        for row in range(6):
            for col in range(4):
                window = state[row, col:col+4]
                score += self.evaluate_window(window, player_id) * self.horiFactor
        
        # Evaluate vertical windows (optimized)
        for row in range(3):
            for col in range(7):
                window = state[row:row+4, col]
                score += self.evaluate_window(window, player_id) * self.vertFactor
        
        # Evaluate diagonal windows
        for row in range(3):
            for col in range(4):
                # Positive slope diagonal
                window = [state[row+i, col+i] for i in range(4)]
                score += self.evaluate_window(window, player_id) * self.slantedFactor
                
                # Negative slope diagonal
                window = [state[row+3-i, col+i] for i in range(4)]
                score += self.evaluate_window(window, player_id) * self.slantedFactor
        
        # Quick center control evaluation
        center_array = state[:, 3]
        center_count = np.count_nonzero(center_array == player_id)
        score += center_count * self.midFactor * 6
        
        return score

    def evaluate_window(self, window, player_id):
        """Optimized window evaluation"""
        score = 0
        player_count = np.count_nonzero(window == player_id)
        empty_count = np.count_nonzero(window == 0)
        opponent_count = 4 - player_count - empty_count
        
        # Fast pattern matching
        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += 5
        elif player_count == 2 and empty_count == 2:
            score += 2
        
        if opponent_count == 3 and empty_count == 1:
            score -= 4
        
        return score
    def minimax(self, state, depth, alpha, beta, maximizing_player):
        state_hash = hash(state.tobytes())
        if state_hash in self.transposition_table and self.transposition_table[state_hash][0] >= depth:
            return self.transposition_table[state_hash][1:]
            
        valid_moves = get_valid_col_id(state)
        
        if depth == 0 or len(valid_moves) == 0:
            return (None, self.evaluate_position(state, self.player_id))
        
        # Quick move ordering for better pruning
        if maximizing_player:
            value = float('-inf')
            column = valid_moves[0]
            
            # Prioritize center and near-center columns
            ordered_moves = sorted(valid_moves, key=lambda x: -abs(x-3))
            
            for col in ordered_moves:
                temp_state = state.copy()
                step(temp_state, col, self.player_id, in_place=True)
                
                new_score = self.minimax(temp_state, depth-1, alpha, beta, False)[1]
                
                if new_score > value:
                    value = new_score
                    column = col
                
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            
            self.transposition_table[state_hash] = (depth, column, value)
            return column, value
        
        else:
            value = float('inf')
            column = valid_moves[0]
            
            ordered_moves = sorted(valid_moves, key=lambda x: -abs(x-3))
            
            for col in ordered_moves:
                temp_state = state.copy()
                step(temp_state, col, 3-self.player_id, in_place=True)
                
                new_score = self.minimax(temp_state, depth-1, alpha, beta, True)[1]
                
                if new_score < value:
                    value = new_score
                    column = col
                
                beta = min(beta, value)
                if alpha >= beta:
                    break
            
            self.transposition_table[state_hash] = (depth, column, value)
            return column, value
        
class AIAgentGabriel(object):
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
                                  4, 3, 3, 3, 3, 3, 2,]
        self.HORIZONTAL_MULTIPLIER = 2.5
        self.VERTICAL_MULTIPLIER = 1.5
        self.DIAGONAL_MULTIPLIER = 2
        self.CENTER_MULTIPLIER = 0.25
        self.MAX_DEPTH = 3
        self.MAX_VALUE = 999999

    def get_optimal_move(self, depth):
        ind = 0
        for i in range(depth):
            ind += 7 ** i
        return self.OPTIMAL_MOVES[ind]
    
    def get_state_score(self, state):
        """
        Returns the score of the current state of the game board.

        Parameters:
        -----------
        state : np.ndarray
            A 2D numpy array representing the current state of the game board.


        Returns:
        --------
        int
            The score of the current state of the game board.
        """
        """ YOUR CODE HERE """
        if is_win(state):
            return self.MAX_VALUE
        
        score = 0
        def get_horizontal(player):
            score = 0
            row = 0
            while row < 6:
                col = 0
                cur = 0
                while col < 7:
                    if state[row, col] == player:
                        cur += 1
                    else:
                        cur = 0
                    if cur == 3:
                        add_score = 0
                        if col > 2 and state[row, col - 3] == 0:
                            add_score += 1
                        if col < 6 and state[row, col + 1] == 0:
                            add_score += 1
                        score += add_score * self.HORIZONTAL_MULTIPLIER
                        if col > 3:
                            break
                        col += 1
                    col += 1
                row += 1
            return score
        
        score += get_horizontal(self.player_id)
        score -= get_horizontal([2, 1][self.player_id - 1])
        def get_vertical(player):
            col = 0
            score = 0
            while col < 7:
                row = 5
                cur = 0
                while row >= 0:
                    if state[row, col] == player:
                        cur += 1
                    else:
                        break
                    if cur == 3:
                        add_score = 1
                        score += add_score * self.VERTICAL_MULTIPLIER
                        col += 1
                        break
                    row -= 1
                col += 1
            return score
        score += get_vertical(self.player_id)
        score -= get_vertical([2, 1][self.player_id - 1])
        def get_diagonal(player):
            score = 0
            row = 3
            def iterate_diagonal(r, c, right):
                score = 0
                cur = 0
                while r < 6 and c >= 0 and c < 7:
                    if state[r, c] == player:
                        cur += 1
                    else:
                        cur = 0
                    if cur == 3:
                        add_score = 0
                        if r < 5 and c + right >= 0 and c + right < 7 and state[r + 1, c + right] == 0:
                            add_score += 1
                        if r > 2 and c - 3 * right >= 0 and c - 3 * right < 7 and state[r - 3, c - 3 * right] == 0:
                            add_score += 1
                        score += add_score * self.DIAGONAL_MULTIPLIER
                        if r > 3:
                            break
                        c += 1
                        r += 1
                    r += 1
                    c += 1
                return score
            while row >= 0:
                score += iterate_diagonal(row, 0, 1)
                score += iterate_diagonal(row, 6, -1)
                row -= 1
            col = 1
            while col < 4:
                score += iterate_diagonal(0, col, 1)
                col += 1
            col = 5
            while col > 2:
                score += iterate_diagonal(0, col, -1)
                col -= 1
            return score
        score += get_diagonal(self.player_id)
        score -= get_diagonal([2, 1][self.player_id - 1])
        return score
        """ YOUR CODE END HERE """


    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return self.get_state_score(state), None
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
            return min_eval, best_move
        
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
        depth = int((np.count_nonzero(state) + 1 - self.player_id) / 2)
        if depth < self.OPTIMAL_LIMIT:
            move = self.get_optimal_move(depth)
            return move
        possible_moves = get_valid_col_id(state)
        best_score = -2 * self.MAX_VALUE
        best_move = None
        for move in possible_moves:
            new_state = step(state, move, self.player_id, False)
            score, _ = self.alpha_beta_pruning(new_state, self.MAX_DEPTH, -self.MAX_VALUE, self.MAX_VALUE, False)
            score += (3 - abs(move - 3)) * self.CENTER_MULTIPLIER
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
        """ YOUR CODE END HERE """
    
    
class AIAgentTestTransposition(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        
        # Keep original opening book
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
                                  4, 3, 3, 3, 3, 3, 2,]
        
        # Keep original multipliers
        self.HORIZONTAL_MULTIPLIER = 3.436
        self.VERTICAL_MULTIPLIER = 2.271
        self.DIAGONAL_MULTIPLIER = 2.146
        self.CENTER_MULTIPLIER = 0.259
        
        self.MAX_DEPTH = 4
        self.MAX_VALUE = 999999
        
        # Add transposition table
        self.transposition_table = {}
        self.max_table_size = 59377

    def get_optimal_move(self, depth):
        ind = 0
        for i in range(depth):
            ind += 7 ** i
        return self.OPTIMAL_MOVES[ind]

    def get_state_score(self, state):
        if is_win(state):
            return self.MAX_VALUE
            
        score = 0
        
        def get_horizontal(player):
            score = 0
            row = 0
            while row < 6:
                col = 0
                cur = 0
                while col < 7:
                    if state[row, col] == player:
                        cur += 1
                    else:
                        cur = 0
                    if cur == 3:
                        add_score = 0
                        if col > 2 and state[row, col - 3] == 0:
                            add_score += 1
                        if col < 6 and state[row, col + 1] == 0:
                            add_score += 1
                        score += add_score * self.HORIZONTAL_MULTIPLIER
                    col += 1
                row += 1
            return score
            
        score += get_horizontal(self.player_id)
        score -= get_horizontal([2, 1][self.player_id - 1])
        
        def get_vertical(player):
            col = 0
            score = 0
            while col < 7:
                row = 5
                cur = 0
                while row >= 0:
                    if state[row, col] == player:
                        cur += 1
                    else:
                        break
                    if cur == 3:
                        add_score = 1
                        score += add_score * self.VERTICAL_MULTIPLIER
                        col += 1
                        break
                    row -= 1
                col += 1
            return score
            
        score += get_vertical(self.player_id)
        score -= get_vertical([2, 1][self.player_id - 1])
        
        def get_diagonal(player):
            row = 5
            score = 0
            
            def iterate_diagonal(row, col, right):
                score = 0
                r = row
                c = col
                cur = 0
                while r < 6 and c >= 0 and c < 7:
                    if state[r, c] == player:
                        cur += 1
                    else:
                        cur = 0
                    if cur == 3:
                        add_score = 0
                        if r < 5 and c + right >= 0 and c + right < 7 and state[r + 1, c + right] == 0:
                            add_score += 1
                        if r > 2 and c - 3 * right >= 0 and c - 3 * right < 7 and state[r - 3, c - 3 * right] == 0:
                            add_score += 1
                        score += add_score * self.DIAGONAL_MULTIPLIER
                        if r > 3:
                            break
                        c += 1
                        r += 1
                    r += 1
                    c += 1
                return score
                
            while row >= 0:
                score += iterate_diagonal(row, 0, 1)
                score += iterate_diagonal(row, 6, -1)
                row -= 1
            col = 1
            while col < 4:
                score += iterate_diagonal(0, col, 1)
                col += 1
            col = 5
            while col > 2:
                score += iterate_diagonal(0, col, -1)
                col -= 1
            return score
            
        score += get_diagonal(self.player_id)
        score -= get_diagonal([2, 1][self.player_id - 1])
        return score

    def alpha_beta_pruning(self, state, depth, alpha, beta, maximizing_player):
        """Enhanced alpha-beta pruning with transposition table"""
        # Transposition table lookup
        state_hash = hash(state.tobytes())
        if state_hash in self.transposition_table:
            stored_depth, stored_value = self.transposition_table[state_hash]
            if stored_depth >= depth:
                return stored_value, None

        if depth == 0:
            value = self.get_state_score(state)
            self.transposition_table[state_hash] = (depth, value)
            return value, None
            
        if is_win(state):
            return -self.MAX_VALUE if maximizing_player else self.MAX_VALUE, None
            
        valid_moves = get_valid_col_id(state)
        # Sort moves to prioritize center columns
        valid_moves = sorted(valid_moves, key=lambda x: -abs(x-3))
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
            # Store in transposition table
            self.transposition_table[state_hash] = (depth, max_eval)
            return max_eval, best_move
        else:
            min_eval = self.MAX_VALUE
            for move in valid_moves:
                new_state = step(state, move, player, False)
                eval, _ = self.alpha_beta_pruning(new_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            # Store in transposition table
            self.transposition_table[state_hash] = (depth, min_eval)
            # Clear table if too large
            if len(self.transposition_table) > self.max_table_size:
                self.transposition_table.clear()
            return min_eval, best_move

    def make_move(self, state):
        # Keep original move selection with opening book
        depth = int((np.count_nonzero(state) + 1 - self.player_id) / 2)
        if depth < self.OPTIMAL_LIMIT:
            return self.get_optimal_move(depth)
            
        # Check for immediate wins/blocks
        valid_moves = get_valid_col_id(state)
        for move in valid_moves:
            # Check winning move
            new_state = step(state, move, self.player_id, False)
            if is_win(new_state):
                return move
            # Check blocking move
            new_state = step(state, move, [2, 1][self.player_id - 1], False)
            if is_win(new_state):
                return move
        
        # Regular search with center preference
        best_score = -2 * self.MAX_VALUE
        best_move = None
        for move in valid_moves:
            new_state = step(state, move, self.player_id, False)
            score, _ = self.alpha_beta_pruning(new_state, self.MAX_DEPTH, -self.MAX_VALUE, self.MAX_VALUE, False)
            score += (3 - abs(move - 3)) * self.CENTER_MULTIPLIER
            if score > best_score:
                best_score = score
                best_move = move
        return best_move


class AIAgent5(object):
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

        # Search parameters
        self.MAX_DEPTH = 4
        self.MAX_VALUE = 999999
        
        # Efficient transposition table
        self.transposition_table = {}
        self.max_table_size = 50000
        self.clear_threshold = 40000
        

    def get_optimal_move(self, depth):
        """Get move from opening book"""
        ind = 0
        for i in range(depth):
            ind += 7 ** i
        return self.OPTIMAL_MOVES[ind]


    def evaluate_position(self, state):
        """Enhanced position evaluation with fork detection and advanced threat detection"""
        # Quick win check
        if is_win(state):
            return self.MAX_VALUE
                
        score = 0
        
        def count_threats(player):
            threat_score = 0
            immediate_threats = 0
            potential_threats = 0
            
            # Horizontal, vertical, and diagonal checks
            for row in range(6):
                for col in range(7):
                    if col <= 3:  # Horizontal patterns
                        window = state[row, col:col+4]
                        player_count = np.count_nonzero(window == player)
                        empty_count = np.count_nonzero(window == 0)
                        
                        # Immediate 3-in-a-row with one empty slot
                        if player_count == 3 and empty_count == 1:
                            immediate_threats += 1
                            threat_score += 100
                        # Potential 2-in-a-row with two empty slots
                        elif player_count == 2 and empty_count == 2:
                            potential_threats += 1
                            threat_score += 10
                        # Advanced split threat like X _ X
                        elif player_count == 2 and empty_count == 1 and window[1] == 0 and window[2] == player:
                            threat_score += 50
                        
                    if row <= 2:  # Vertical patterns
                        window = state[row:row+4, col]
                        player_count = np.count_nonzero(window == player)
                        empty_count = np.count_nonzero(window == 0)
                        
                        if player_count == 3 and empty_count == 1:
                            immediate_threats += 1
                            threat_score += 100
                        elif player_count == 2 and empty_count == 2:
                            potential_threats += 1
                            threat_score += 10
                        
                    if row <= 2 and col <= 3:  # Diagonal patterns (positive slope)
                        window = [state[row+i, col+i] for i in range(4)]
                        player_count = sum(1 for x in window if x == player)
                        empty_count = sum(1 for x in window if x == 0)
                        
                        if player_count == 3 and empty_count == 1:
                            immediate_threats += 1
                            threat_score += 100
                        elif player_count == 2 and empty_count == 2:
                            potential_threats += 1
                            threat_score += 10
                        elif player_count == 2 and empty_count == 1 and window[1] == 0 and window[2] == player:
                            threat_score += 50  # Advanced split diagonal threat

                    if row >= 3 and col <= 3:  # Diagonal patterns (negative slope)
                        window = [state[row-i, col+i] for i in range(4)]
                        player_count = sum(1 for x in window if x == player)
                        empty_count = sum(1 for x in window if x == 0)
                        
                        if player_count == 3 and empty_count == 1:
                            immediate_threats += 1
                            threat_score += 100
                        elif player_count == 2 and empty_count == 2:
                            potential_threats += 1
                            threat_score += 10
                        elif player_count == 2 and empty_count == 1 and window[1] == 0 and window[2] == player:
                            threat_score += 50  # Advanced split diagonal threat

            # Return the calculated threat score along with threat counts for fork detection
            return threat_score, immediate_threats, potential_threats

        
        # Calculate threats for both players
        player_threats, player_immediate, player_potential = count_threats(self.player_id)
        opponent_threats, opponent_immediate, opponent_potential = count_threats(self.opponent_id)
        
        # Aggressive-defensive balance
        score += player_threats - opponent_threats * 1.2
        
        # Fork detection
        if player_immediate > 1:
            score += 200  # Reward for creating a fork
        if opponent_immediate > 1:
            score -= 200  # Penalty for allowing an opponent's fork
        
        # Center control bonus
        center_array = state[:, 3]
        center_score = np.count_nonzero(center_array == self.player_id) * 15
        score += center_score
        
        # Height penalty to prefer lower positions
        for col in range(7):
            col_height = np.count_nonzero(state[:, col])
            if col_height > 0:
                score -= (col_height * 2)  # Penalty for high stacks
        
        # Mobility bonus
        valid_moves = get_valid_col_id(state)
        score += len(valid_moves) * 5  # Bonus for keeping options open
        
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


class AIAgentGPT(object):
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.max_depth = 4
        
        # Optimized position weights for evaluation
        self.position_weights = np.array([
            [3, 4, 5, 7, 5, 4, 3],
            [4, 6, 8, 10, 8, 6, 4],
            [5, 8, 11, 13, 11, 8, 5],
            [5, 8, 11, 13, 11, 8, 5],
            [4, 6, 8, 10, 8, 6, 4],
            [3, 4, 5, 7, 5, 4, 3]
        ])
        
        # Transposition table with limits to manage memory usage
        self.transposition_table = {}
        self.max_table_size = 50000
        self.clear_threshold = 40000
        
        # Weight multipliers for different situations
        self.three_in_row_weight = 50
        self.two_in_row_weight = 10
        self.blocking_weight = 60
        self.center_weight = 3.5
        self.win_score = 100000
        
    def make_move(self, state):
        # Early moves: center column for best opening control
        moves_played = np.count_nonzero(state)
        if moves_played <= 1 and 3 in get_valid_col_id(state):
            return 3
        
        # Search using minimax with alpha-beta pruning
        _, best_move = self.minimax(state, self.max_depth, float('-inf'), float('inf'), True)
        return best_move

    def minimax(self, state, depth, alpha, beta, maximizing_player):
        state_hash = hash(state.tobytes())
        if state_hash in self.transposition_table:
            stored_depth, stored_score = self.transposition_table[state_hash]
            if stored_depth >= depth:
                return stored_score, None
        
        valid_moves = get_valid_col_id(state)
        if depth == 0 or len(valid_moves) == 0 or is_win(state):
            return self.evaluate_board(state), None
        
        if maximizing_player:
            max_eval = float('-inf')
            best_move = random.choice(valid_moves)  # Default to random choice if all evaluations are equal
            for move in valid_moves:
                new_state = step(state, move, self.player_id, in_place=False)
                eval_score, _ = self.minimax(new_state, depth-1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            self.transposition_table[state_hash] = (depth, max_eval)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = random.choice(valid_moves)
            for move in valid_moves:
                new_state = step(state, move, self.opponent_id, in_place=False)
                eval_score, _ = self.minimax(new_state, depth-1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            self.transposition_table[state_hash] = (depth, min_eval)
            return min_eval, best_move

    def evaluate_board(self, state):
        score = 0
        # Center control
        center_array = state[:, 3]
        center_count = np.count_nonzero(center_array == self.player_id)
        score += center_count * self.center_weight
        
        # Evaluate all possible windows
        score += self.evaluate_windows(state, self.player_id) 
        score -= self.evaluate_windows(state, self.opponent_id) * 1.1  # Slight bias for defensive play
        
        return score

    def evaluate_windows(self, state, player):
        score = 0
        
        # Horizontal windows
        for row in range(6):
            for col in range(4):
                window = state[row, col:col+4]
                score += self.evaluate_window(window, player)
                
        # Vertical windows
        for col in range(7):
            for row in range(3):
                window = state[row:row+4, col]
                score += self.evaluate_window(window, player)
        
        # Diagonal windows (positive and negative slopes)
        for row in range(3):
            for col in range(4):
                # Positive slope diagonal
                window = [state[row+i, col+i] for i in range(4)]
                score += self.evaluate_window(window, player)
                
                # Negative slope diagonal
                window = [state[row+3-i, col+i] for i in range(4)]
                score += self.evaluate_window(window, player)
        
        return score
    
    def evaluate_window(self, window, player):
        score = 0
        opponent = 3 - player
        player_count = np.count_nonzero(window == player)
        empty_count = np.count_nonzero(window == 0)
        opponent_count = np.count_nonzero(window == opponent)
        
        if player_count == 4:
            score += self.win_score
        elif player_count == 3 and empty_count == 1:
            score += self.three_in_row_weight
        elif player_count == 2 and empty_count == 2:
            score += self.two_in_row_weight
        
        # Prioritize blocking opponent three-in-row opportunities
        if opponent_count == 3 and empty_count == 1:
            score -= self.blocking_weight
        
        return score
