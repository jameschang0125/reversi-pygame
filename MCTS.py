import math
import random

class fake():
	def __init__(self):
		self.directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

	def norm(self, inp):
		return [[inp[8*j+i] for i in range(8)] for j in range(8)]

	def update(self, bin, move): # move = (x,y) # self = black = -1
		# check validity
		b = [[-bin[i][j] for j in range(8)] for i in range(8)] # note: include canonical
		x, y = move
		for dx, dy in self.directions:
			u, v = x, y
			while True:
				u += dx; v += dy
				L = []
				if 0<=u<=7 and 0<=v<=7:
					if b[v][u] == -1:
						L.append((v,u))
					elif b[v][u] == -1:
						for i, j in L:
							b[i][j] = 1
					else: break
				else: break
		b[y][x] = 1
		return b

	def no_move(self, b):
		return [[-b[i][j] for j in range(8)] for i in range(8)]

	def num_fliped(self, b, move): # 0 means invalid move
		x, y = move
		if b[y][x] != 0: return 0
		count = 0
		for dx, dy in self.directions:
			u, v = x, y
			c = 0
			while True:
				u += dx; v += dy
				if 0<=u<=7 and 0<=v<=7:
					if b[v][u] == 1:
						c += 1
					elif b[v][u] == -1:
						count += c
						break
					else: break
				else: break
		return count

	def count(self, b):
		black, white = 0, 0
		for i in range(8):
			for j in range(8):
				if b[i][j] == -1: black += 1
				if b[i][j] == 1: white += 1
		return (black, white)

	def debug(self, b):
		print('===============')
		for i in range(8):
			print(' '.join(['-' if b[i][j]==0 else ('O' if b[i][j] == -1 else 'X') for j in range(8)]))
		print('===============')

class MCTS():
	def __init__(self, num_sim = 2000):
		self.game = fake()

		self.Vsa = {}        # stores value function
		self.Nsa = {}       # stores #times edge s,a was visited
		self.Ns = {}        # stores #times board s was visited
		self.Ps = {}        # stores initial policy

		self.Es = {}        # stores game.getGameEnded ended for board s


		self.edgesq = [(2,0),(3,0),(4,0),(5,0),(2,7),(3,7),(4,7),(5,7),(0,2),(0,3),(0,4),(0,5),(7,2),(7,3),(7,4),(7,5)]
		self.adj = [(1,2),(1,3),(1,4),(1,5),(6,2),(6,3),(6,4),(6,5),(2,1),(3,1),(4,1),(5,1),(2,6),(3,6),(4,6),(5,6)]
		self.distance = {} # a dict that stores the distance to the corner, maximally 6
		self.danger = {(0,0):[(1,0),(0,1)], (0,7):[(1,7),(0,6)], (7,0):[(7,1),(6,0)], (7,7):[(7,6),(6,7)]}
		self.danger3 = {(0,0):(1,1), (0,7):(1,6), (7,0):(6,1), (7,7):(6,6)}
		self.corner = [(0,0),(0,7),(7,0),(7,7)]
		#===============================parameters===============================
		self.breadth = 0.5
		self.breadth_late = 1.2
		self.edge_buff = 2
		self.corner_buff = 1.15
		self.num_sim = num_sim
		self.magic = 1.2 # the parameter C_puct

	def sum2D(self, l):
		a = 0
		for i in range(8):
			a += sum(l[i])
		return a
	
	def p(self, board):
		n_vec = [[self.game.num_fliped(board, (i,j)) for j in range(8)] for i in range(8)]
		n_total = self.sum2D(n_vec)
		v_vec = [[1 if n_vec[i][j] > 0 else 0 for j in range(8)] for i in range(8)]
		v_total = self.sum2D(v_vec)
		if v_total == 0: return [[0] * 8] * 8, 0

		n_prime = [[1 / n_vec[i][j] if n_vec[i][j] > 0 else 0 for j in range(8)] for i in range(8)]
		np_total = self.sum2D(n_prime)
		black, white = self.game.count(board)
		step = black + white

		BB = (self.breadth * (64 - step) + self.breadth_late * step) / 64

		lowr = 0.2/v_total

		p_vec = [[0 if n_vec[i][j] == 0 else max(lowr, (1 - BB)*n_prime[i][j]/np_total + BB*v_vec[i][j]/v_total) for j in range(8)] for i in range(8)]
		for i, j in self.corner:
			p_vec[i][j] *= self.corner_buff
		for i, j in self.edgesq:
			p_vec[i][j] *= self.edge_buff
		p_total = self.sum2D(p_vec)
		return [[p_vec[i][j] / p_total for j in range(8)] for i in range(8)], v_total

			

	def convex(self, array):
		cmt = 0
		for j in range(8):
			a = [array[_] for _ in range(8)]
			for i in range(1, j+1): # j = 1~6
				if a[i] > a[i-1]: a[i] = a[i-1]
			for i in range(6, j-1, -1):
				if a[i] > a[i+1]: a[i] = a[i+1]
			s = sum(a)
			if cmt < s:
				cmt = s
		return cmt

	def sigmoid(self, v): # modified
		return 0.25 / (1 + math.exp(v * 0.4 - 1.5))

	def stable_cor(self, board, me):
		# add all corner up.
		score = 0
		count = 0
		top = 7
		for i in range(8):
			for j in range(8):
				if j > top: break
				if board[i][j] == me:
					count += 1
				else:
					top = j
					break
			if top == 0: break
		if count > 0:
			score += self.sigmoid(count)

		count = 0
		top = 7
		for i in range(8):
			for j in range(8):
				if j > top: break
				if board[7-i][j] == me:
					count += 1
				else:
					top = j
					break
			if top == 0: break
		if count > 0:
			score += self.sigmoid(count)

		count = 0
		top = 7
		for i in range(8):
			for j in range(8):
				if j > top: break
				if board[i][7-j] == me:
					count += 1
				else:
					top = 7-j
					break
			if top == 0: break
		if count > 0:
			score += self.sigmoid(count)

		count = 0
		top = 7
		for i in range(8):
			for j in range(8):
				if j > top: break
				if board[7-i][7-j] == me:
					count += 1
				else:
					top = 7-j
					break
			if top == 0: break
		if count > 0:
			score += self.sigmoid(count)

		return score

	def counter_cornering(self, board, me):
		score = 0
		if board[0][0] == board[0][7] == -me:
			count = 0
			for i in range(8):
				if board[0][i] == me:
					count += 1
				if board[0][i] == 0:
					break
			else:
				score += count / 6
		if board[7][0] == board[7][7] == -me:
			count = 0
			for i in range(8):
				if board[7][i] == me:
					count += 1
				if board[7][i] == 0:
					break
			else:
				score += count / 6
		if board[0][7] == board[7][7] == -me:
			count = 0
			for i in range(8):
				if board[i][7] == me:
					count += 1
				if board[i][7] == 0:
					break
			else:
				score += count / 6
		if board[0][0] == board[7][0] == -me:
			count = 0
			for i in range(8):
				if board[i][0] == me:
					count += 1
				if board[i][0] == 0:
					break
			else:
				score += count / 6
		return score

	def v(self, board, num_moves): # a value between -1 (lose) ~ 1 (win)
		# using some common knowledges

		# corner
		corner = - (board[0][0] + board[0][7] + board[7][0] + board[7][7]) /4

		# edges
		edge = 0
		for i, j in self.edgesq: edge += board[i][j]
		edge /= 16

		

		
		black, white = self.game.count(board)
		step = black + white - 4 # from 0 to 60

		# avoiding squares that are adjacent to the edge ?

		# "stable" discs and\or excluded from "minimize discs"?
		## double counting, every stable disc will be counted twice
		## count four sides

		stability = self.stable_cor(board, -1) - self.stable_cor(board, 1)


		# minimize our discs
		disc_score = (white - black) / (black + white + 1)

		# wedges? (HARD)

		# early, mid, lategame


		# danger square
		# avoiding squares that are adjacent to the corner
		danger = 0
		for i in self.danger:
			corner_piece = board[i[0]][i[1]]
			for j in self.danger[i]:
				if board[j[0]][j[1]] != corner_piece:
					danger += board[j[0]][j[1]]
			j = self.danger3[i]
			if board[j[0]][j[1]] != corner_piece:
				if corner_piece == 0:
					danger += 2.5 * board[j[0]][j[1]]
				else:
					danger += board[j[0]][j[1]]
		danger /= 12

		# mobility # this is a very non-explicive function
		mobility = (num_moves - 4) / 10


		#print("corner", '%.4f' % corner, "edge", '%.4f' % edge, "disc_score", '%.4f' % disc_score, "bad_sq", '%.4f' % bad_sq, "mobility", '%.4f' % mobility, "stability", '%.4f' % stability, "danger",  '%.4f' % danger)
		# wedge = self.counter_cornering(board, -1) - self.counter_cornering(board, 1)
		

		# early ~ late

		prog = step / 60
		'''
		w_mob = self.weight(prog, 0.08, 0)
		w_cor = self.weight(prog, 0.3, 0.18)
		w_edg = self.weight(prog, 0.19, 0.16)
		w_bad = self.weight(prog, 0.07, 0) # almost the same as "danger"
		w_sta = self.weight(prog, 0.03, 0.39)
		w_dis = self.weight(prog, 0.09, -0.03)
		w_dan = self.weight(prog, 0.19, 0.1)
		w_wed = self.weight(prog, 0.05, 0.2)
		'''
		w_mob = self.weight(prog, 0.1, 0)
		w_cor = self.weight(prog, 0.24, 0.2)
		w_edg = self.weight(prog, 0.24, 0.26)
		w_sta = self.weight(prog, 0.04, 0.45)
		w_dis = self.weight(prog, 0.11, -0.04)
		w_dan = self.weight(prog, 0.27, 0.13)



		total_score = w_mob * mobility + w_cor * corner + w_edg * edge + w_sta * stability + w_dis * disc_score + w_dan * danger # + w_wed * wedge
		if total_score > 1:
			total_score = 1

		return total_score

	def weight(self, prog, f, l):
		return f * (1 - prog) + l * prog

	def analysis(self, inp):
		canonicalBoard = self.game.norm(inp)
		a, b = self.game.count(canonicalBoard)
		if a + b <= 5:
			self.Vsa = {} 
			self.Nsa = {} 
			self.Ns = {} 
			self.Ps = {}
			self.Es = {} 
		#self.game.debug(canonicalBoard)
		s = str(canonicalBoard)
		for i in range(self.num_sim):
			self.search(canonicalBoard)
		cur_best = 0
		best_move = (0,0)
		for i in range(8):
			for j in range(8):
				if (s,(i,j)) in self.Nsa:
					if self.Nsa[(s,(i,j))] > cur_best:
						cur_best = self.Nsa[(s,(i,j))]
						best_move = (i,j)


		# debug
		p, num_moves = self.p(canonicalBoard)
		#self.v(canonicalBoard, num_moves, True)

		return best_move

	def search(self, canonicalBoard):
		#self.game.debug(canonicalBoard)

		s = str(canonicalBoard)
		if s not in self.Es:
			p, num_moves = self.p(canonicalBoard)
			if self.sum2D(p) == 0:
				if self.sum2D(self.p(self.game.no_move(canonicalBoard))[0]) == 0: # end game
					black, white = self.game.count(canonicalBoard)
					if black > white: self.Es[s] = 1 #?
					elif white > black: self.Es[s] = -1
					else: self.Es[s] = -0.00001
				else: self.Es[s] = 0
			else: self.Es[s] = 0
			self.Ns[s] = 0
		if self.Es[s]!=0: return -self.Es[s]

		if s not in self.Ps:
			self.Ps[s] = p
			return -self.v(canonicalBoard, num_moves)
			# self.Vsa[s] = v

		# pick the action with the highest upper confidence bound
		cur_best = -float('inf')
		best_act = (0,0)
		for i in range(8):
			for j in range(8):
				if self.Ps[s][i][j] > 0:
					a = (i,j)
					if (s,a) in self.Vsa:
						u = self.Vsa[(s,a)] + self.magic*self.Ps[s][i][j]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
					else:
						u = self.magic*self.Ps[s][i][j]*math.sqrt(self.Ns[s] + 0.1)
					if u > cur_best:
						cur_best = u
						best_act = a

		a = best_act
		next_board = self.game.update(canonicalBoard, a)

		v = self.search(next_board)

		if (s,a) in self.Vsa:
			self.Vsa[(s,a)] = (self.Nsa[(s,a)]*self.Vsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
			self.Nsa[(s,a)] += 1

		else:
			self.Vsa[(s,a)] = v
			self.Nsa[(s,a)] = 1

		self.Ns[s] += 1
		
		return -v

