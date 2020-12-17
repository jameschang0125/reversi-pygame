from MCTS import fake
from .base_agent import BaseAgent
import pygame
import sys
import random

class FasterRandom(BaseAgent): # for white
	def __init__(self, color = "black", rows_n = 8, cols_n = 8, width = 600, height = 600):
		super().__init__(color, rows_n, cols_n, width, height)
		self.game = fake()

	def sum2D(self, l):
		a = 0
		for i in range(8):
			a += sum(l[i])
		return a

	def step(self, reward, obs):
		board = self.game.norm(obs)
		board = self.game.no_move(board)
		n = [[self.game.num_fliped(board, (i,j)) for j in range(8)] for i in range(8)]
		valids = []
		for i in range(8):
			for j in range(8):
				if n[i][j] > 0:
					valids.append((i,j))
		#print('===choose random===')
		#self.game.debug(board)
		#print(valids)
		x, y = random.choice(valids)
		output = (self.col_offset + x * self.block_len, self.row_offset + y * self.block_len)
		return output, pygame.USEREVENT