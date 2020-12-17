from MCTS import MCTS
from .base_agent import BaseAgent
import pygame
import sys
#from .mess import mess

class Master(BaseAgent):
	def __init__(self, color = "black", rows_n = 8, cols_n = 8, width = 600, height = 600):
		super().__init__(color, rows_n, cols_n, width, height)
		self.mcts = MCTS(2000) # num_sim

		#self.mess = mess()

	def step(self, reward, obs):
		# if self.mess.progress(obs) > 48:
		#	return self.mess.step(0, obs)
		#else:
		x, y = self.mcts.analysis(obs)
		#print(x,y)
		output = (self.col_offset + x * self.block_len, self.row_offset + y * self.block_len)
		return output, pygame.USEREVENT