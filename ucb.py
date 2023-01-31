from common import *

import numpy as np
import scipy.stats


class UCB_bern(Algo):
	def label(self):
		return r"UCB($\beta=%.1f$)" % (self.options["beta"])

	def reset(self):
		self.K, self.T = self.config.K, self.config.T
		self.beta = self.options["beta"]

		self.counts = np.zeros( self.K, dtype=np.int32 )
		self.rsums = np.zeros( self.K, dtype=np.int32 )

	def feedback(self, t, a, r):
		self.counts[a] += 1
		self.rsums[a] += r

	def action(self, t):
		if t < self.K:
			return t
		return np.argmax( self.rsums/self.counts + self.beta / np.sqrt( self.counts ) )
