from common import *

import numpy as np
import scipy.stats


class Uniform(Algo):
	def label(self):
		return "Uniform"

	def reset(self):
		self.K = self.config.K

	def action(self,t):
		return np.random.randint( self.K )


class TS_bern(Algo):
	def label(self):
		return "TS"

	def reset(self):
		self.K, self.T = self.config.K, self.config.T
		self.priors = np.array( self.config.priors )
		
		self.counts = np.zeros( self.K, dtype=np.int32 )
		self.rsums = np.zeros( self.K, dtype=np.int32 )

	def feedback(self, t, a, r):
		self.counts[a] += 1
		self.rsums[a] += r

	def action(self, t):
		mu_sampled = scipy.stats.beta.rvs( self.priors[:,0]+self.rsums, self.priors[:,1]+self.counts-self.rsums )

		return np.argmax( mu_sampled )

