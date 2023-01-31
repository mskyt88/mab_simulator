import sys, os
import numpy as np
import pickle


def ensure_path( path ):
	print( path )
	path = "./"+path
	ps = path.split("/")
	for i in range(1,len(ps)-1):
		pdir = "/".join( ps[:i] )
		if ps[i] not in os.listdir( pdir ):
			os.mkdir( pdir +"/"+ps[i] )


def find_option(key, default=None):
	for a in sys.argv:
		if a.startswith(key+"="):
			return a[ len(key)+1 : ]
	return default




class Config:
	def __init__(self, fn):
		self.algorithms = []
		self.variables = {}

		fd = open(fn)
		category = "DEFAULT"
		for line in fd:
			line = line.strip()
			if line.startswith("#"):
				continue
			elif line.endswith(":"):
				category = line[:-1].strip()
			elif line.startswith("algorithm"):
				self.algorithms.append( ( category, line.split("=")[1].strip() ) )
			elif line.strip() == "":
				continue
			else:
				exec( line, globals(), self.variables )
		fd.close()

	def __getattr__(self, key):
		if key == "algorithms":
			return self.algorithms
		elif key == "variables":
			return self.variables
		elif key in self.variables:
			return self.variables[key]
		return None





class Result:
	def __init__(self, name, options, config):
		self.name, self.options, self.config_variables = name, options, config.variables

		self.rewardss = []
		self.regretss = []

		self.eps = []
		
	def accumulate(self, rewards, regrets, elapsed_time):
		self.rewardss.append( rewards )
		self.regretss.append( regrets )
		self.eps.append( elapsed_time )

	def merge(self, another):
		self.rewardss.extend( another.rewardss )
		self.regretss.extend( another.regretss )
		self.eps.extend( another.eps )


	def avg_cum_regrets(self):
		return np.cumsum( np.mean( self.regretss, axis=0 ) )

	def save(self, fn):
		with open(fn, "wb") as fd:
			pickle.dump( self, fd )

	def load_and_merge(self, fn):
		with open(fn, "rb") as fd:
			ret = pickle.load( fd )
		self.config_variables = ret.config_variables
		self.merge( ret )



class Algo:
	def __init__(self, name, config, options):
		self.name = name
		self.config = config
		self.options = options

	def label(self):
		return self.name

	def reset(self):
		pass

	def action(self, t):
		return 0

	def feedback(self, t, a, r):
		pass