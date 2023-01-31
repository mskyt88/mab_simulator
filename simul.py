import numpy as np
import scipy.stats
import sklearn.metrics
import datetime

from common import *


class Simulator:
	def __init__(self, config):
		assert( config.problem in ["beta_bernoulli"] )

		self.config = config

	def generate(self, pid, sid):
		np.random.seed( seed = pid*10000 + sid )
		K, T = self.config.K, self.config.T

		if self.config.problem == "beta_bernoulli":
			priors = np.array( self.config.priors )
			mus = scipy.stats.beta.rvs( priors[:,0], priors[:,1] )
			rss = np.zeros( (T,K), dtype=np.int32 )
			for k in range(K):
				rss[:,k] = scipy.stats.bernoulli.rvs( mus[k], size=T )

		return mus, rss


	def simulate(self, algos, pid, callback, debug=False):
		last_report_timestamp = datetime.datetime.now()

		K, T = self.config.K, self.config.T
		S = self.config.S

		# 2. simulate MAB algorithms
		results = [ Result(algo.name, algo.options, self.config) for algo in algos ]

		for s in range(S):
			# 2-1. generate random reward realizations
			mus, rss = self.generate( pid, s )

			mu_max = mus.max()

			# 2-2. simulate MAB algorithms
			for ai,algo in enumerate(algos):
				np.random.seed( seed = pid*10000+s+1 )
				algo.reset()

				algo_rewards = []
				algo_regrets = []

				time_st = datetime.datetime.now()
				for t in range(T):
					a = algo.action(t)
					r = rss[t][a]
					algo.feedback( t, a, r )

					algo_rewards.append( mus[a] )
					algo_regrets.append( mu_max - mus[a] )

				time_en = datetime.datetime.now()
				
				results[ ai ].accumulate( algo_rewards, algo_regrets, (time_en-time_st).total_seconds() )

			# 2-3. print out progress of simulations
			if s%10 == 0 or debug == True:
				print()
				cum_regs = [ (res.avg_cum_regrets()[-1],ai) for ai,res in enumerate(results) ]
				cum_regs.sort( reverse=True )
				for cum_reg,ai in cum_regs:
					print( "%4d, %16s, %s" % (s, algos[ai].name, cum_reg) )
					
			# 2-4. archive the result (at least every five minutes)
			if (datetime.datetime.now() - last_report_timestamp).seconds > 10 or debug == True:
				callback( results )
				last_report_timestamp = datetime.datetime.now()

		callback( results )

		return results





if __name__ == "__main__":

	# 1. load configs
	config_name = find_option("config")
	if config_name.endswith(".config"):
		config_name = config_name[ : -len(".config") ]

	config = Config( "configs/" + config_name + ".config" )
	res_dir = "results/" + config_name+"/"
	ensure_path( res_dir )

	# 2. setup algorithms
	algos = []
	algo_options = []

	for cat, cmd in config.algorithms:
		category = find_option( "category", "default" )
		if category != "all" and cat.lower() != category.lower():
			continue

		exec( "import " + cmd[ :cmd.find(".") ] )
		name, option = eval( cmd[ cmd.find("(")+1 : -1 ], globals(), config.variables )
		algo_class = eval( cmd[ :cmd.find("(")] )
		algo = algo_class( name, config, option )

		only = find_option( "only", None )
		if only is not None and algo.name not in only.split(","):
			continue

		print( algo.name, option )
		algos.append( algo )
		algo_options.append( option )

	if len(algos) == 0:
		raise Exception( "no algorithm is configured" )

	# 3. simulate
	pid = int( find_option( "pid", "0" ) )
	debug = ("debug" in sys.argv)

	sim = Simulator( config )

	def archive(results):
		for algo,r in zip(algos,results):
			r.save( res_dir+algo.name+"_%d.result" % pid )
		print( "archived" )

	sim.simulate( algos, pid, archive, debug )
