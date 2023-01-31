import numpy as np
import scipy.stats
from matplotlib import pyplot


from common import *
from simul import *


def publish( config, algos, results, fig_dir, prefix ):
	K, T = config.K, config.T
	regretss = [ res.avg_cum_regrets() for res in results ]

	ts = 1 + np.arange(T)
	for algo,regrets in zip( algos, regretss ):
		pyplot.plot( ts, regrets, label=algo.label() )
	pyplot.xlabel( r"Time $T$" )
	pyplot.ylabel( r"Cumulative regret $\mathbb{E}[ \sum_{t=1}^T \mu^* - \mu_{A_t}]$" )
	pyplot.legend( loc='best' )
	pyplot.grid( True )
	pyplot.savefig( fig_dir + prefix + "_regret.pdf", bbox_inches='tight' )
	pyplot.close()



if __name__ == "__main__":

	# 1. load configs
	config_name = find_option("config")
	if config_name.endswith(".config"):
		config_name = config_name[ : -len(".config") ]

	config = Config( "configs/" + config_name + ".config" )
	res_dir = "results/" + config_name+"/"
	fig_dir = "figures/" + config_name+"/"

	ensure_path( fig_dir )
	prefix = config_name

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

	# 3. load
	results = []
	for algo in algos:
		result = Result( algo.name, algo.options, config )
		
		print( algo.name, end=" : " )
		for fn in os.listdir( res_dir ):
			if fn.endswith(".result") == False:
				continue
			parsed = fn.split("_")

			if parsed[0] == algo.name:
				print( fn, end=", " )
				result.load_and_merge( res_dir + "/" + fn )
		print()

		results.append( result )

	# 4. publish
	publish( config, algos, results, fig_dir, prefix )
