from warnings import filterwarnings
filterwarnings('ignore')

import sys
import argparse
from src.data import getData
from src.callbacks import create_callbacks
from src.model import create_model
from src.optimizer import optimizer
from src.train import train
from src.prediction import prediction
from src.monteCarlo import monteCarlo
from src.portfolio import make_portfolio

from src.variables import tickers, period, observation_window, monte_carlo_simulation, \
                          seed, n_trials_optuna, epochs, risk_free_rate, verbose, \
                          minimum_participation, maximum_participation


def parse_args(args):

    parser = argparse.ArgumentParser(
        description="Search and Download GEDI L4A Granules",
        usage="gedi_l4a_search_download.py --doi <DOI> --date1 <start_date> --date2 <end_date> --poly <path_to_geojson_file> --outdir <path_to_directory>\n"
    )

    # parser.add_argument(
    #     "--doi",
    #     required=True, 
    #     type=check_doi, 
    #     help="DOI e.g., 10.3334/ORNLDAAC/2056 for GEDI L4A V2.1"
    # )

    return parser.parse_args(args)


# def check_doi(d: str):
#     try:
#         return d
    
#     except (ValueError, IndexError):
#         msg = "not a valid DOI"
#         raise argparse.ArgumentTypeError(msg)


def main():
    # parser = parse_args(sys.argv[1:])

    # no-download = parser.no-download
    # no-ai = parser.no-ai
    # no-train = parser.no-train
    # only-download = parser.only-download
    # only-train = parser.only-train


    if "--no-download" not in sys.argv:
        for ticker in tickers:
            getData(ticker, period)
            
    if "--only-download" in sys.argv:
        return

    if "--no-ai" not in sys.argv:
        
        if "--no-train" not in sys.argv:
            train(
                create_model,
                create_callbacks,
                optimizer,
                tickers,
                period,
                observation_window,
                seed,
                n_trials_optuna,
                epochs,
                graphics=True,
                verbose=verbose
            )
            
        if "--only-train" in sys.argv:
            return

        prediction(tickers, period, seed, observation_window, graphics=True)
        
        monteCarlo(
            tickers,
            observation_window,
            monte_carlo_simulation,
            seed,
            graphics=True
        )

        make_portfolio(tickers,
                       period,
                       seed,
                       risk_free_rate,
                       minimum_participation,
                       maximum_participation,
                       use_ia = True)
    
    else:
        make_portfolio(tickers,
                       period,
                       seed,
                       risk_free_rate,
                       minimum_participation, 
                       maximum_participation,
                       use_ia = False
                       )

if __name__ == '__main__':
    main()
