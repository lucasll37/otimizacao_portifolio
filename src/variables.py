from typing import List, Dict

# Lista de tickers para os quais as análises serão feitas.
tickers: List[str] = ["EMBR3.SA", "BRPR3.SA"]

# tickers: List[str] = ["EMBR3.SA", "BRPR3.SA", "CSNA3.SA", "EQTL3.SA", "CPLE6.SA", \
#                       "BBDC3.SA", "BRFS3.SA", "BBSE3.SA", "BRAP4.SA", "ELET3.SA", \
#                       "CPFE3.SA", "BBDC4.SA", "ABEV3.SA", "BBAS3.SA", "GGBR4.SA", \
#                       "ECOR3.SA", "CCRO3.SA", "CSAN3.SA", "CMIG4.SA", "CYRE3.SA", \
#                       "CIEL3.SA", "BRKM5.SA"]

# Dicionário definindo os períodos de interesse para análise.
period: Dict[str, str] = {
    "start": "2013-07-02",
    "boundary": "2021-06-30",
    "end": "2023-07-01"
}

# Dicionário definindo a janela de observação para análise/modelagem.
observation_window: Dict[str, int] = {
    "stepsBack": 240,
    "stepsFoward": 60
}

# Semente para garantir a reprodutibilidade em processos aleatórios.
SEED: int = 25

# Número de tentativas para otimização com Optuna.
n_trials_optuna: int = 1

# Número de épocas para treinamento do modelo.
epochs: int = 2

# Número de simulações para a simulação de Monte Carlo.
monte_carlo_simulation: int = 250

# Nível de verbosidade para a saída de logs.
verbose: int = 1

# Retorno anual livre de risco (Selic).
risk_free_rate: float = -0.1

# Mínimo de participação de um ativo na carteira.
minimum_participation: float = 0.05
