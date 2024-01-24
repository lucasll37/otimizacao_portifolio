from typing import List, Dict

# Lista de tickers para os quais as análises serão feitas.
tickers: List[str] = [
    "ABEV3.SA",     # Ambev S.A
    # "ALOS3.SA",     # Aliansce Sonae Shopping Centers S.A
    "ALPA4.SA",     # Alpargatas S.A
    "ARZZ3.SA",     # Arezzo Indústria e Comércio S.A
    # "ASAI3.SA",     # Assaí Atacadista S.A
    "AZUL4.SA",     # Azul S.A
    "B3SA3.SA",     # B3 - Brasil Bolsa Balcão S.A
    "BBDC3.SA",     # Banco Bradesco S.A
    "BBDC4.SA",     # Banco Bradesco S.A
    "BBAS3.SA",     # Banco do Brasil S.A
    "BBSE3.SA",     # BB Seguridade Participações S.A
    "BEEF3.SA",     # Minerva S.A
    "BHIA3.SA",     # Bahema S.A
    "BPAC11.SA",    # Banco BTG Pactual S.A
    "BRAP4.SA",     # Bradespar S.A
    "BRFS3.SA",     # BRF S.A
    "BRKM5.SA",     # Braskem S.A
    "CCRO3.SA",     # CCR S.A
    "CIEL3.SA",     # Cielo S.A
    "CMIG4.SA",     # Cemig Companhia Energética de Minas Gerais
    # "CMIN3.SA",     # CSN Mineração S.A
    "COGN3.SA",     # Cogna Educação S.A
    "CPFE3.SA",     # CPFL Energia S.A
    "CPLE6.SA",     # Copel Companhia Paranaense de Energia
    "CRFB3.SA",     # Carrefour Brasil Grupo Carrefour Brasil
    "CSAN3.SA",     # Cosan S.A
    "CSNA3.SA",     # Companhia Siderúrgica Nacional
    "CVCB3.SA",     # CVC Brasil Operadora e Agência de Viagens S.A
    "CYRE3.SA",     # Cyrela Brazil Realty S.A Empreendimentos e Participações
    "DXCO3.SA",     # Dexco S.A
    "EGIE3.SA",     # Engie Brasil Energia S.A
    "ELET3.SA",     # Eletrobras Centrais Elétricas Brasileiras S.A
    "ELET6.SA",     # Eletrobras Centrais Elétricas Brasileiras S.A
    "EMBR3.SA",     # Embraer S.A
    "ENEV3.SA",     # Eneva S.A
    "ENGI11.SA",    # Engie Brasil Energia S.A
    "EQTL3.SA",     # Equatorial Energia S.A
    "EZTC3.SA",     # EZTEC Empreendimentos e Participações S.A
    "FLRY3.SA",     # Fleury S.A
    "GGBR4.SA",     # Gerdau S.A
    "GOAU4.SA",     # Metalúrgica Gerdau S.A
    "GOLL4.SA",     # Gol Linhas Aéreas Inteligentes S.A
    "HAPV3.SA",     # Hapvida Participações e Investimentos S.A
    "HYPE3.SA",     # Hypera Pharma
    # "IGTI11.SA",    # IGCT
    "IRBR3.SA",     # IRB Brasil Resseguros S.A
    "ITSA4.SA",     # Itaúsa - Investimentos Itaú S.A
    "ITUB4.SA",     # Itaú Unibanco Holding S.A
    "JBSS3.SA",     # JBS S.A
    "KLBN11.SA",    # Klabin S.A
    "LREN3.SA",     # Lojas Renner S.A
    # "LWSA3.SA",     # Locaweb Serviços de Internet S.A
    "MGLU3.SA",     # Magazine Luiza S.A
    "MRFG3.SA",     # Marfrig Global Foods S.A
    "MRVE3.SA",     # MRV Engenharia e Participações S.A
    "MULT3.SA",     # Multiplan Empreendimentos Imobiliários S.A
    # "NTCO3.SA",     # Natura &Co Holding S.A
    "PCAR3.SA",     # Grupo Pão de Açúcar
    "PETR3.SA",     # Petróleo Brasileiro S.A - Petrobras
    "PETR4.SA",     # Petróleo Brasileiro S.A - Petrobras
    # "PETZ3.SA",     # Petz
    # "PRIO3.SA",     # PetroRio S.A
    "RADL3.SA",     # Raia Drogasil S.A
    "RAIL3.SA",     # Rumo S.A
    # "RAIZ4.SA",     # Raízen S.A
    # "RDOR3.SA",     # Rede D'Or São Luiz S.A
    # "RECV3.SA",     # Rede D'Or São Luiz S.A
    "RENT3.SA",     # Localiza Rent a Car S.A
    # "RRRP3.SA",     # 3R Petroleum Óleo e Gás S.A
    "SANB11.SA",    # Banco Santander Brasil S.A
    "SBSP3.SA",     # Sabesp Companhia de Saneamento Básico do Estado de São Paulo
    "SLCE3.SA",     # SLC Agrícola S.A
    "SMTO3.SA",     # São Martinho S.A
    # "SOMA3.SA",     # Grupo Soma
    "SUZB3.SA",     # Suzano S.A
    "TAEE11.SA",    # Taesa Transmissora Aliança de Energia Elétrica S.A
    "TIMS3.SA",     # TIM S.A
    "TOTS3.SA",     # Totvs S.A
    "TRPL4.SA",     # Transmissão Paulista Companhia Paulista de Força e Luz
    "UGPA3.SA",     # Ultrapar Participações S.A
    "USIM5.SA",     # Usiminas Usinas Siderúrgicas de Minas Gerais S.A
    "VALE3.SA",     # Vale S.A
    # "VAMO3.SA",     # Vamos Locação de Caminhões, Máquinas e Equipamentos S.A
    # "VBBR3.SA",     # Vibra Energia S.A
    "VIVT3.SA",     # Telefônica Brasil S.A
    "WEGE3.SA",     # WEG S.A
    "YDUQ3.SA"      # Yduqs Participações S.A
]

# tickers: List[str] = [
#     "ABEV3.SA",     # Ambev S.A
# ]

# Dicionário definindo os períodos de interesse para análise.
period: Dict[str, str] = {
    # Formato: YYYY-MM-DD
    "start": "2017-01-08",
    "boundary": "2023-01-02",
    "end": "2023-07-03"
}

# Dicionário definindo a janela de observação para análise/modelagem.
observation_window: Dict[str, int] = {
    "stepsBack": 90,
    "stepsFoward": 15
}

# Semente para garantir a reprodutibilidade em processos aleatórios.
seed: int = 25

# Número de tentativas para otimização com Optuna.
n_trials_optuna: int = 0 # 432

# Número de épocas para treinamento do modelo.
epochs: int = 1000

# Número de simulações para a simulação de Monte Carlo.
monte_carlo_simulation: int = 250

# Nível de verbosidade para a saída de logs.
verbose: int = 1

# Retorno anual livre de risco (Selic).
# https://www.bcb.gov.br/estabilidadefinanceira/selicdadosdiarios
risk_free_rate: float = 0.1168 # Atualizado Dia 11/01/2024

# Mínimo de participação de um ativo na carteira.
minimum_participation: float = 0.02 # 2%

# Máximo de participação de um ativo na carteira.
maximum_participation: float = 0.10 # 10%
