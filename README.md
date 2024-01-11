# Otimização de Portifólio
Este projeto integra aprendizado de máquina e otimização de Markowitz para construir carteiras de investimento eficientes, considerando riscos históricos e retornos esperados estimados por Inteligência Artificial.

## Descrição
O projeto foca na aplicação da fronteira de eficiência de Markowitz para otimizar carteiras de investimento, visando alcançar o maior índice Sharpe possível. Utiliza-se um conjunto de assets disponíveis para compor a carteira ideal, levando em consideração os riscos históricos e os retornos esperados na janela temporal aqui definida como stepsFoward. Os riscos são calculados com base no desempenho histórico dos assets na janela temporal definida como stepsBack.

O retorno esperado dos assets é estimado utilizando técnicas de inteligência artificial. Uma rede neural recorrente (RNN/LSTM) é empregada para capturar padrões comportamentais dos ativos financeiros no intervalo de tempo estipulado entre period['start'] e period['boundary']. Esses dados são utilizados para treinar e validar o modelo de previsão.

A janela de tempo de period['boundary'] a period['end'] é utilizada para testar a performance do modelo e para a seleção do modelo mais eficaz. A abordagem integrada de aprendizado de máquina e otimização financeira quantitativa permite uma análise mais profunda e uma seleção mais precisa dos componentes da carteira de investimentos, maximizando assim o retorno ajustado ao risco.

## Ambiente

- Python 3.11.x

## Instalação

### Criação e ativação do ambiente virtual

> Windows
  ```
  python -m venv env
  .\env\Scripts\activate
  ```	

> Linux ou macOS
```
python3 -m venv env
source env/bin/activate
```

### Instalar dependências
```bash
pip install -r ./requirements.txt
```

## Configuração de Variáveis

O arquivo `./src/variables.py` contém as variáveis que devem ser ajustadas para personalizar a análise. Abaixo estão as instruções para modificar cada variável:

### tickers
- `tickers`: Lista de tickers para os quais as análises serão feitas. Siga convenção do Yahoo Finance disponível [aqui](https://finance.yahoo.com/).
- Exemplo:
  ```python
  tickers: List[str] = ["EMBR3.SA", "BRPR3.SA", ...]
    ```

### period
- `period`: Dicionário definindo os períodos de interesse para análise. `period['boundary']` define o limite entre os períodos de treinamento e teste.
- Exemplo:
  ```python
  period: Dict[str, str] = {
      "start": "2013-07-02",
      "boundary": "2021-06-30",
      "end": "2023-07-01"
  }
    ```
### observation_window
- `observation_window`: Dicionário definindo a janela de observação para análise.
- Exemplo:
  ```python
  observation_window: Dict[str, int] = {
      "stepsBack": 240,
      "stepsFoward": 60
  }
  ```

### SEED
- `SEED`: Semente para garantir a reprodutibilidade em processos aleatórios.
- Exemplo:
  ```python
  SEED: int = 25
  ```

### n_trials_optuna
- `n_trials_optuna`: Número de tentativas para otimização com [Optuna](https://optuna.org/).
- Exemplo:
  ```python
  n_trials_optuna: int = 10
  ```

### epochs
- `epochs`: Número de épocas para treinamento do modelo.
- Exemplo:
  ```python
  epochs: int = 100000
  ```

### monte_carlo_simulation
- `monte_carlo_simulation`: Número de simulações para a simulação de Monte Carlo.
- Exemplo:
  ```python
  monte_carlo_simulation: int = 250
  ```

### verbose
- `verbose`: Nível de verbosidade para a saída de logs.
- Exemplo:
  ```python
  verbose: int = 1
  ```

### risk_free_rate
- `risk_free_rate`: Retorno anual livre de risco (Selic).
- Exemplo:
  ```python
  risk_free_rate: float = 0.1
  ```

### minimum_participation
- `minimum_participation`: Mínimo de participação de um ativo na carteira.
- Exemplo:
  ```python
  minimum_participation: float = 0.1
  ```


## Executar o projeto
```bash
python ./src/main.py
```

## Estrutura do Projeto
```
otimizacao_portifolio/
│
├── .history/
│
├── results/
│   ├── data/
│   ├── graphics/
│   ├── logs/
│   ├── portfolio/
│   ├── prediction/
│   ├── serialized objects/
│   └── trained models/
│
├── sqlite/
│   ├── *asset*/
│
├── src/
│   ├── __pycache__
│   ├── __init__.py
│   ├── callbacks.py
│   ├── model.py
│   ├── monteCarlo.py
│   ├── obtaining.py
│   ├── portfolio.py
│   ├── prediction.py
│   ├── train.py
│   └── variables.py
│
├── .gitignore
│
├── LICENSE
│
├── main.py
│
├── README.md
│
└── requirements.txt
```

### Descrição dos Diretórios e Arquivos:

- `.history/`: Contém arquivos de histórico gerados por algumas ferramentas de desenvolvimento.
- `results/`: Armazena os outputs do projeto, como dados processados, gráficos, logs, objetos serializados e modelos treinados.

    - `data/`: Dados financeiros históricos.
    - `graphics/`: Gráficos e visualizações gerados no treinamento.
    - `logs/`: Logs do treinamento (tensorboard).
    - `portfolio/`: Análises e resultados relacionados à otimização do portfólio.
    - `prediction/`: Previsões geradas pelos modelos.
    - `serialized objects/`: Objetos de transformação dos dados (serializados).
    - `trained models/`: Modelos de machine learning treinados e salvos.

- `sqlite/`: Diretório para o banco de dados SQLite, organizado por ativos.
- `src/`: Diretório que contém o código fonte do projeto, incluindo scripts para diferentes funcionalidades, como callbacks, modelagem, simulação de Monte Carlo, obtenção de dados, gerenciamento de portfólio, previsão e treinamento, além de variáveis de configuração.

  - `__pycache__/`: Contém arquivos de bytecode compilados que são criados pelo Python para acelerar a inicialização do programa.
  - `__init__.py`: Arquivo necessário para tratar o diretório como um módulo Python, possibilitando a importação de arquivos dentro deste diretório.
  - `callbacks.py`: Define funções de callback que podem ser usadas durante o treinamento de modelos para, por exemplo, salvar checkpoints, ajustar parâmetros ou monitorar o desempenho.
  - `model.py`: Inclui a definição e a configuração do modelo de machine learning utilizado no projeto.
  - `monteCarlo.py`: Implementa simulações de Monte Carlo, que podem ser usadas para projeções financeiras, análise de risco, etc.
  - `obtaining.py`: Responsável pela aquisição, limpeza e pré-processamento de dados.
  - `portfolio.py`: Contém lógicas relacionadas à gestão e análise de portfólios de investimentos.
  - `prediction.py`: Funções e scripts dedicados à realização de previsões com base no modelo treinado.
  - `train.py`: Script para treinar o modelo de machine learning, incluindo a definição de parâmetros, escolha de algoritmos e avaliação de desempenho.
  - `variables.py`: Arquivo para definir variáveis globais, constantes e configurações usadas em todo o projeto, como parâmetros de modelos, caminhos de arquivos, configurações de ambiente, etc.

- `.gitignore`: Define quais arquivos e diretórios devem ser ignorados pelo sistema de controle de versão Git.
- `LICENSE`: Contém a licença de uso e distribuição do projeto.
- `main.py`: O script principal para executar o projeto.
- `README.md`: Fornece uma visão geral do projeto, instruções de instalação e outras informações importantes.
- `requirements.txt`: Lista todas as bibliotecas necessárias para executar o projeto.

## Licença

Este projeto está sob a licença XYZ - veja o arquivo LICENSE para detalhes.