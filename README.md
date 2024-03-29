# Otimização de Portifólio
Este projeto integra aprendizado de máquina e otimização de Markowitz para construir carteiras de investimento eficientes, considerando riscos históricos e retornos esperados estimados por Inteligência Artificial.

## Descrição
O projeto foca na aplicação da fronteira de eficiência de Markowitz para otimizar carteiras de investimento, visando alcançar o maior índice Sharpe possível. Utiliza-se um conjunto de assets disponíveis para compor a carteira ideal, levando em consideração os riscos históricos e os retornos esperados na janela temporal aqui definida como *stepsFoward*. Os riscos são calculados com base no desempenho histórico dos assets na janela temporal definida como *stepsBack*.

O retorno esperado dos assets é estimado utilizando técnicas de inteligência artificial. Uma rede neural recorrente (RNN/LSTM) é empregada para capturar padrões comportamentais dos ativos financeiros no intervalo de tempo estipulado entre `period['start']` e `period['boundary']`. Esses dados são utilizados para treinar e validar o modelo de previsão.

A janela de tempo de `period['boundary']` a `period['end']` é utilizada para testar a performance do modelo e para a seleção do modelo mais eficaz. A abordagem integrada de aprendizado de máquina e otimização financeira quantitativa permite uma análise mais profunda e uma seleção mais precisa dos componentes da carteira de investimentos, maximizando assim o retorno ajustado ao risco.

## Ambiente

- [Anaconda 3](https://www.anaconda.com/download) (Python 3.11.x)
- GPU habilitada para o uso de bibliotecas de deep learning (opcional)

## Principais Bibliotecas
- [Flake 7.0.0](https://flake8.pycqa.org/en/latest/index.html)
- [Keras 2.13.1](https://keras.io/)
- [Matplotlib 3.7.1](https://matplotlib.org/)
- [Optuna 3.5.0](https://optuna.org/)
- [Optuna-dashboard 0.14.0](https://optuna.org/)
- [Pandas 1.5.3](https://pandas.pydata.org/)
- [PyPortfolioOpt 1.5.5](https://pyportfolioopt.readthedocs.io/en/latest/)
- [Scikit-learn 1.3.0](https://scikit-learn.org/stable/)
- [TensorFlow 2.13.0](https://www.tensorflow.org/)
- [YFinance 0.2.35](https://pypi.org/project/yfinance/)

## (OPCIONAL) Extensões sugeridas do VS Code
- EditorConfig for VS Code
- Git Graph
- Material Icon Theme
- Optuna Dashboard for VS Code
- Rainbow CSV

## Instalação

### Criação do ambiente virtual

```
conda create --name otimizacao_portifolio python=3.11
```

### Ativação do ambiente virtual
```
conda activate otimizacao_portifolio
```

### Instalar dependências
```bash
pip install -r ./requirements.txt
```

### Desativação do ambiente virtual (ao finalizar a execução)
```
conda deactivate
```

## Configuração de Variáveis

O arquivo [`./src/variables.py`](./src/variables.py) contém as variáveis que devem ser ajustadas para personalizar a análise. Abaixo estão as instruções para modificar cada variável:

### tickers
- `tickers`: Lista de tickers para os quais as análises serão feitas. Siga convenção do Yahoo Finance disponível [aqui](https://finance.yahoo.com/).
- Exemplo:

  ```python
    tickers: List[str] = [
        "ABEV3.SA",     # Ambev S.A
        "ALPA4.SA",     # Alpargatas S.A
        "ARZZ3.SA",     # Arezzo Indústria e Comércio S.A
        "ASAI3.SA",     # Assaí Atacadista S.A
        # ...
    ]
    ```

### period
- `period`: Dicionário definindo os períodos de interesse para análise. `period['boundary']` define o limite entre os períodos de treinamento e teste. O formato deve ser YYYY-MM-DD e os dias devem ser úteis (deve haver cotação para o ativo na dada em questão).
- Exemplo:

  ```python
  period: Dict[str, str] = {
      "start": "2013-07-02",
      "boundary": "2021-06-30",
      "end": "2023-07-01"
  }
    ```
### observation_window
- `observation_window`: Dicionário definindo a janela de observação para análise. Interprete: O modelo observa o preço de fechamento dos últimos *stepsBack* dias e, com base neles, infere o preço de fechamento dos próximos *stepsFoward* dias.
- Exemplo:

  ```python
  observation_window: Dict[str, int] = {
      "stepsBack": 240,
      "stepsFoward": 60
  }
  ```

### seed
- `seed`: Semente para garantir a reprodutibilidade em processos aleatórios.
- Exemplo:

  ```python
  seed: int = 25
  ```

### n_trials_optuna
- `n_trials_optuna`: Número de tentativas para otimização de hiperparâmetros com o Optuna. Quando ajustado para 0, o treinamento é feito com base nos melhores hiperparâmetros encontrados em execuções anteriores (armazenados no registro do Optuna). Caso ainda não exista registro, o treinamento é feita com base em hiperparâmetros aleatórios obtidos no espaço de busca do otimizador.
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
  minimum_participation: float = 0.01
  ```

### maximum_participation
- `maximum_participation`: Máximo de participação de um ativo na carteira.
- Exemplo:

  ```python
  minimum_participation: float = 0.15
  ```

## Executar o projeto
```bash
python -u ./main.py [flags]
```
### Comportamento
Faz o download dos dados, treina o modelo, faz a previsão, gera simulação de Monte Carlo para a evolução dos preços e otimiza o portfólio com base no retorno esperado obtido via IA.

### Argumentos Opcionais
- `--no-download`: Executa o projeto sem realizar o download dos dados. Espera-se que os mesmos já tem sido baixados previamente.

- `--only-download`: Não executa projeto, somente realiza o download dos dados.

- `--no-train`: Executa o projeto sem realizar o treinamento. Caso a otimização do portfolio seja com base no retorno esperado obtido via IA, espera-se que o modelo já tenha sido treinado previamente.

- `--no-ai`: Executa a otimização do portifolio com base no retorno esperado obtido via média histórica. Implicitamente, não realiza o treinamento do modelo, não faz a previsão do retorno esperado e não faz a simulação de Monte Carlo.

- `--only-train`: Não executa projeto, somente realiza download dos dados (a depender da flag `--no-download`) e o treinamento do modelo (a depender da flag `--no-ai`).

#### Obs.:
Apesar de inapropriado o uso combinado de certas *flags*, existe a seguinte ordem de prioridade:

- A flag `--no-download` sobrepuja a flag `--only-download`.
- A flag `--no-train` sobrepuja a flag `--only-train`.
- A flag `--no-ai` sobrepuja a flag `--only-train`.

#### Exemplos
- Executa o projeto da maneira *default* tal como descrito em **Comportamento**.

  ```bash
  python -u ./main.py
  ```

- Realiza somente o download dos dados.

  ```bash
  python -u ./main.py --only-download
  ```

  - Realiza somente o treinamento dos modelos.

  ```bash
  python -u ./main.py --no-download --only-train
  ```

- Realiza otimização de portfólio com base no retorno esperado obtido via IA, sem realizar o download dos dados e sem treinar o modelo.

  ```bash
  python -u ./main.py --no-download --no-train
  ```

- Realiza otimização de portfólio com base no retorno esperado obtido via média histórica sem realizar o download dos dados.

  ```bash
  python -u ./main.py --no-download --no-ai
  ```

## Logs da otimização de hiperparâmetros
Para visualizá-los, execute o seguinte comando:

```bash
optuna-dashboard sqlite:///./sqlite/optuna/study.db
```

## Logs do treinamento
Os logs de treinamento são armazenados em `./results/logs/`. Para visualizá-los, execute o seguinte comando:

```bash
tensorboard --logdir ./results/logs/
```

## Backtest
O backtest é executado separadamente do projeto. O seu correto funcionamento depende da escolha adequada de datas no arquivo de variáveis [`./src/variables.py`](./src/variables.py) (*period*). Por razões claras, `period['end']` deve ser uma data pretérita. O backtesting é feito no periodo que sucede `observation_window['stepsFoward']` dias a data `period['end']`. **Certifique-se de que os dados desse período estejam disponíveis!**

Por fim, uma vez tendo executado o projeto (existência do diretório `./results`), execute o seguinte comando:

```bash
python -u ./src/backtest.py
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
│   └── uptuna
│       └── study.py
│
├── src/
│   ├── __pycache__
│   ├── __init__.py
│   ├── backtest.py
│   ├── callbacks.py
│   ├── model.py
│   ├── monteCarlo.py
│   ├── data.py
│   ├── optimizer.py
│   ├── portfolio.py
│   ├── prediction.py
│   ├── train.py
│   └── variables.py
│
├── .editorconfig
│
├── .gitignore
│
├── LICENSE
│
├── main.py
│
├── README.md
│
├── requirements.txt
│
└── setup.cfg
```

### Descrição dos Diretórios e Arquivos:

- `.history/`: Contém arquivos de histórico gerados por algumas ferramentas de desenvolvimento.
- `results/`: Armazena os outputs do projeto, como dados processados, gráficos, logs, objetos serializados e modelos treinados.

    - `backtest/`: Resultados do backtesting.
    - `data/`: Dados financeiros históricos.
    - `graphics/`: Gráficos e visualizações gerados no treinamento.
    - `logs/`: Logs do treinamento (tensorboard).
    - `portfolio/`: Análises e resultados relacionados à otimização do portfólio.
    - `prediction/`: Previsões geradas pelos modelos.
    - `serialized objects/`: Objetos de transformação dos dados (serializados).
    - `trained models/`: Modelos de machine learning treinados e salvos.

- `sqlite/`: Diretório para o banco de dados SQLite.

    - `Optuna/`: Banco de dados SQLite para armazenar os resultados da otimização de hiperparâmetros.

        - `study.db`: Banco de dados SQLite para armazenar os históricos da otimização de hiperparâmetros.

- `src/`: Diretório que contém o código fonte do projeto, incluindo scripts para diferentes funcionalidades, como callbacks, modelagem, simulação de Monte Carlo, obtenção de dados, gerenciamento de portfólio, previsão e treinamento, além de variáveis de configuração.

  - `__pycache__/`: Contém arquivos de bytecode compilados que são criados pelo Python para acelerar a inicialização do programa.
  - `__init__.py`: Arquivo necessário para tratar o diretório como um módulo Python, possibilitando a importação de arquivos dentro deste diretório.
  - [`backtest.py`](./src/backtest.py): Realiza backtesting no portfolio para o período de projeção (`observation_window['stepsFoward']`).
  - [`callbacks.py`](./src/callbacks.py): Define funções de callback que podem ser usadas durante o treinamento de modelos para, por exemplo, salvar checkpoints, ajustar parâmetros ou monitorar o desempenho.
  - [`model.py`](./src/model.py): Inclui a definição e a configuração do modelo de machine learning utilizado no projeto.
  - [`monteCarlo.py`](./src/monteCarlo.py): Implementa simulações de Monte Carlo, que podem ser usadas para projeções financeiras, análise de risco, etc.
  - [`data.py`](./src/data.py): Responsável pela aquisição, limpeza e pré-processamento de dados.
  - [`optimizer.py`](./src/optimizer.py): Responsável pela otimização dos hiperpâmetros do modelo.
  - [`portfolio.py`](./src/portfolio.py): Contém lógicas relacionadas à gestão e análise de portfólios de investimentos.
  - [`prediction.py`](./src/prediction.py): Funções e scripts dedicados à realização de previsões com base no modelo treinado.
  - [`train.py`](./src/train.py): Script para treinar o modelo de machine learning, incluindo a definição de parâmetros, escolha de algoritmos e avaliação de desempenho.
  - [`variables.py`](./src/variables.py): Arquivo para definir variáveis globais, constantes e configurações usadas em todo o projeto, como parâmetros de modelos, caminhos de arquivos, configurações de ambiente, etc.

- [`.editorconfig`](.editorconfig): Define estilos de codificação para facilitar a colaboração entre diferentes desenvolvedores e IDEs.
- [`.gitignore`](.gitignore): Define quais arquivos e diretórios devem ser ignorados pelo sistema de controle de versão Git.
- [`LICENSE.md`](LICENSE.md): Contém a licença de uso e distribuição do projeto.
- [`main.py`](main.py): O script principal para executar o projeto.
- `README.md`: Fornece uma visão geral do projeto, instruções de instalação e outras informações importantes.
- [`requirements.txt`](requirements.txt): Lista todas as bibliotecas necessárias para executar o projeto.
- [`setup.cfg`](setup.cfg): Arquivo de configuração do Flake8, que é uma ferramenta de linting para Python.

## Observações
- O diretório [`./results`](./results/) e suas subpastas são criadas automaticamente ao executar o projeto e pode ser movido ou excluído sem comprometer o correto funcionamento do projeto.

- A cada execução, caso já exista o diretório [`./results`](./results/), é feito a sobreposição dos arquivos com nome comum. Raciocine com esse comportamente para manter o resultado das otimizações de mesmo escopo agrupados.

- Caso decida alterar o espaço de buscas da otimização dos hiperparâmetros em [`./src/optimizer.py`](./src/optimizer.py), remova manualmente o diretório [`./sqlite/`](./sqlite/).

- A configuração de callbacks do treinamento do modelo pode ser alterada em [`./src/callbacks.py`](./src/callbacks.py).

- O uso de GPU torna o processo de otimização, treinamento e inferência aproximadamente 10x mais rápido.


## Autoria
- [Lucas Lima](https://www.linkedin.com/in/lucaslima25)

## Contribuições, Dúvidas e/ou Feedbacks
- *lucas.silva1037@gmail.com*

## Licença
Este projeto está sob a licença XYZ - veja o arquivo LICENSE para detalhes.