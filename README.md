# Ferramenta de Categorização de Resumos de Reuniões

Esta ferramenta permite treinar e usar modelos de Machine Learning para categorizar resumos de reuniões em cinco categorias:
- Atualizações de Projeto
- Achados de Pesquisa
- Gestão de Equipe
- Reuniões com Clientes
- Outras

## Pré-requisitos

- Python 3.8 ou superior
- Instale as dependências:

  ```bash
  pip install -r requirements.txt
  ```

## Estrutura do Projeto

```
├── data/
│   └── resumos.csv         # Seu dataset de treinamento (colunas: text, category)
├── models/                 # Modelos e encoders salvos
├── src/                    # Código-fonte
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py            # Pipeline original (MultiHeadTextClassifier)
│   ├── train_tfidf.py      # Pipeline TF-IDF + LogisticRegression
│   ├── train_bert.py       # Pipeline de fine-tuning BERT em português
│   ├── predict.py          # CLI para modelo original
│   └── predict_tfidf.py    # CLI para modelo TF-IDF + LogisticRegression
├── tests/                  # Testes automatizados
└── requirements.txt
```

## Como Treinar

### 1) TF-IDF + LogisticRegression (baseline rápido)

```bash
python3 src/train_tfidf.py \
  --data_path data/resumos.csv \
  --label_column category \
  --output_dir models/tfidf
```

### 2) BERT em Português (mais preciso)

```bash
python3 src/train_bert.py \
  --data_path data/resumos.csv \
  --label_column category \
  --epochs 5 \
  --batch_size 8 \
  --output_dir models/bert_finetuned
```

## Como Prever

### TF-IDF + LogisticRegression

```bash
python3 src/predict_tfidf.py "Seu texto de reunião aqui"
```

### Modelo Original

```bash
python3 src/predict.py "Seu texto de reunião aqui"
```

#### Dica para facilitar a execução

Adicione um alias no seu shell (~/.bashrc ou ~/.zshrc):

```bash
alias predict="python3 src/predict_tfidf.py"
```

A partir de então, basta executar:

```bash
predict "Resumo da reunião"
```

## Testes

```bash
pytest
```

> **Observações:**
> - Ajuste `--label_column` caso seu CSV use outro nome para a coluna de categorias.
> - Para treinar com BERT, é necessário internet para baixar o modelo base.
> - O script `predict.py` carrega, por padrão, o modelo em `models/final`.
