# Meeting Summary Categorization Tool

This tool helps categorize meeting summaries into predefined categories using machine learning. The application includes both a model training component and a web service for making predictions.

## Features

- Text categorization into 5 categories: Atualizações de Projeto, Achados de Pesquisa, Gestão de Equipe, Reuniões com Clientes, Outras
- Multi-head model architecture to reduce category prediction bias
- Web interface for easy interaction
- API endpoints for integration with other systems
- Confidence scores for each prediction

## Setup

### Installation
   ```

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your training data:
   - Place the CSV data file that you wanna train in the `data/` directory
   - The CSV should have at least two columns: `text` and `category`
   - Example file: `data/resumos.csv`

## Usage

### Training the Model

To train a new model:

```bash
cd src
python train.py --data_path ../data/resumos.csv --model_dir ../models --epochs 5 --use_class_weights
```

Options:
- `--data_path`: Path to your CSV data file
- `--model_dir`: Directory to save the trained model
- `--epochs`: Number of training epochs
- `--use_class_weights`: Use class weights for balanced training

### Running the Web Service

To start the web server:

```bash
cd src
python app.py --model_dir ../models/best_model --port 8080
```

Options:
- `--model_dir`: Directory containing the trained model
- `--port`: Port number to run the server on (default: 5000)

The server will be available at: http://127.0.0.1:8080

### API Endpoints

The following endpoints are available:

- `GET /`: Homepage with UI
- `GET /health`: Health check
- `GET /categories`: Get available categories
- `POST /classify`: Classify a meeting summary
  ```json
  {"summary": "Text of the meeting summary"}
  ```
- `POST /classify_with_confidence`: Classify with confidence scores
  ```json
  {"summary": "Text of the meeting summary"}
  ```

### Example API Request

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"summary":"Análise dos dados coletados na pesquisa de usuário"}' \
  http://127.0.0.1:8080/classify_with_confidence
```

## Troubleshooting

- If you get a "port already in use" error, try a different port:
  ```bash
  python app.py --model_dir ../models/best_model --port 8081
  ```

- If you get a "file not found" error when running the app, make sure you're in the `src` directory:
  ```bash
  cd src
  python app.py --model_dir ../models/best_model --port 8080
  ```

- If predictions show low confidence or bias toward one category, you may need to retrain with:
  - More diverse training data
  - More training epochs
  - Class weights enabled
