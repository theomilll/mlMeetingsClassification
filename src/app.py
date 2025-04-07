import argparse
import glob
import json
import os
import sys

from flask import Flask, jsonify, render_template_string, request

from predict import SummaryClassifier

app = Flask(__name__)

# Initialize classifier globally
classifier = None

# Simple HTML template for the homepage
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Meeting Summary Categorization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        textarea {
            width: 100%;
            min-height: 100px;
            margin-bottom: 10px;
            padding: 10px;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        .category {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .confidence {
            margin-top: 10px;
        }
        .confidence-bar {
            height: 20px;
            background-color: #ddd;
            margin-bottom: 5px;
            position: relative;
        }
        .confidence-fill {
            height: 100%;
            background-color: #4CAF50;
            position: absolute;
            top: 0;
            left: 0;
        }
        .confidence-label {
            position: absolute;
            right: 5px;
            color: black;
            font-size: 12px;
            line-height: 20px;
        }
    </style>
</head>
<body>
    <h1>Meeting Summary Categorization</h1>
    <p>Enter a meeting summary text to classify it into one of the predefined categories.</p>
    
    <div>
        <textarea id="summary" placeholder="Enter meeting summary text here..."></textarea>
        <button onclick="classifySummary()">Classify</button>
    </div>
    
    <div id="result" class="result" style="display: none;">
        <div class="category">Category: <span id="category"></span></div>
        <div class="confidence">
            <h3>Confidence Scores:</h3>
            <div id="confidence-scores"></div>
        </div>
    </div>

    <script>
        function classifySummary() {
            const summary = document.getElementById('summary').value;
            if (!summary) {
                alert('Please enter a summary text');
                return;
            }
            
            fetch('/classify_with_confidence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ summary: summary }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                document.getElementById('category').textContent = data.category;
                
                const confidenceScores = document.getElementById('confidence-scores');
                confidenceScores.innerHTML = '';
                
                // Sort confidence scores
                const sortedScores = Object.entries(data.confidence).sort((a, b) => b[1] - a[1]);
                
                // Create confidence bars
                sortedScores.forEach(([category, score]) => {
                    const div = document.createElement('div');
                    div.innerHTML = `
                        <div>${category}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${score * 100}%"></div>
                            <div class="confidence-label">${(score * 100).toFixed(2)}%</div>
                        </div>
                    `;
                    confidenceScores.appendChild(div);
                });
                
                document.getElementById('result').style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            });
        }
    </script>
</body>
</html>
"""

def find_model_dir(base_dir="./models"):
    """Find the best available model directory"""
    # First check if final model exists
    final_dir = os.path.join(base_dir, "final")
    if os.path.exists(final_dir) and os.path.isdir(final_dir):
        return final_dir
    
    # Check for checkpoint directories
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-epoch-*"))
    if checkpoints:
        # Sort by epoch number to get the latest
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        return checkpoints[-1]
    
    # Try the base directory itself
    if os.path.exists(base_dir):
        model_files = glob.glob(os.path.join(base_dir, "*.bin")) + glob.glob(os.path.join(base_dir, "*.pt"))
        if model_files:
            return base_dir
    
    return None

@app.route('/', methods=['GET'])
def home():
    """Render the homepage with a simple UI."""
    return render_template_string(HOME_TEMPLATE)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if classifier is None:
        return jsonify({"status": "not ready", "error": "Classifier not initialized"}), 503
    return jsonify({"status": "healthy"})

@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify a summary text.
    
    Expected JSON payload:
    {
        "summary": "Text of the meeting summary"
    }
    
    Returns:
    {
        "category": "Predicted category"
    }
    """
    # Check if classifier is initialized
    if classifier is None:
        return jsonify({"error": "Classifier not initialized"}), 503
    
    # Check if request has JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get data from request
    data = request.get_json()
    
    # Check if summary is in data
    if 'summary' not in data:
        return jsonify({"error": "JSON must contain 'summary' field"}), 400
    
    # Get summary text
    summary = data['summary']
    
    # Ensure summary is not empty
    if not summary.strip():
        return jsonify({"error": "Summary text cannot be empty"}), 400
    
    # Classify summary
    try:
        category = classifier.classify(summary)
        return jsonify({"category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classify_with_confidence', methods=['POST'])
def classify_with_confidence():
    """
    Classify a summary text and return confidence scores.
    
    Expected JSON payload:
    {
        "summary": "Text of the meeting summary"
    }
    
    Returns:
    {
        "category": "Predicted category",
        "confidence": {
            "category1": 0.8,
            "category2": 0.1,
            ...
        },
        "metadata": {
            "threshold_met": true/false,
            "is_confident": true/false,
            "top_margin": 0.x
        }
    }
    """
    # Check if classifier is initialized
    if classifier is None:
        return jsonify({"error": "Classifier not initialized"}), 503
    
    # Check if request has JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get data from request
    data = request.get_json()
    
    # Check if summary is in data
    if 'summary' not in data:
        return jsonify({"error": "JSON must contain 'summary' field"}), 400
    
    # Get summary text
    summary = data['summary']
    
    # Ensure summary is not empty
    if not summary.strip():
        return jsonify({"error": "Summary text cannot be empty"}), 400
    
    # Classify summary
    try:
        category, confidence = classifier.classify(summary, return_confidence=True, confidence_threshold=0.3)
        
        # Calculate additional metadata
        sorted_confidence = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
        top_confidence = sorted_confidence[0][1]
        
        metadata = {
            "threshold_met": top_confidence >= 0.3,
            "is_confident": top_confidence >= 0.3,
            "top_margin": sorted_confidence[0][1] - sorted_confidence[1][1] if len(sorted_confidence) > 1 else 1.0
        }
        
        return jsonify({
            "category": category,
            "confidence": confidence,
            "metadata": metadata
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """
    Get available categories.
    
    Returns:
    {
        "categories": ["category1", "category2", ...]
    }
    """
    # Check if classifier is initialized
    if classifier is None:
        return jsonify({"error": "Classifier not initialized"}), 503
    
    try:
        categories = list(classifier.label_encoder.keys())
        return jsonify({"categories": categories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def check_model_exists(model_dir):
    """
    Check if the model directory exists and contains the necessary files.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        True if the model directory exists and contains the necessary files
    """
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist")
        return False
    
    # Check for necessary files
    required_files = [
        'config.json',
        'pytorch_model.bin',
        'tokenizer.json',
        'tokenizer_config.json',
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            print(f"Error: Required file {file} not found in model directory")
            return False
    
    # Check for label encoder
    encoder_path = os.path.join(os.path.dirname(model_dir), 'label_encoder.csv')
    if not os.path.exists(encoder_path):
        print(f"Error: Label encoder file not found at {encoder_path}")
        return False
    
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the meeting summary categorization API")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
    parser.add_argument('--model_dir', type=str, default='./models/final', help='Directory containing the model')
    
    return parser.parse_args()

def init_classifier(model_dir):
    """Initialize the classifier."""
    global classifier
    
    try:
        print(f"Initializing classifier from {model_dir}...")
        if not check_model_exists(model_dir):
            print("Model check failed. Please train a model first using train.py.")
            return None
        
        from predict import SummaryClassifier
        classifier = SummaryClassifier(model_dir)
        print("Classifier initialized successfully!")
        return classifier
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run the meeting summary categorization API")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
    parser.add_argument('--model_dir', type=str, default='./models/final', help='Directory containing the model')
    
    args = parser.parse_args()
    
    # Try to find the best available model directory
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        # Try to find an alternative model directory
        alternative_dir = find_model_dir("../models")  # Try looking in parent directory too
        if alternative_dir:
            print(f"Specified model directory {model_dir} not found. Using {alternative_dir} instead.")
            model_dir = alternative_dir
        else:
            print(f"Warning: No model found at {model_dir} or any alternative location.")
            print("The server will start, but classification functionality will be unavailable.")
            print("Please run the training script first: python train.py --data_path ../data/resumos.csv --model_dir ../models")
    
    # Initialize classifier
    global classifier
    
    if os.path.exists(model_dir):
        print(f"Initializing classifier from {model_dir}...")
        
        try:
            # Initialize classifier
            classifier = SummaryClassifier(model_dir)
            print("Classifier initialized successfully!")
            print(f"Available categories: {list(classifier.label_encoder.keys())}")
        except Exception as e:
            print(f"Error initializing classifier: {e}")
            print("Classification functionality will be unavailable.")
            classifier = None
    else:
        classifier = None
    
    # Print available endpoints
    print(f"\nServer will be available at http://{args.host}:{args.port}")
    print("Available endpoints:")
    print("  /              - Homepage with UI")
    print("  /health        - Health check")
    print("  /classify      - Classify a meeting summary")
    print("  /classify_with_confidence - Classify with confidence scores")
    print("  /categories    - Get available categories")
    
    # Modify home route to show helpful message when no classifier is available
    @app.route('/', methods=['GET'])
    def home_override():
        if classifier is None:
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Meeting Summary Categorization</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    .warning {
                        background-color: #fff3cd;
                        border: 1px solid #ffeeba;
                        color: #856404;
                        padding: 15px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }
                    pre {
                        background-color: #f8f9fa;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }
                </style>
            </head>
            <body>
                <h1>Meeting Summary Categorization</h1>
                <div class="warning">
                    <h2>Model Not Available</h2>
                    <p>The classification model has not been trained yet. Please run the training script first:</p>
                    <pre>python src/train.py --data_path data/resumos.csv --model_dir models --epochs 3</pre>
                    <p>Once the training is complete, restart this server to enable classification functionality.</p>
                </div>
            </body>
            </html>
            """)
        return render_template_string(HOME_TEMPLATE)
    
    # Run app
    app.run(host=args.host, port=args.port)
    
if __name__ == "__main__":
    main() 