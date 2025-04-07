# Meeting Summary Categorization Tool

This project has been refactored to focus on the core machine learning functionality for text categorization. Users will need to provide their own training data.

## Setup

1. Place your labeled data in the data/ directory in CSV format with 'text' and 'category' columns
2. Train the model: python categorization/src/train.py --data_path data/your_dataset.csv
3. Start the web server: python categorization/src/app.py

Check the README.md inside the categorization directory for detailed instructions.
