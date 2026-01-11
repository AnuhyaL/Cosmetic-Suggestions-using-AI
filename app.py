import os
import torch
import json
import pickle
from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

class SkincareReviewPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        artifacts_path = os.path.join(os.path.dirname(__file__), 'model_artifacts')
        
        with open(os.path.join(artifacts_path, 'model_config.json'), 'r') as f:
            self.model_config = json.load(f)
        
        with open(os.path.join(artifacts_path, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(artifacts_path, 'tokenizer'))
        
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=self.model_config['num_labels']
        )
        self.model.load_state_dict(torch.load(os.path.join(artifacts_path, 'best_model_weights.pth'), map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess_input(self, skin_type='', product='', brand='', ingredients='', review='', max_length=128):
        input_text = f"{skin_type} {product} {brand} {ingredients} {review}".strip()
        
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict_category(self, **kwargs):
        processed_input = self.preprocess_input(**kwargs)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=processed_input['input_ids'], 
                attention_mask=processed_input['attention_mask']
            )
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
        
        predicted_category = self.label_encoder.inverse_transform(preds.cpu().numpy())[0]
        
        return predicted_category
    
    def get_available_categories(self):
        return list(self.label_encoder.classes_)

predictor = SkincareReviewPredictor()

@app.route('/')
def index():
    return render_template('index.html', categories=predictor.get_available_categories())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        prediction = predictor.predict_category(**data)
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)