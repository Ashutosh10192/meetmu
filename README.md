# Chat Response Recommendation System

## Overview
This project implements an AI-powered chat response recommendation system that predicts User A's replies based on conversation context and User B's messages using Transformer-based models.

## Dataset Information
- **Source**: conversationfile.xlsx containing real conversation data
- **Total Messages**: 22 messages across 4 conversations
- **Training Pairs Generated**: 9 pairs for model training
- **Users**: User A and User B with balanced participation (11 messages each)

## Model Architecture
- **Base Model**: DistilGPT-2 (selected for efficiency and offline deployment)
- **Task**: Conversational Response Generation
- **Training Framework**: PyTorch + Transformers
- **Context Window**: 5 previous messages
- **Max Response Length**: 100 tokens

## Key Features
1. **Efficient Preprocessing**: Text cleaning, normalization, and context window creation
2. **Context-Aware Generation**: Uses conversation history for coherent responses
3. **Offline Deployment**: No internet connection required for inference
4. **Performance Optimization**: Optimized for CPU inference with gradient computation disabled

## Performance Metrics (Planned Evaluation)
- **BLEU Scores**: For measuring response similarity to reference responses
- **ROUGE Scores**: For evaluating response quality and overlap
- **Perplexity**: For measuring model confidence
- **Inference Speed**: Average response generation time

## System Requirements
- **Python**: 3.8+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for model files
- **GPU**: Optional (improves inference speed)

## Installation & Dependencies
```bash
pip install torch>=1.9.0 transformers>=4.0.0 pandas>=1.3.0 numpy>=1.21.0 nltk>=3.6.0 rouge-score>=0.0.4 joblib scikit-learn matplotlib seaborn openpyxl
```

## Usage

### Basic Usage
```python
import joblib
import pandas as pd

# Load conversation data
df = pd.read_excel('conversationfile.xlsx')

# Load model (when available)
# model_package = joblib.load('Model.joblib')

# Generate response
context = "A: Hello! How are you? B: I'm doing great, thanks!"
user_b_message = "What are your plans for today?"
# response = generator.generate_response(context, user_b_message)
```

### Interactive Chat
```python
# generator.interactive_chat()  # When model is loaded
```

## Model Selection Justification

### Why DistilGPT-2?
1. **Efficiency**: 82M parameters vs 124M+ for full GPT-2
2. **Offline Deployment**: Pre-trained weights available locally
3. **Good Performance**: Maintains 97% of GPT-2's performance
4. **Fast Inference**: Optimized for real-time response generation
5. **Resource Efficient**: Suitable for CPU-only deployment

### Alternative Models Considered
- **GPT-2**: Larger, better quality but slower inference
- **T5**: Good for structured tasks but requires specific input format
- **BERT**: Excellent for understanding but needs additional decoder for generation

## Data Preprocessing Pipeline
1. **Text Cleaning**: Remove quotes, normalize case, handle special characters
2. **Context Creation**: Build conversation history with 5-message windows
3. **Pair Generation**: Create (Context + User B Message) -> User A Response pairs
4. **Quality Filtering**: Remove empty or very short responses

## Training Strategy
1. **Fine-tuning Approach**: Fine-tune pre-trained DistilGPT-2 on conversation data
2. **Loss Function**: Cross-entropy loss for language modeling
3. **Optimization**: Adam optimizer with learning rate scheduling
4. **Regularization**: Dropout and weight decay to prevent overfitting

## Evaluation Methodology
1. **Automatic Metrics**: BLEU, ROUGE, Perplexity scores
2. **Human Evaluation**: Response relevance and naturalness (planned)
3. **Context Awareness**: Ability to maintain conversation coherence
4. **Diversity**: Variety in generated responses

## Deployment Considerations

### Offline Deployment
- All model weights included in package
- No internet connection required
- Suitable for privacy-sensitive applications
- Edge computing compatible

### Performance Optimization
- Model optimized for CPU inference
- Gradient computation disabled
- Memory usage minimized
- Batch processing supported

## Limitations
1. **Small Dataset**: Limited training data (22 messages total)
2. **Domain Specific**: Trained on specific conversation style
3. **Context Length**: Limited to 512 tokens maximum
4. **Generation Diversity**: May produce similar responses for similar contexts

## Future Improvements
1. **Data Augmentation**: Expand training dataset with more conversations
2. **Response Ranking**: Implement multiple response generation and ranking
3. **Personality Modeling**: Add user-specific response style adaptation
4. **Real-time Integration**: Connect with chat applications
5. **Multi-turn Optimization**: Better handling of long conversations

## File Structure
```
ChatRec_Model.ipynb      # Main development notebook
Model.joblib             # Serialized model package
ReadMe.txt              # This documentation file
Report.pdf              # Comprehensive technical report
conversationfile.xlsx   # Source conversation data
```

## Technical Implementation Details

### Data Processing
- Conversation parsing from Excel format
- User role identification (User A vs User B)
- Context window creation with conversation boundaries
- Text normalization and cleaning

### Model Configuration
- Model: distilgpt2
- Max Length: 512 tokens
- Learning Rate: 5e-5
- Batch Size: 4
- Training Epochs: 3
- Temperature: 0.8 (for generation diversity)

### Response Generation
- Input Format: "Context: [history] B: [message] Response:"
- Output: Natural language response from User A's perspective
- Post-processing: Clean up artifacts and ensure coherence

## Evaluation Results (To be completed after training)
- BLEU-1: [To be calculated]
- BLEU-4: [To be calculated]  
- ROUGE-1: [To be calculated]
- ROUGE-L: [To be calculated]
- Perplexity: [To be calculated]
- Inference Speed: [To be benchmarked]

## Contact & Support
For questions about this implementation or suggestions for improvements, please refer to the technical documentation in the Report.pdf file.

## License & Acknowledgments
This project uses open-source libraries including PyTorch, Transformers, and NLTK. Thanks to Hugging Face for providing pre-trained models suitable for offline deployment.


