# mini-transformer-lab

A compact transformer-based language model built from scratch in pure PyTorch — designed for learning, tinkering, and really understanding how LLMs work under the hood.

> 🔬 **Built with curiosity, debugged with persistence**
> 
> Almost every error and bug in this project was debugged (sometimes painfully) with a bit of help from AI tools — and yeah, my brain had its fair share of contributions too in debugging, coding, and the whole idea. Because honestly, what else would you expect from a self-taught Python programmer building their own LLM?


## 📋 Before You Start
Please check the [compatibility guide](COMPATIBILITY.md) for important information about supported platforms and testing environment.


## 🚀 Quick Start

### Installation

 ```bash
# Clone this repository
git clone https://github.com/Aranya-Marjara/mini-transformer-lab.git
cd mini-transformer-lab

# Install dependencies (installing in a virtual environment is highly recommended)
pip install torch 
 ```
### Create a file 
 ```
nano your_text.txt
 ```

###  Data Requirements

Your training file should be large enough for the context length:
- **Minimum size**: `(batch_size × context_length + 1)` characters
- **Example**: For batch_size=32, context_length=256 → need ~8,200+ characters
- **Small data?** Use: `--context_length 64 --batch_size 8`


## Paste the text you want to train your model on. (Your data should be large enough for the context length!)
## Here’s an example — use a smaller context length.
 ```
Artificial intelligence has changed the world of technology forever. From simple rule-based systems to advanced large language models, AI continues to evolve with astonishing speed. The transformer architecture revolutionized the way machines understand language, allowing them to capture long-range dependencies and contextual meaning with ease.

Understanding how these models work under the hood is a fascinating challenge. Each token, attention head, and layer contributes to the model's ability to reason, summarize, and generate text. Building a small transformer model from scratch is not just an engineering task but also an educational journey into the core mechanics of intelligence.

Training even a mini transformer on your own data helps reveal how neural networks learn from patterns. As the loss goes down, the model starts to grasp structure — words begin to align, and meaning starts to emerge. These tiny experiments reflect, in miniature, what the biggest LLMs in the world are doing at scale.

Open-source research and tinkering allow anyone to learn how these systems work. Sharing your code publicly means that others can build on your work, improve it, and contribute ideas. True progress happens when knowledge is free, transparent, and collaborative.

This text exists to provide enough data for your mini-transformer-lab model to train successfully, test attention layers, and generate a few coherent lines. Keep experimenting, because every small step adds up to something bigger in the world of AI.
 ```


### Basic Usage
# Train a new model on your text (use smaller context length — recommended)
 ```
python3 mini-transformer-lab.py train --data your_text.txt --epochs 10 --context_length 64 --batch_size 8
 ```

# Generate some text
 ```
python3 mini-transformer-lab.py generate --checkpoint checkpoint_epoch_10.pt --prompt "The future of AI is"
 ```

### Quick Test
 ```bash
# Create a tiny test file
echo "Hello world! This is a test." > test.txt

# Train quickly
python3 mini-transformer-lab.py train --data test.txt --epochs 5 --context_length 16 --batch_size 2

# Generate
python3 mini-transformer-lab.py generate --checkpoint checkpoint_epoch_5.pt --prompt "Hello"
 ```


### Troubleshooting

- **"Data too short for context length"**  
  Use smaller context: `--context_length 64 --batch_size 8`  
  Or get more data or use the sample text above.

- **"Checkpoint not found"**  
  Use actual checkpoint names: `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`

- **"First outputs are gibberish"**  
  This is normal for early training!  
  Train for more epochs (20–50) or use a larger dataset for more coherent text.
