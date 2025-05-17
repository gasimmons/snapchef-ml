import sys
import os
from pandas.core.frame import XMLParsers
import wandb
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.preprocessing import load_data, get_example

with open("config.yaml") as f:
    config = yaml.safe_load(f)

wandb.init(project=config["project"], config=config)


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.reverse_vocab = []
    
    def build_vocab(self, texts):
        idx = 2
        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
        self.reverse_vocab = ["<pad>", "<unk>"] + [k for k, v in sorted(self.vocab.items(), key=lambda x: x[1]) if v >= 2]

    def encode(self, text, max_len=100):
        return [self.vocab.get(w, 1) for w in text.split()][:max_len]

tokenizer = SimpleTokenizer()

df = load_data("models/recipe_gen_models/data/recipenlg.csv", limit=5000)
examples = get_example(df)

ingredients = [ing for ing, _ in examples]
instructions = [instr for _, instr in examples]
tokenizer.build_vocab(ingredients + instructions)

class RecipeDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=100):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ing, instr = self.examples[idx]
        x = torch.tensor(self.tokenizer.encode(ing, self.max_len))
        y = torch.tensor(self.tokenizer.encode(instr, self.max_len))
        return x, y

dataset = RecipeDataset(examples, tokenizer)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# -------------------------------
# 3. Model, Loss, Optimizer
# -------------------------------
model = RecipeGenModel(
    vocab_size=len(tokenizer.vocab),
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
loss_fn = nn.CrossEntropyLoss()

# -------------------------------
# 4. Training Loop
# -------------------------------
for epoch in range(config["epochs"]):
    total_loss = 0
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        output = model(x)  # (B, T, V)
        output = output.view(-1, output.shape[-1])
        y = y.view(-1)

        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss})

wandb.finish()
