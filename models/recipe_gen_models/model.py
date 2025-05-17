import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

class RecipeGenModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size=128, hidden_size=256):        
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.encoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.output_layer = nn.Linear(hidden_size, output_vocab_size)


    def forward(self, ingredients, target_recipe=None, teacher_forcing_ratio=0.5):
        batch_size = ingredients.size(0)

        embedded_ingredients = self.embedding(ingredients)
        encoder_outputs, encoder_hidden = self.encoder(embedded_ingredients)

        decoder_hidden = encoder_hidden

        if target_recipe is not None:
            max_length = target_recipe.size()
        else:
            max_length = 100

        outputs = torch.zeros(batch_size, max_length, self.output_layer.out_features).to(ingredients.device)

        decoder_input = torch.ones(batch_size, 1).long().to(ingredients.device)

        for t in range(max_length):
            embedded_decoder_input = self.embedding(decoder_input)

            expanded_hidden = decoder_hidden.transpose(0,1).repeat(1, encoder_outputs.zsize(1), 1)
            attention_input = torch.cat((encoder_outputs, expanded_hidden), dim=2)
            attention_scores = self.attention(attention_input).squeeze(2)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(1)
            
            context = torch.bmm(attention_weights, encoder_outputs)
            
            decoder_output, decoder_hidden = self.decoder(
                torch.cat((embedded_decoder_input, context), dim=2), 
                decoder_hidden
            )
            
            prediction = self.output_layer(decoder_output.squeeze(1))
            outputs[:, t] = prediction
            
            if target_recipe is not None and t < max_length - 1:
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    decoder_input = target_recipe[:, t+1].unsqueeze(1)
                else:
                    top1 = prediction.argmax(1)
                    decoder_input = top1.unsqueeze(1)
            else:
                top1 = prediction.argmax(1)
                decoder_input = top1.unsqueeze(1)
                
        return outputs
