import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding vector - embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM - takes embedded word vectors (of a specified size) as inputs and outputs hidden states 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # fully-connected output layer - the linear layer - maps the hidden state output dimension to the vocab_size
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        #initialize weights - using xavier initializer
        self.init_weights()
        
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embed.weight)
        
        
    
    def forward(self, features, captions):
        # create embedded word vectors for each token in a batch of captions
        caption = self.embed(captions[:,:-1])
        # lstm input - embeddings
        embeddings = torch.cat((features.unsqueeze(dim=1), caption), dim=1)
        
        lstm_out, _ = self.lstm(embeddings)
        
        # Convert LSTM outputs to word predictions - scores
        output_scores = self.fc(lstm_out)
        
        return output_scores

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        predictions = []
        count = 0
        word_item = None
        
        while count < max_len and word_item != 1 :
            
            #Predict output - sentences 
            lstm_out, states = self.lstm(inputs, states)
            output_scores = self.fc(lstm_out)
            
            #Get max value
            prob, word = output_scores.max(2)
            
            #append word
            word_item = word.item()
            predictions.append(word_item)
            
            #next input is current prediction
            inputs = self.embed(word)
            
            count+=1
        
        return predictions