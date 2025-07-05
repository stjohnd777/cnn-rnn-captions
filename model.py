import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-34 for a lighter model
        resnet = models.resnet18(pretrained=True)
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
        # TODO: Complete this function
        # Embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM takes embedded word vectors (of size embed_size) as inputs
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # Linear layer to map the hidden state output dimension
        # to the vocabulary size
        self.linear = nn.Linear(hidden_size, vocab_size)


    def forward(self, features, captions):
            # embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
            # TODO: Complete this function
            # Remove end token from captions
            captions = captions[:, :-1]

            # Embed the captions
            embeddings = self.embed(captions)

            # Concatenate the features and embedded captions
            # Features are added as the first input to the LSTM
            inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)

            # Pass the inputs through LSTM
            lstm_out, _ = self.lstm(inputs)

            # Pass LSTM outputs through the linear layer
            outputs = self.linear(lstm_out)

            return outputs

    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        predicted_sentence = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_sentence.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence
