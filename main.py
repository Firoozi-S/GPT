import torch

import gpt
import utils
import globals

with open(file = 'input.txt', mode = 'r', encoding = 'utf-8') as input:
    data = input.read()

unique_characters = set(data)

vocab_size = len(unique_characters)

char_to_int_map = {char: idx for idx, char in enumerate(unique_characters)}
int_to_char_map = {idx: char for idx, char in enumerate(unique_characters)}

encoder = lambda s: [char_to_int_map[c] for c in s]
decoder = lambda i: ''.join([int_to_char_map[c] for c in i])

data = torch.tensor(encoder(data))

train_data = data[ : int(0.9 * len(data))]
validation_data = data[int(0.9 * len(data)) : ]

model  = gpt.GPT(vocab_size)
optimizer = torch.optim.Adam(params = model.parameters(), lr = globals.LEARNING_RATE, eps = globals.EPSILON)

for i in range(globals.EPOCHS):
    if i % globals.EVALUATION_INTERVAL == 0 or i == globals.EPOCHS - 1:
        losses = utils.LossEstimation(model = model, train_data = train_data, validation_data = validation_data)
        print('Step: {}: train loss: {}, validation loss: {}'.format(i, losses[train_data], losses[validation_data]))

    x, y = utils.DataBatch(data = data)

    logits, loss = model.forward(x, y)

    print('Loss for batch {} is {}'.format(i + 1, loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


context = torch.zeros(size = (1, 1), dtype = torch.long)
print(decoder(model.generate(index = context, num_tokens = globals.NUM_TOKENS)[0].tolist()))

