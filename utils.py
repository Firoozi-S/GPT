import globals

import torch


def DataBatch(data):
    index = torch.randint(low = 0, high = len(data) - globals.BLOCK_SIZE, size = (globals.BATCH_SIZE, ))
    x = torch.stack(tensors = [data[i : i + globals.BLOCK_SIZE] for i in index])
    y = torch.stack(tensors = [data[i + 1 : 1 + i + globals.BLOCK_SIZE] for i in index])
    return x, y   

def LossEstimation(model, train_data, validation_data):
    average_loss = {}
    model.eval()
    for split in [train_data, validation_data]:
        losses = torch.zeros(globals.EVALUATION_ITERATION)
        for i in range(globals.EVALUATION_ITERATION):
            x, y = DataBatch(split)
            _, loss = model(x,y)
            losses[i] = loss.item()
            print('Evaluation iteration: {}, Iteration loss: {}'.format(i + 1, losses[i]))
        average_loss[split] = losses.mean()
    model.train()
    return average_loss

    