from util import Average

def evaluate(model, dataloader):
    loss = Average()
    acc = Average()
    model.eval()
    for (X, label) in dataloader:
        # compute loss
        prediction = model(X)
        curr_loss = model.loss_function(prediction, label)
        loss.update(curr_loss.item())

        # compute accuracy
        curr_acc = model.accuracy(prediction, label)
        acc.update(curr_acc.item())

    return {
        "loss": loss.get(),
        "acc":  acc.get()
    }