import torch
import torch.nn as nn
import torch.nn.functional as F

import settings


class BOW(nn.Module):
    """
    Bag-of-words (BOW) classification model
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary
        @param emb_dim: size of the word embedding
        """
        super(BOW, self).__init__()
        self.embed = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0,  # !!
        )
        self.linear = nn.Linear(
            emb_dim,
            settings.NUM_CLASSES,
        )

    def forward(self, data, length):
        """
        @param data: matrix of size (batch_size, max_sentence_length);
        Each row represents a review that is represented using n-gram index.
        Note that they are padded to have same length.
        @param length: int tensor of size (batch_size);
        Tensor represents the non-trivial length of each sentence in the data
        (excludes padding).
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0], 1).expand_as(out).float()

        # Return logits
        out = self.linear(out.float())
        return out


def eval_model(model, loader):
    """
    Helper function to test model performance on given dataset
    @param: data loader for the dataset to test against
    """
    correct = 0
    total = 0

    model.eval()

    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


def train(model, train_loader, val_loader, lr=settings.CONFIG["lr"]):
    """
    Train model
    """
    # Save accuracies for plotting
    train_accs = []
    val_accs = []

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    for epoch in range(settings.CONFIG["num_epochs"]):
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

            # Validate every 100 iterations
            if i > 0 and i % 100 == 0:
                train_acc = eval_model(model, train_loader)
                val_acc = eval_model(model, val_loader)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                print(
                    "Epoch: [{}/{}], Step: [{}/{}], Validation accuracy: {}, Training accuracy: {}".format(
                        epoch + 1,
                        settings.CONFIG["num_epochs"],
                        i + 1,
                        len(train_loader),
                        val_acc,
                        train_acc,
                    ))

    print("After training for n={} epochs...".format(
        settings.CONFIG["num_epochs"]))
    print("Training accuracy: {}".format(eval_model(model, train_loader)))
    print("Validation accuracy: {}".format(eval_model(model, val_loader)))
    # print("Test accuracy: {}".format(eval_model(test_loader, model)))

    return train_accs, val_accs
