import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
# from allennlp.modules.conditional_random_field import ConditionalRandomField
from torchtext import data

from ignite.engines import Events, Engine
from models import RNNTagger, CRFTagger
from handlers import ModelCheckpoint
from helper import create_supervised_evaluator
# from ignite.metrics import CategoricalAccuracy, Precision, Recall
from metrics import SequenceTagAccuracy

BASE_PATH = "../data/"
BATCH_SIZE = 16
HIDDEN_SIZE = 512
MAX_EPOCHS = 2
LAYER_SIZE = 1
EMBEDDING_SIZE = 300
LEARNING_RATE = 0.01

if __name__ == '__main__':
    # Dataset
    sentence = data.Field(lower=True, include_lengths=True, batch_first=False)
    tags = data.Field(batch_first=True)

    train_dataset = data.TabularDataset(
        path=BASE_PATH + "postags_train.csv",
        format="csv",
        skip_header=True,
        fields=[("sentence", sentence), ("tags", tags)])
    test_dataset = data.TabularDataset(
        path=BASE_PATH + "postags_test.csv",
        format="csv",
        skip_header=True,
        fields=[("sentence", sentence), ("tags", tags)])
    tags.build_vocab(train_dataset.tags)
    sentence.build_vocab(train_dataset.sentence)

    train_iter = data.BucketIterator(
        train_dataset,
        BATCH_SIZE,
        repeat=False,
        shuffle=True,
        sort_key=lambda x: len(x.sentence),
        sort_within_batch=True)
    test_iter = data.BucketIterator(
        test_dataset,
        BATCH_SIZE,
        repeat=False,
        shuffle=True,
        sort_key=lambda x: len(x.sentence),
        sort_within_batch=True)

    tags.build_vocab(train_dataset.tags)
    sentence.build_vocab(train_dataset.sentence)

    # Net initialization
    embedding = nn.Embedding(len(sentence.vocab), EMBEDDING_SIZE, 1)
    rnn_tagger = RNNTagger(
        nemb=EMBEDDING_SIZE,
        nhid=HIDDEN_SIZE,
        nlayers=LAYER_SIZE,
        drop=0.5,
        ntags=len(tags.vocab))
    crf_tagger = CRFTagger(rnn_tagger=rnn_tagger, ntags=len(tags.vocab))
    opt = Adam(
        lr=LEARNING_RATE,
        params=filter(lambda p: p.requires_grad, crf_tagger.parameters()))

    # Trainer initialization
    def process_function(engine, batch):
        crf_tagger.train()
        opt.zero_grad()
        sentence = batch.sentence[0]
        sent_len = batch.sentence[1]
        tags = batch.tags

        x = embedding(sentence)
        result = crf_tagger(x, sent_len.numpy(), tags)
        result.backward()
        opt.step()

        return result.detach()

    def evaluation_function(engine, batch):
        crf_tagger.eval()
        sentence = batch.sentence[0]
        sent_len = batch.sentence[1]
        tags = batch.tags

        x = embedding(sentence)
        result = torch.tensor(
            crf_tagger.decode(x, sent_len.numpy()), dtype=torch.int32)
        result = result.transpose(1, 0)

        return result, tags.detach()

    trainer = Engine(process_function)
    evaluator = create_supervised_evaluator(
        model=crf_tagger,
        inference_fn=evaluation_function,
        metrics={
            "acc": SequenceTagAccuracy(tags.vocab),
        })
    checkpoint = ModelCheckpoint(
        "models",
        "postag-en",
        save_interval=100,
        n_saved=5,
        create_dir=True,
        require_empty=False)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, checkpoint,
                              {"bilstm-crf": crf_tagger})
    trainer.add_event_handler(Events.COMPLETED, checkpoint,
                              {"bilstm-crf": crf_tagger})

    def log_average_training_loss(window=10):
        history = []

        def log_training_loss(engine):
            history.append(engine.state.output.numpy())
            if engine.state.iteration % window == 0:
                iterations_per_epoch = len(engine.state.dataloader)
                current_iteration = engine.state.iteration % \
                    iterations_per_epoch
                if current_iteration == 0:
                    current_iteration = iterations_per_epoch
                avg_loss = np.array(history).mean()
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                      "".format(engine.state.epoch, current_iteration,
                                iterations_per_epoch, avg_loss))
                del history[:]

        return log_training_loss

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_iter)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["acc"]
        print("=====================================")
        print("Validation Results - Epoch: {}".format(engine.state.epoch))
        print("Avg accuracy: {:.2f}".format(avg_accuracy))
        print("=====================================")

    trainer.add_event_handler(Events.ITERATION_COMPLETED,
                              log_average_training_loss())

    trainer.run(train_iter, max_epochs=MAX_EPOCHS)
