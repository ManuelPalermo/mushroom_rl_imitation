import numpy as np


def dataset_as_sequential(*dataset, seq_size):
    """
    Returns a sequential version of a dataset by stacking consecutive
    samples with desired seq_size. An extra dimension is added to the
    new dataset: [B, Seq, ...].

    Args:
        dataset: dataset to turn sequential;
        seq_size(int): size of the sequence to generate;

    Returns:
        A sequential version of the dataset

    """
    size = len(dataset[0]) - (seq_size - 1)
    indexes = np.arange(0, size, 1)
    seq_dataset = []
    for data in dataset:
        seq_dataset.append(
                np.array([data[indexes + s] for s in range(seq_size)]
                         ).swapaxes(0, 1))
    return seq_dataset


def minibatch_generator_sequential(batch_size, seq_size, *dataset):
    """
    Generator that creates minibatches of sequential data from the
    full dataset. An extra dimension is added to the data [B, Seq, ...]

    Args:
        batch_size (int): the maximum size of each minibatch;
        seq_size (int): desired sequence size for each minibatch.
        dataset: the dataset to be splitted.

    Returns:
        The current minibatch.

    """
    size = len(dataset[0]) - (seq_size - 1)
    num_batches = int(np.ceil(size / batch_size))
    indexes = np.arange(0, size, 1)
    np.random.shuffle(indexes)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size))
               for i in range(0, num_batches)]
    for (batch_start, batch_end) in batches:
        batch = []
        for i in range(len(dataset)):
            batch.append(np.array(
                    [dataset[i][indexes[batch_start:batch_end]+s]
                     for s in range(seq_size)]).swapaxes(0, 1))
        yield batch


def minibatch_sample(batch_size, *dataset, continuous=False):
    """
    Same as minibatch generator but only samples 1 random batch
    from the dataset.

    Args:
        batch_size (int): the maximum size of each minibatch;
        continuous(bool): if the sampled batch data is continuous.
        dataset: the dataset to sample from.

    Returns:
        A batch from the dataset.

    """
    size = len(dataset[0])
    if continuous:
        batch_start = np.random.randint(0, size - batch_size)
        indexes = np.arange(batch_start, batch_start + batch_size)
    else:
        indexes = np.random.choice(range(size), min(size, batch_size),
                                   replace=False)
    return [dataset[i][indexes] for i in range(len(dataset))]


def minibatch_sample_sequential(batch_size, seq_size, *dataset, continuous=False):
    """
    Same as minibatch generator sequential but only samples 1 random
    batch from the dataset. An extra dimension is added to the
    data [B, Seq, ...].

    Args:
        batch_size (int): the maximum size of each minibatch;
        seq_size (int): size of the sequence to generate;
        continuous(bool): if the sampled batch data is continuous;
        dataset: the dataset to sample from.

    Returns:
        A batch from the dataset.

    """
    size = len(dataset[0]) - (seq_size - 1)
    if continuous:
        batch_start = np.random.randint(0, size - batch_size)
        indexes = np.arange(batch_start, batch_start + batch_size)
    else:
        indexes = np.random.choice(range(size), min(size, batch_size),
                                   replace=False)
    batch = []
    for i in range(len(dataset)):
        batch.append(np.array(
                [dataset[i][indexes+s] for s in range(seq_size)]
        ).swapaxes(0, 1))
    return batch
