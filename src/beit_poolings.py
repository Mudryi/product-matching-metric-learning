import torch


def get_hidden_states(model_output):
    hidden_states = model_output["hidden_states"]

    hidden_states = torch.stack(hidden_states, dim=0)
    # hidden_states = torch.squeeze(hidden_states, dim=1)
    hidden_states = hidden_states.permute(1, 2, 0, 3)
    return hidden_states


def concatenate_pooling(hidden_states):
    last_four_layers = [hidden_states[:, :, i] for i in (-1, -2, -3, -4)]
    cat_hidden_states = torch.cat(last_four_layers, -1)
    cat_hidden_states = torch.mean(cat_hidden_states, dim=1)
    # cat_hidden_states = torch.reshape(torch.mean(cat_hidden_states, dim=0), (1, -1))

    return cat_hidden_states


def mean_pooling(last_hidden_state):
    return torch.mean(last_hidden_state, dim=1)


def mean_max_pooling(last_hidden_state):
    mean_pooling_embeddings = torch.mean(last_hidden_state, dim=1)
    max_pooling_embeddings = torch.max(last_hidden_state, dim=1).values

    return torch.cat((mean_pooling_embeddings, max_pooling_embeddings),
                     dim=1)


def max_pooling(last_hidden_state):
    return torch.max(last_hidden_state, dim=1).values
