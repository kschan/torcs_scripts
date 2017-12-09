import glob, pickle
import numpy as np

def concatenate_training_data():
    logged_input_files = glob.glob("../../good_logs/logged_inputs*")
    logged_inputs = np.concatenate([np.load(file) for file in logged_input_files], axis = 0)
    print("[ INFO ] Logged_inputs: ", logged_inputs.shape)

    logged_state_files = glob.glob("../../good_logs/logged_states*")
    logged_states = np.concatenate([np.load(file) for file in logged_state_files], axis = 0)
    print("[ INFO ] Logged_states: ", logged_states.shape)

    return logged_states, logged_inputs

def get_next_batch(train_dataset, train_labels, batch_size):
    batch_indices = np.random.randint(train_labels.shape[0], size=batch_size)
    batch_dataset = train_dataset[batch_indices, :]
    batch_labels = train_labels[batch_indices, :]

    return batch_dataset, batch_labels

def load_data(model):
    logged_states, logged_inputs = concatenate_training_data()
    num_samples    = logged_states.shape[0]
    random_indices = np.random.permutation(num_samples)

    train_indices  = random_indices[:(num_samples - 20000)]
    train_dataset  = logged_states[train_indices, :]
    train_dataset  = train_dataset[:, model.states_idxs]
    train_labels   = logged_inputs[train_indices, :]
    train_labels   = train_labels[:, model.output_idxs].reshape((-1, model.num_outputs))

    valid_indices  = random_indices[(num_samples-20000):(num_samples-10000)]
    valid_dataset  = logged_states[valid_indices, :]
    valid_dataset  = valid_dataset[:, model.states_idxs]
    valid_labels   = logged_inputs[valid_indices, :]
    valid_labels   = valid_labels[:, model.output_idxs].reshape((-1, model.num_outputs))

    test_indices   = random_indices[(num_samples-10000):]
    test_dataset   = logged_states[test_indices, :]
    test_dataset   = test_dataset[:, model.states_idxs]
    test_labels    = logged_inputs[test_indices, :]
    test_labels    = test_labels[:, model.output_idxs].reshape((-1, model.num_outputs))

    return train_dataset, train_labels, \
           valid_dataset, valid_labels, \
           test_dataset , test_labels

def read_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_data_pid(model, INPUT, OUTPUT, PID_PATH):
    # Initialize
    input_list = []
    for _ in INPUT:
        input_list.append([])
    output_list = []
    for _ in OUTPUT:
        output_list.append([])

    # Read data
    dict_paths = glob.glob(PID_PATH)
    for dict_path in sorted(dict_paths):
        dict_file = read_dict(dict_path)

        for input_idx, input_val in enumerate(INPUT):
            input_list[input_idx].append(dict_file[input_val])
        for output_idx, output_val in enumerate(OUTPUT):
            output_list[output_idx].append(dict_file[output_val])

    # Concat data
    train_dataset = np.array([], dtype=np.float64).reshape((len(input_list[0]), 0))
    for input_idx in range(len(INPUT)):
        train_dataset = np.concatenate((train_dataset, np.asarray(input_list[input_idx], dtype=np.float64).reshape((-1, 1))), axis=1)

    train_label = np.array([], dtype=np.float64).reshape((len(output_list[0]), 0))
    for output_idx in range(len(OUTPUT)):
        train_label = np.concatenate((train_label, np.asarray(output_list[output_idx], dtype=np.float64).reshape((-1, 1))), axis=1)

    # Normalize train data
    for input_idx in range(len(INPUT)):
        max_val = np.max(train_dataset[:, input_idx])
        min_val = np.min(train_dataset[:, input_idx])
        print('[ INFO ] For', INPUT[input_idx],'-> max_val is:', max_val, 'and min_val is:', min_val)
        train_dataset[:, input_idx] = 2.0*(train_dataset[:, input_idx] - min_val)/(float(max_val - min_val)) - 1

    return train_dataset, train_label, \
           None, None, \
           None, None
