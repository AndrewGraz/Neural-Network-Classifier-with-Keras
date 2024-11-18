# Andrew Graziano
# 1002165893
# CSE-4309-001
import numpy as np
import tensorflow as tf
import os

def nn_keras(directory, dataset, layers, units_per_layer, epochs):
    # Load the training and test data
    (training_set, test_set) = read_uci1(directory, dataset)
    (training_inputs, training_labels) = training_set
    (test_inputs, test_labels) = test_set
    
    # Normalize the data
    max_value = np.max(np.abs(training_inputs))
    training_inputs = training_inputs / max_value
    test_inputs = test_inputs / max_value

    input_shape = training_inputs[0].shape
    number_of_classes = np.max([np.max(training_labels), np.max(test_labels)]) + 1
    
    # Initialize sequential model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    
    # Add hidden layers using sigmoid activation
    for _ in range(layers - 2):  
        model.add(tf.keras.layers.Dense(units_per_layer, activation='sigmoid')) 
    model.add(tf.keras.layers.Dense(number_of_classes, activation='sigmoid'))  

    # Compile model using Adam optimizer  
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy']) 
    
    # Train model with training data
    model.fit(training_inputs, training_labels, epochs=epochs)
    
    # Find predictions 
    prediction = model.predict(test_inputs)

    # Calculate accuracy
    total_accuracy = 0
    num_test_samples = len(test_labels)
    for object_id in range(num_test_samples):
        nn_output = prediction[object_id].flatten() # Find output for each test 
        predicted_class = np.argmax(nn_output) 
        true_class = test_labels[object_id][0] # Find the true class
        
        # Check for any ties and calculate accuracy
        (indices,) = np.nonzero(nn_output == nn_output[predicted_class])
        number_of_ties = np.prod(indices.shape)
        if predicted_class == true_class:
            accuracy = 1.0 / number_of_ties
        else:
            accuracy = 0.0
        total_accuracy += accuracy

        # Print results for each test
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' %
              (object_id, predicted_class, true_class, accuracy))
    
    # Find and print the classification accuracy
    classification_accuracy = total_accuracy / num_test_samples
    print('classification accuracy=%6.4f' % classification_accuracy)


def read_uci_file(pathname, labels_to_ints, ints_to_labels):
    if not(os.path.isfile(pathname)):
        print("read_data: %s not found", pathname)
        return None

    in_file = open(pathname)
    file_lines = in_file.readlines()
    in_file.close()

    rows = len(file_lines)
    if (rows == 0):
        print("read_data: zero rows in %s", pathname)
        return None
        
    
    cols = len(file_lines[0].split())
    data = np.zeros((rows, cols-1))
    labels = np.zeros((rows,1))
    for row in range(0, rows):
        line = file_lines[row].strip()
        items = line.split()
        if (len(items) != cols):
            print("read_data: Line %d, %d columns expected, %d columns found" %(row, cols, len(items)))
            return None
        for col in range(0, cols-1):
            data[row][col] = float(items[col])
        
        # the last column is a string representing the class label
        label = items[cols-1]
        if (label in labels_to_ints):
            ilabel = labels_to_ints[label]
        else:
            ilabel = len(labels_to_ints)
            labels_to_ints[label] = ilabel
            ints_to_labels[ilabel] = label
        
        labels[row] = ilabel

    labels = labels.astype(int)
    return (data, labels)


def read_uci1(directory, dataset_name):
    training_file = directory + "/" + dataset_name + "_training.txt"
    test_file = directory + "/" + dataset_name + "_test.txt"

    labels_to_ints = {}
    ints_to_labels = {}

    (train_data, train_labels) = read_uci_file(training_file, labels_to_ints, ints_to_labels)
    (test_data, test_labels) = read_uci_file(test_file, labels_to_ints, ints_to_labels)
    return ((train_data, train_labels), (test_data, test_labels))
