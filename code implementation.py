from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scipy


def load_and_center_dataset(filename):
    # Your implementation goes here!

    data = np.load('YaleB_32x32.npy')
    mu = np.mean(data, axis=0)
    center_function = data-mu
    return(center_function)

def get_covariance(dataset):
    # Your implementation goes here!

    cor = (dataset.T @ dataset)/(len(dataset)-1)

    return(cor)

def get_eig(S, m):
    # Your implementation goes here!

    w, v = eigh(S, subset_by_index=[len(S)-m, len(S)-1])

    w = np.diag(w, k=0)
    w = np.flip(w)
    v = np.fliplr(v)

    return(w, v)

def get_eig_prop(S, prop):
    # Your implementation goes here!

    a_value, b_vector = eigh(S)
    a_value_sum = sum(a_value)
    a_value_initiate = prop * a_value_sum
    a_value, b_vector = eigh(S, subset_by_value=[a_value_initiate, a_value_sum])

    a_value = np.diag(a_value, k=0)
    a_value = np.flip(a_value)
    b_vector = np.fliplr(b_vector)

    return(a_value, b_vector)

def project_image(image, U):
    # Your implementation goes here!
    projection = np.zeros((len(image),))

    for i in range(len(U[0])):
        my_array = U[:, i]
        aij = np.dot(np.transpose(my_array), image)
        # projection += np.dot(my_array, aij)
        projection += my_array * aij

    return(projection)

def display_image(orig, proj):
    # Your implementation goes here!

    o = np.reshape(orig, (32, 32))
    p = np.reshape(proj, (32, 32))
    o = np.transpose(o)
    p = np.transpose(p)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    shw = ax1.imshow(o, aspect='equal')
    bar = plt.colorbar(shw)
    ax1.set_title('Original')

    shw2 = ax2.imshow(p, aspect='equal')
    bar1 = plt.colorbar(shw2)
    ax2.set_title('Projection')
    plt.show()
    return(proj)

