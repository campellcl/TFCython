from tempfile import TemporaryFile
import tensorflow as tf
import pyximport
pyximport.install()
from sklearn.decomposition import _cdnmf_fast
from sklearn.decomposition._nmf import _update_coordinate_descent
from sklearn.utils import check_array
import numpy as np

global NUM_ITERATIONS

def _tf_update_coordinate_descent(X: np.ndarray, W: tf.Variable, Ht: tf.Variable):
    HHt = tf.matmul(tf.transpose(Ht), Ht)
    XHt = tf.matmul(X, Ht)
    return _tf_update_cdnmf_fast(W, HHt, XHt)


def _tf_update_cdnmf_fast(W: tf.Variable, HHt: tf.Variable, XHt: tf.Variable):
    n_components = W.shape[1]
    n_samples = W.shape[0]
    W_updated = tf.Variable(initial_value=W)

    for s in range(n_components):
        for j in range(n_samples):
            grad = -XHt[j, s]
            for r in range(n_components):
                grad += HHt[s, r] * W_updated[j, r]

            hess = HHt[s, s]
            # hess = tf.linalg.diag_part(HHt)[s]
            if hess != 0:
                W_updated[j, s].assign(max(W_updated[j, s] - grad / hess, 0.))
            else:
                print(f"Hessian is 0.0 for component: {s} sample: {j}")
    return W_updated


def _tf_fit_coordinate_descent(X: tf.Variable, W: tf.Variable, H: tf.Variable):
    W_updated = tf.Variable(initial_value=W, trainable=False)
    Ht = tf.Variable(initial_value=tf.transpose(H), trainable=False)

    for i in range(NUM_ITERATIONS):
        print(f"Iteration [{i + 1}/{NUM_ITERATIONS}]:")
        Ht.assign(tf.transpose(H))
        updated_W = _tf_update_coordinate_descent(X=X, W=W, Ht=Ht)
        W.assign(updated_W)

        updated_Ht = _tf_update_coordinate_descent(X=tf.transpose(X), W=Ht, Ht=W)
        H.assign(tf.transpose(updated_Ht))

def _fit_compare_coordinate_descents(X: np.ndarray, W: tf.Variable, H: tf.Variable, num_iterations: int):
    n_components = W.shape[1]
    n_samples = W.shape[0]
    tf_W = tf.Variable(initial_value=W)
    sk_W = W.numpy().copy()
    sk_W = check_array(sk_W, order='C')
    tf_H = tf.Variable(initial_value=H)
    sk_H = H.numpy().copy()
    tf_Ht = tf.Variable(initial_value=tf.transpose(tf_H))
    sk_Ht = tf.transpose(sk_H)

    for i in range(num_iterations):
        print(f"Iteration [{i+1}/{num_iterations}]:")
        violation = 0.0
        tf_Ht.assign(tf.transpose(tf_H))
        sk_Ht = sk_H.T
        sk_Ht = check_array(sk_Ht, order='C')

        # Update SkLearn W:
        violation += _update_coordinate_descent(X=X, W=sk_W, Ht=sk_Ht, l1_reg=0.0, l2_reg=0.0, shuffle=False, random_state=None)
        # Update TensorFlow W:
        updated_W = _tf_update_coordinate_descent(X=X, W=tf_W, Ht=tf_Ht)
        tf_W.assign(updated_W)

        # Run W assertions:
        np.testing.assert_allclose(actual=tf_W, desired=sk_W, rtol=1e-7)

        # Update SkLearn H:
        violation += _update_coordinate_descent(X=X.T, W=sk_Ht, Ht=sk_W, l1_reg=0.0, l2_reg=0.0, shuffle=False, random_state=None)
        sk_H = sk_Ht.T
        # Update TensorFlow H:
        updated_Ht = _tf_update_coordinate_descent(X=X.T, W=tf_Ht, Ht=tf_W)
        tf_H.assign(tf.transpose(updated_Ht))

        # Run H assertions:
        np.testing.assert_allclose(actual=tf_H, desired=sk_H, rtol=1e-7)

def _compare_coordinate_descents(X: tf.Variable, W: tf.Variable, H: tf.Variable, num_iterations: int):
    n_components = W.shape[1]
    n_samples = W.shape[0]

    tf_W = tf.Variable(initial_value=W)
    sk_W = tf.Variable(initial_value=W)
    tf_H = tf.Variable(initial_value=H)
    sk_H = tf.Variable(initial_value=H)

    for i in range(num_iterations):
        print(f"Iteration [{i + 1}/{num_iterations}]")
        tf_Ht = tf.transpose(tf_H)
        sk_Ht = tf.transpose(sk_H)
        tf_HHt = tf.matmul(tf.transpose(tf_Ht), tf_Ht)
        sk_HHt = tf.matmul(tf.transpose(sk_Ht), sk_Ht)
        tf_XHt = tf.matmul(X, tf_Ht)
        sk_XHt = tf.matmul(X, sk_Ht)

        for s in range(n_components):
            for j in range(n_samples):
                tf_grad = -tf_XHt[j, s]
                sk_grad = -sk_XHt[j, s]
                for r in range(n_components):
                    tf_grad += tf_HHt[s, r] * tf_W[j, r]
                    sk_grad += sk_HHt[s, r] * sk_W[j, r]

                tf_hess = tf_HHt[s, s]
                sk_hess = sk_HHt[s, s]
                # hess = tf.linalg.diag_part(HHt)[s]
                if tf_hess != 0:
                    tf_W[j, s].assign(max(tf_W[j, s] - tf_grad / tf_hess, 0.))
                else:
                    print(f"Hessian is 0.0 for component: {s} sample: {j}")

                if sk_hess != 0:
                    sk_W[j, s].assign(max(sk_W[j, s] - sk_grad / sk_hess, 0.))
                else:
                    print(f"Hessian is 0.0 for component: {s} sample: {j}")
        np.testing.assert_allclose(actual=tf_W, desired=sk_W, rtol=1e-7)
        np.testing.assert_allclose(actual=tf_H, desired=sk_H, rtol=1e-7)
    return W


def main():

    _X = np.load("X.npy")
    _X_norm = np.load("X_norm.npy")
    _W_init = np.load("W_init.npy")
    _H_init = np.load("H_init.npy")

    X = tf.Variable(initial_value=_X_norm)
    W_init = tf.Variable(initial_value=_W_init)
    H_init = tf.Variable(initial_value=_H_init)

    # _tf_fit_coordinate_descent(X=X, W=W_init, H=H_init)
    # _compare_coordinate_descents(X=X, W=W_init, H=H_init, num_iterations=4)
    _fit_compare_coordinate_descents(X=_X_norm, W=W_init, H=H_init, num_iterations=4)

if __name__ == '__main__':
    main()