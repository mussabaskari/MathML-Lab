"""
regularized_layers.py
-----------------------
From-scratch batch normalization and dropout, plus a configurable
2-hidden-layer network that can optionally use either or both, with the
layer ordering: Linear -> BatchNorm (optional) -> ReLU -> Dropout (optional).

No PyTorch, TensorFlow, Keras, JAX, automatic differentiation, or
sklearn normalization/dropout implementation is used anywhere in this
module.

Gradient convention (matching Parts 6/7/11): backward quantities that
flow between layers (dZ, dA) are PER-SAMPLE gradients dl_i/d(.), not yet
averaged over the batch; the (1/n) batch-average is applied only at the
point a parameter gradient (dW, db, dgamma, dbeta) is finalized - exactly
mirroring how db_l = (1/n) * sum(dZ_l) works elsewhere in this project.
`batchnorm_backward` below returns dgamma/dbeta as raw (unaveraged) sums,
matching the standard textbook derivation; the (1/n) scaling is applied
where it is used, in `backward_regularized_network`.
"""

import numpy as np


# ---------------------------------------------------------------------
# Activation (ReLU only needed here; reuses the same convention as
# Parts 6/7/11)
# ---------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable logistic sigmoid (same technique as Parts 4/6/7/11)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    neg_exp = np.exp(z[~positive])
    out[~positive] = neg_exp / (1.0 + neg_exp)
    return out


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def binary_cross_entropy(y_true, probabilities, eps=1e-12):
    """Binary cross-entropy, identical formula/stability technique to Parts 4/6/7/11."""
    p = np.clip(probabilities, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


# ---------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------

def standardize_batch(x, eps=1e-5):
    """
    Per-feature standardization over the batch dimension (axis 0):
        mu_B = mean(x, axis=0)
        var_B = mean((x - mu_B)^2, axis=0)
        x_hat = (x - mu_B) / sqrt(var_B + eps)
    `eps` prevents division by zero and improves numerical stability
    when a feature has (near-)zero variance in the batch.
    """
    mu = np.mean(x, axis=0)
    var = np.mean((x - mu) ** 2, axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return x_hat, mu, var


# ---------------------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------------------

def batchnorm_forward(x, gamma, beta, eps=1e-5, training=True,
                       running_mean=None, running_var=None, momentum=0.9):
    """
    Batch normalization forward pass, per feature (column) of x (n, d):
        Training:  x_hat = (x - mu_B) / sqrt(var_B + eps)   (batch statistics)
        Inference: x_hat = (x - running_mean) / sqrt(running_var + eps)
        y = gamma * x_hat + beta

    gamma/beta are trainable scale/shift parameters: without them,
    every unit would be forced into a standardized (zero-mean,
    unit-variance) form regardless of what scale is actually useful for
    the next layer; gamma/beta let the network learn an appropriate
    scale and offset on top of the normalized value.

    Training mode also updates running statistics with the documented
    momentum convention:
        running_mean_new = momentum*running_mean_old + (1-momentum)*batch_mean
        running_var_new  = momentum*running_var_old  + (1-momentum)*batch_var

    Returns (out, cache, new_running_mean, new_running_var). In inference
    mode, cache is None (nothing is needed for a backward pass at
    inference time) and the running statistics are returned unchanged.
    """
    if training:
        x_hat, mu, var = standardize_batch(x, eps=eps)
        out = gamma * x_hat + beta
        new_running_mean = momentum * running_mean + (1 - momentum) * mu
        new_running_var = momentum * running_var + (1 - momentum) * var
        cache = (x, x_hat, mu, var, gamma, eps)
        return out, cache, new_running_mean, new_running_var
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        return out, None, running_mean, running_var


def batchnorm_backward(dout, cache):
    """
    Batch normalization backward pass. Derived from
        x_hat = (x - mu)/sqrt(var+eps),  y = gamma*x_hat + beta
    via the chain rule through mu and var (both functions of every
    element of x, since they are batch statistics). Returns
    (dx, dgamma, dbeta) - dgamma/dbeta are raw (unaveraged) sums over
    the batch, matching the standard derivation; callers using this
    project's "average at the parameter gradient" convention apply the
    (1/n) scaling themselves (see backward_regularized_network).
    """
    x, x_hat, mu, var, gamma, eps = cache
    m = x.shape[0]
    std_inv = 1.0 / np.sqrt(var + eps)

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - mu) * -0.5 * std_inv ** 3, axis=0)
    dmu = np.sum(dx_hat * -std_inv, axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
    dx = dx_hat * std_inv + dvar * 2 * (x - mu) / m + dmu / m

    return dx, dgamma, dbeta


# ---------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------

def dropout_forward(x, dropout_rate, rng, training=True):
    """
    Inverted dropout. Training: sample m_i ~ Bernoulli(1-dropout_rate)
    independently per element, output = m_i * x_i / (1-dropout_rate).
    Inference: output = x unchanged (the 1/(1-p) scaling during training
    is exactly what makes this valid - see the expectation derivation in
    the notebook).
    Raises ValueError for an invalid dropout_rate.
    Returns (out, cache) where cache = (mask, dropout_rate, training).
    """
    if not (0 <= dropout_rate < 1):
        raise ValueError(f"dropout_rate must satisfy 0 <= dropout_rate < 1; got {dropout_rate}")
    if training:
        mask = (rng.random(x.shape) >= dropout_rate).astype(float)
        out = mask * x / (1 - dropout_rate)
    else:
        mask = None
        out = x
    return out, (mask, dropout_rate, training)


def dropout_backward(dout, cache):
    """
    Dropout backward pass: the same mask and scaling used forward is
    applied to the upstream gradient (dropout is an elementwise linear
    map given a fixed mask, so its "backward" is exactly this).
    """
    mask, dropout_rate, training = cache
    if training:
        return dout * mask / (1 - dropout_rate)
    return dout


# ---------------------------------------------------------------------
# Configurable regularized network (2 hidden layers, fixed ordering:
# Linear -> BatchNorm (optional) -> ReLU -> Dropout (optional))
# ---------------------------------------------------------------------

def initialize_regularized_network(layer_sizes, seed=42):
    """He-initialized weights/biases for an arbitrary-depth network (biases start at zero)."""
    rng = np.random.default_rng(seed)
    params = {}
    num_layers = len(layer_sizes) - 1
    for l in range(1, num_layers + 1):
        fan_in, fan_out = layer_sizes[l - 1], layer_sizes[l]
        params[f"W{l}"] = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        params[f"b{l}"] = np.zeros(fan_out)
    return params


def initialize_batchnorm_state(layer_sizes):
    """gamma=1, beta=0, running_mean=0, running_var=1 for every hidden layer (not the output layer)."""
    state = {}
    num_layers = len(layer_sizes) - 1
    for l in range(1, num_layers):  # hidden layers only
        state[f"gamma{l}"] = np.ones(layer_sizes[l])
        state[f"beta{l}"] = np.zeros(layer_sizes[l])
        state[f"running_mean{l}"] = np.zeros(layer_sizes[l])
        state[f"running_var{l}"] = np.ones(layer_sizes[l])
    return state


def forward_regularized_network(X, params, layer_sizes, use_batchnorm=False, bn_state=None,
                                 dropout_rate=0.0, rng=None, training=True, bn_momentum=0.9):
    """
    Forward pass: for every hidden layer, Linear -> BatchNorm (if
    use_batchnorm) -> ReLU -> Dropout (if dropout_rate > 0); output layer
    is Linear -> sigmoid. `bn_state`'s running statistics are updated
    in place when training=True and use_batchnorm=True.
    """
    num_layers = len(layer_sizes) - 1
    cache = {"A0": X}
    A = X
    for l in range(1, num_layers + 1):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        if l < num_layers:
            if use_batchnorm:
                Z_norm, bn_cache, new_rm, new_rv = batchnorm_forward(
                    Z, bn_state[f"gamma{l}"], bn_state[f"beta{l}"], eps=1e-5, training=training,
                    running_mean=bn_state[f"running_mean{l}"], running_var=bn_state[f"running_var{l}"],
                    momentum=bn_momentum,
                )
                if training:
                    bn_state[f"running_mean{l}"] = new_rm
                    bn_state[f"running_var{l}"] = new_rv
                cache[f"bn_cache{l}"] = bn_cache
            else:
                Z_norm = Z
            A_relu = relu(Z_norm)
            cache[f"Z{l}"] = Z_norm
            if dropout_rate > 0:
                A, drop_cache = dropout_forward(A_relu, dropout_rate, rng, training=training)
                cache[f"drop_cache{l}"] = drop_cache
            else:
                A = A_relu
            cache[f"A{l}"] = A
        else:
            A = sigmoid(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
    return A, cache


def backward_regularized_network(y, cache, params, layer_sizes, use_batchnorm=False, dropout_rate=0.0):
    """
    Backward pass mirroring forward_regularized_network's layer order in
    reverse: dZ_L = A_L - y (sigmoid+BCE), then per hidden layer,
    Dropout-backward -> ReLU-derivative -> BatchNorm-backward (if used)
    -> Linear-backward. Applies the project's (1/n) parameter-gradient
    averaging convention to dgamma/dbeta as well as dW/db.
    """
    num_layers = len(layer_sizes) - 1
    n = cache["A0"].shape[0]
    y_col = y.reshape(-1, 1)
    grads = {}

    dZ = cache[f"A{num_layers}"] - y_col
    for l in range(num_layers, 0, -1):
        A_prev = cache[f"A{l - 1}"]
        grads[f"dW{l}"] = (1.0 / n) * (A_prev.T @ dZ)
        grads[f"db{l}"] = (1.0 / n) * np.sum(dZ, axis=0)

        if l > 1:
            dA = dZ @ params[f"W{l}"].T
            if dropout_rate > 0:
                dA = dropout_backward(dA, cache[f"drop_cache{l - 1}"])
            dZ_norm = dA * relu_derivative(cache[f"Z{l - 1}"])
            if use_batchnorm:
                dZ, dgamma_raw, dbeta_raw = batchnorm_backward(dZ_norm, cache[f"bn_cache{l - 1}"])
                grads[f"dgamma{l - 1}"] = dgamma_raw / n
                grads[f"dbeta{l - 1}"] = dbeta_raw / n
            else:
                dZ = dZ_norm

    return grads


def train_regularized_network(X, y, layer_sizes, learning_rate, iterations,
                               use_batchnorm=False, dropout_rate=0.0, l2_lambda=0.0,
                               seed=42, bn_momentum=0.9, divergence_threshold=1e10):
    """
    Batch gradient descent training loop for the configurable network.
    L2 regularization (if l2_lambda > 0) adds lambda*W to dW for every
    weight matrix (biases, gamma, beta are not L2-regularized).
    Returns a dict with "params", "bn_state" (or None), "loss_history", "diverged".
    """
    params = initialize_regularized_network(layer_sizes, seed=seed)
    bn_state = initialize_batchnorm_state(layer_sizes) if use_batchnorm else None
    rng = np.random.default_rng(seed) if dropout_rate > 0 else None

    loss_history = []
    diverged = False
    num_layers = len(layer_sizes) - 1

    for _ in range(iterations):
        A_final, cache = forward_regularized_network(
            X, params, layer_sizes, use_batchnorm=use_batchnorm, bn_state=bn_state,
            dropout_rate=dropout_rate, rng=rng, training=True, bn_momentum=bn_momentum,
        )
        loss = binary_cross_entropy(y, A_final.flatten())
        loss_history.append(loss)
        if not np.isfinite(loss) or loss > divergence_threshold:
            diverged = True
            break

        grads = backward_regularized_network(y, cache, params, layer_sizes,
                                              use_batchnorm=use_batchnorm, dropout_rate=dropout_rate)
        for l in range(1, num_layers + 1):
            dW = grads[f"dW{l}"] + (l2_lambda * params[f"W{l}"] if l2_lambda > 0 else 0.0)
            params[f"W{l}"] = params[f"W{l}"] - learning_rate * dW
            params[f"b{l}"] = params[f"b{l}"] - learning_rate * grads[f"db{l}"]
            if use_batchnorm and l < num_layers:
                bn_state[f"gamma{l}"] = bn_state[f"gamma{l}"] - learning_rate * grads[f"dgamma{l}"]
                bn_state[f"beta{l}"] = bn_state[f"beta{l}"] - learning_rate * grads[f"dbeta{l}"]

    return {"params": params, "bn_state": bn_state, "loss_history": np.array(loss_history), "diverged": diverged}


def predict_regularized(X, params, layer_sizes, use_batchnorm=False, bn_state=None, threshold=0.5):
    """Inference-mode prediction: BatchNorm uses running statistics, dropout is identity."""
    A_final, _ = forward_regularized_network(
        X, params, layer_sizes, use_batchnorm=use_batchnorm, bn_state=bn_state,
        dropout_rate=0.0, rng=None, training=False,
    )
    return (A_final.flatten() >= threshold).astype(int)


def evaluate_loss_regularized(X, y, params, layer_sizes, use_batchnorm=False, bn_state=None):
    """Inference-mode BCE data loss (no dropout, BatchNorm uses running statistics)."""
    A_final, _ = forward_regularized_network(
        X, params, layer_sizes, use_batchnorm=use_batchnorm, bn_state=bn_state,
        dropout_rate=0.0, rng=None, training=False,
    )
    return binary_cross_entropy(y, A_final.flatten())
