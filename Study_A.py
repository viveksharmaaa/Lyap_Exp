import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from numpy.linalg import qr as np_qr
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

os.makedirs("figures/Study_A", exist_ok=True)

# Create folders if they don't exist
os.makedirs("figures/Study_A/LE", exist_ok=True)
os.makedirs("figures/Study_A/trainloss", exist_ok=True)


# ============================================================
#  Device / dtype
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32   # change to float64 if you want more accuracy


# ============================================================
#  Helper function for computing \lambda_max through DNS
# ============================================================
# ============================================================

def flatten_params_from_model(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_params_from_flat(model, flat_vec, ref_params):
    """
    Set model parameters from a flat vector, using ref_params for shapes.
    """
    with torch.no_grad():
        new_tensors = unflatten_like(flat_vec, ref_params)
        for p, new in zip(model.parameters(), new_tensors):
            p.copy_(new)


def sgd_step_in_place(model, xb, yb, lr, loss_fn):
    """
    One plain SGD step (no momentum, no weight decay) applied in-place.
    """
    params = list(model.parameters())
    # enable gradients for this one step
    for p in params:
        p.requires_grad_(True)

    yhat = model(xb)
    loss = loss_fn(yhat, yb)
    grads = torch.autograd.grad(loss, params, create_graph=False)

    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= lr * g

    # turn off grads again
    for p in params:
        p.requires_grad_(False)

    return float(loss.item())

# ============================================================
#  Flatten / unflatten helpers
# ============================================================
def flatten_params(tensors):
    """Flatten list/tuple of tensors to a 1D vector."""
    return torch.cat([t.reshape(-1) for t in tensors])

def unflatten_like(vec, ref_tensors):
    """
    Unflatten a 1D tensor 'vec' into a list of tensors with
    the same shapes as 'ref_tensors'.
    """
    out = []
    idx = 0
    for t in ref_tensors:
        numel = t.numel()
        out.append(vec[idx: idx + numel].reshape(t.shape))
        idx += numel
    assert idx == vec.numel()
    return out

def grad_flat(model, Xb, Yb):
    """
    Compute gradient of loss wrt all model parameters and flatten to a 1D vector.
    """
    params = list(model.parameters())
    loss = F.mse_loss(model(Xb), Yb)
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
    return flatten_params(grads)

def get_param_list(model):
    return [p for p in model.parameters()]

# ============================================================
# Dataset: Boston Housing Dataset
# ============================================================

def generate_data(seed=0,Nsub=None):
    """
    Boston Housing data loaded from CMU repository http://lib.stat.cmu.edu/datasets/boston (raw text format).

    The dataset contains 506 samples with 13 numerical input features (e.g., CRIM, ZN, INDUS, RM, LSTAT, TAX, etc.) and a single continuous target value (median home price).
    Returns X, Y as float32 arrays in the same format as the synthetic dataset.
    """

    rng = np.random.RandomState(seed)

    # Load data from URL
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

    # Combine the two alternating row structures
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    Y = raw_df.values[1::2, 2].reshape(-1, 1)

    # Normalize X and Y for stable optimization
    scaler_X = StandardScaler().fit(X)
    scaler_Y = StandardScaler().fit(Y)

    X = scaler_X.transform(X)
    Y = scaler_Y.transform(Y)

    # Convert to float32
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    # Optional subsampling to enforce N if specified
    if Nsub is not None:
        idx = rng.choice(X.shape[0], N, replace=False)
        X = X[idx]
        Y = Y[idx]

    return X, Y

# ============================================================
# 5. HVP via autograd
# ============================================================

def hvp_simple(model, Xb, Yb, v_flat):
    params = [p for p in model.parameters() if p.requires_grad]
    loss_fn = nn.MSELoss()

    # Forward pass
    yhat = model(Xb)
    loss = loss_fn(yhat, Yb)

    # First gradient
    grads = torch.autograd.grad(
        loss, params,
        create_graph=True,
        retain_graph=True
    )

    # Unflatten v like parameters
    v_tensors = unflatten_like(v_flat, params)

    # Compute inner product g·v
    g_dot_v = torch.zeros(1, device=device, dtype=dtype)
    for g, v in zip(grads, v_tensors):
        g_dot_v = g_dot_v + (g * v).sum()

    # Second gradient = Hessian-vector product
    Hv = torch.autograd.grad(
        g_dot_v, params,
        retain_graph=False,
        allow_unused=False
    )

    return flatten_params(Hv)


# ============================================================
# 5. Maximum eigenvalue of full-batch Hessian using power iteration
# ============================================================
def power_iteration_hessian(model, Xb, Yb, iters=100, tol=1e-6):
    """
    lambda_new: approximated maximum eigenvalue of Hessian H of loss σ_max(H)
    v: corresponding eigenvector direction

    Taken from Z. Yao, A. Gholami, K Keutzer, M. Mahoney. PyHessian: Neural Networks Through the Lens of the Hessian, Spotlight at ICML workshop on Beyond First-Order Optimization Methods in Machine Learning, 2020
    """

    Xb = torch.from_numpy(Xb).to(device=device, dtype=dtype)
    Yb= torch.from_numpy(Yb).to(device=device, dtype=dtype)

    params = list(model.parameters())
    D = sum(p.numel() for p in params)

    # random initial vector
    v = torch.randn(D, device=device, dtype=dtype)
    v = v / torch.norm(v)

    lambda_old = 0.0

    for k in range(iters):
        # Hv using double backprop
        Hv = hvp_simple(model, Xb, Yb, v)

        # Rayleigh quotient
        lambda_new = torch.dot(v, Hv).item()

        # normalize
        v = Hv / (torch.norm(Hv) + 1e-12)

        if abs(lambda_new - lambda_old) < tol:
            break

        lambda_old = lambda_new

    return lambda_new, v


# ============================================================
# Sanity Check of HVP against finite differences
# ============================================================
def hvp_finite_difference(model, Xb, Yb, v_flat, eps=1e-3):
    """
    Finite difference approximation of H v:
        Hv ≈ (∇L(w+eps v) - ∇L(w-eps v)) / (2 eps)
    """
    params = list(model.parameters())
    w0 = flatten_params(params).detach().clone()

    # helper: set parameters
    def set_flat_params(w_flat):
        idx = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.copy_(w_flat[idx:idx+n].view_as(p))
                idx += n

    # g(w + eps*v)
    set_flat_params(w0 + eps * v_flat)
    g_plus = grad_flat(model, Xb, Yb)

    # g(w - eps*v)
    set_flat_params(w0 - eps * v_flat)
    g_minus = grad_flat(model, Xb, Yb)

    # restore original weights
    set_flat_params(w0)

    return (g_plus - g_minus) / (2.0 * eps)

def sanity_check_hvp(model, Xb, Yb, eps=1e-4):
    # random vector v
    params = list(model.parameters())
    D = sum(p.numel() for p in params)

    v = torch.randn(D, device=device, dtype=dtype)
    v /= v.norm()

    Hv_hvp = hvp_simple(model, Xb, Yb, v)

    Hv_fd = hvp_finite_difference(
        model, Xb, Yb, v, eps=eps
    )

    rel_err = torch.norm(Hv_hvp - Hv_fd) / (torch.norm(Hv_fd) + 1e-12)

    if rel_err > 1e-1:
        print("========== HVP SANITY CHECK ==========")
        print("||Hv_pearl|| =", Hv_hvp.norm().item())
        print("||Hv_fd||    =", Hv_fd.norm().item())
        print("High FD error:", rel_err.item())
        print("=======================================\n")

    return rel_err.item()



# ============================================================
#  3 layered Multi Layered Perceptron with tanh() activation function
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, width, out_dim):  # g controls scale
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, out_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
#  Training to convergence function  (SGD)
# ============================================================

def train_to_convergence(model,
                         X_train, Y_train,
                         X_test, Y_test,
                         lr=1e-4,
                         max_steps=40000,
                         batch_size=32,
                         min_steps=10000,
                         tol=1e-2,
                         verbose=False,seed=0,train_plots=False):

    assert min_steps > 0 and min_steps <= max_steps
    model.train()
    N_train = X_train.shape[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    X_train_t = torch.from_numpy(X_train).to(device=device, dtype=dtype)
    Y_train_t = torch.from_numpy(Y_train).to(device=device, dtype=dtype)
    X_test_t  = torch.from_numpy(X_test).to(device=device, dtype=dtype)
    Y_test_t  = torch.from_numpy(Y_test).to(device=device, dtype=dtype)

    train_losses = []

    np.random.seed(0) #for reproducibility of batch sampling

    for t in range(max_steps):
        idx = np.random.choice(N_train, batch_size, replace=True)
        xb = X_train_t[idx]
        yb = Y_train_t[idx]

        optimizer.zero_grad()
        yhat = model(xb)
        loss = loss_fn(yhat, yb)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if verbose and (t + 1) % 5000 == 0:
            print(f"  [step {t+1}] train loss = {loss.item():.4e}")

        if t > min_steps:
            if np.mean(train_losses[-200:]) < tol:
                if verbose:
                    print(f"Converged at step {t + 1}, smoothed loss {np.mean(train_losses[-200:]):.3e}")
                break


    if train_plots:
        plt.figure(figsize=(7, 6))
        plt.plot(train_losses)
        plt.xlabel("Epochs")
        plt.ylabel("train_loss")
        plt.title("Training Loss convergence")
        fname_train = f"figures/Study_A/trainloss/train_seed_{seed}_lr_{lr}.png"
        plt.tight_layout()
        plt.savefig(fname_train, dpi=200)
        plt.close()

    # Compute the test loss D_test
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_test_t), Y_test_t).item()

    return train_losses[-1],test_loss


# ============================================================
#  Lyapunov exponent at frozen solution
# ============================================================

def compute_lyapunov_at_solution(model,
                                 X_train, Y_train,
                                 lr,
                                 iterations=8000,
                                 k=10,
                                 batch_size=32,
                                 renorm_every=15,
                                 check_hvp=True,
                                 seed=0,
                                 spec_plots=True,nStepTransient=200,dns=True):
    """
    Estimate the top-k Lyapunov exponents of the SGD map
        θ_{t+1} = θ_t - lr * ∇_θ L(θ_t; minibatch)
    using the Benettin–Wolf QR method.

    Returns
    -------
    lambda_max : float
        Estimate of the largest Lyapunov exponent.
    lambda_trajectory : list[np.ndarray]
        List of running spectra over time, shape (num_renorms, k).
    """

    # ---------- Reproducibility ----------
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.eval()  # we evaluate around the (approx) solution; no parameter updates

    # Move data to torch
    X_train_t = torch.from_numpy(X_train).to(device=device, dtype=dtype)
    Y_train_t = torch.from_numpy(Y_train).to(device=device, dtype=dtype)
    N_train = X_train.shape[0]

    lambda_trajectory = []
    num_ons=(iterations-nStepTransient)//renorm_every


    # ---------- Dimension of parameter space ----------
    with torch.no_grad():
        D = flatten_params(get_param_list(model)).numel()

    # ---------- Initialize tangent basis (D x k) with QR ----------
    q0 = np.random.randn(D, k)
    q_np, _ = np_qr(q0)  # q_np is orthonormal in R^{D x k}
    q = torch.from_numpy(q_np).to(device=device, dtype=dtype)  # (D, k)

    # Accumulate log of expansion factors
    LS = np.zeros(k, dtype=np.float64)  # log-magnitudes accumulator
    LSAll=np.zeros((num_ons,k))
    total_steps = 0                      # total SGD steps accounted for
    num_renorms = -1

    if dns:
        eps = 1e-4

        # Two evolving copies of the model
        base_model = copy.deepcopy(model).to(device)
        pert_model = copy.deepcopy(model).to(device)

        # Store param structure for flatten/unflatten
        ref_params = [p.detach().clone() for p in base_model.parameters()]

        # Initial perturbation direction
        theta0 = flatten_params_from_model(base_model)
        D = theta0.numel()

        d0 = torch.randn(D, device=device, dtype=dtype)
        d0 = eps * d0 / (d0.norm() + 1e-16)

        # Set perturbed parameters
        set_params_from_flat(pert_model, theta0 + d0, ref_params)

        # DNS accumulation
        loss_fn=nn.MSELoss()
        DNS_LS = 0.0
        DNS_steps = 0.0
        dns_index = -1
        DNS_traj = []
        dns_norms = np.zeros(num_ons)

    # How often to sanity-check the HVP (in SGD steps) with finite-difference approximation
    hvp_check_every = max(1, iterations // 10) if check_hvp else None

    for step in range(iterations):
        # ----- Sample minibatch -----
        idx = np.random.choice(N_train, batch_size, replace=True)
        xb = X_train_t[idx]
        yb = Y_train_t[idx]


        # ----- Skip Lyapunov computation until transient is over -----
        if step >= nStepTransient:

            if dns:
                # Base model SGD step
                base_loss = sgd_step_in_place(base_model, xb, yb, lr, loss_fn)
                # Perturbed model SGD step
                pert_loss = sgd_step_in_place(pert_model, xb, yb, lr, loss_fn)

            # ----- Apply linearized map: v -> (I - lr * H) v for each tangent vector -----
            with torch.enable_grad():
                # Ensure model params are being tracked for autograd in hvp_simple
                for j in range(k):
                    # v is a parameter-space direction (flattened)
                    v = q[:, j].detach().view(-1)      # (D,)
                    Hv = hvp_simple(model, xb, yb, v)  # must return same shape as v
                    q[:, j] = (v - lr * Hv).detach()   # update tangent vector

            # ----- Optional HVP sanity check (finite-difference) -----
            if check_hvp and ((step + 1) % hvp_check_every == 0):
                sanity_check_hvp(model, xb, yb, eps=1e-1)  # eps reasonably small



            # ----- Periodic QR re-orthonormalization -----
            if (step + 1) % renorm_every == 0:
                # QR on GPU (faster than round-tripping through NumPy)
                # q: (D, k), r: (k, k)
                num_renorms += 1
                q, r = torch.linalg.qr(q, mode="reduced")

                # Diagonal entries of R encode local expansion/contraction
                diag_r = torch.diag(r).abs().clamp_min(1e-16)  # avoid log(0)
                local_exponents = np.log(diag_r.detach().cpu().numpy().astype(np.float64))/(renorm_every *lr)
                LSAll[num_renorms, :] = local_exponents
                LS += np.log(diag_r.detach().cpu().numpy().astype(np.float64))
                total_steps += renorm_every * lr

                if dns:
                    dns_index += 1

                    # --- measure separation ---
                    theta_base = flatten_params_from_model(base_model)
                    theta_pert = flatten_params_from_model(pert_model)

                    delta = theta_pert - theta_base
                    norm_delta = float(delta.norm())

                    # store norms for debugging
                    dns_norms[dns_index] = norm_delta

                    # accumulate log-stretch
                    DNS_LS += np.log(norm_delta / eps)
                    DNS_steps += renorm_every * lr

                    # --- renormalize back to eps ---
                    if norm_delta < 1e-32:
                        norm_delta = 1e-32

                    delta_new = (eps / norm_delta) * delta

                    with torch.no_grad():
                        set_params_from_flat(pert_model, theta_base + delta_new, ref_params)

                    # running DNS estimate
                    DNS_traj.append(DNS_LS / DNS_steps)
    # ---------- Final spectrum ----------
    if total_steps > 0:
        final_spectrum =  LS / float(total_steps)
    else:
        final_spectrum = np.zeros(k, dtype=np.float64)

    DNS_final = None
    if dns and DNS_steps > 0:
        DNS_final = DNS_LS / DNS_steps

    # ---------- Plot spectrum convergence ----------
    if spec_plots and len(LSAll) > 0:
        arr = np.stack(LSAll, axis=0)  # (num_renorms, k)
        plt.figure(figsize=(7, 6))
        for i in range(k):
            plt.plot(arr[:, i], label=fr"$\lambda_{{{i+1}}}$")
        plt.xlabel(r"Time (SGD steps $*\, \eta$)")
        plt.ylabel(r"Local $\lambda_k$")
        plt.title("Local exponents")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname_lyap = f"figures/Study_A/LE/lyapunov_seed_{seed}_lr_{lr}.png"
        plt.tight_layout()
        plt.savefig(fname_lyap, dpi=200)
        plt.close()

    # if dns and DNS_traj is not None and len(DNS_traj) > 0:
    #     dns_arr = np.array(DNS_traj)
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(dns_arr, color="black", linestyle="--", linewidth=2.4)
    #
    #     plt.xlabel(r"Time (SGD steps $*\, \eta$)")
    #     plt.ylabel(r"DNS $\lambda_{\max}$")
    #     plt.title(fr"DNS Lyapunov Exponent (lr={lr}, seed={seed})")
    #     plt.grid(True, alpha=0.3)
    #
    #     fname_dns = f"DNS_lyapunov_seed_{seed}_lr_{lr}.png"
    #     plt.tight_layout()
    #     plt.savefig(fname_dns, dpi=200)
    #     plt.close()

    if dns:
        return float(final_spectrum[0]), LSAll, DNS_final, np.array(DNS_traj)
    else:
        return float(final_spectrum[0]), LSAll


# ============================================================
# Single run wrapper: train + Lyapunov
# ============================================================

def run_single_config(X_train, Y_train, X_test, Y_test,
                      n, m, width, lr, seed,
                      max_steps=40000,
                      batch_size=32,
                      hvp_iters=8000,
                      hvp_k=10,
                      verbose=False,train_loss_plots=False):

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(in_dim=n, width=width, out_dim=m).to(device)

    if verbose:
        print(f"  width={width}, lr={lr:.1e},  seed={seed}: training...")

    train_loss,test_loss= train_to_convergence(
        model, X_train, Y_train, X_test, Y_test,
        lr=lr,
        max_steps=max_steps, #must be 10^5
        batch_size=batch_size,
        verbose=verbose,seed=seed,train_plots=train_loss_plots
    )

    if verbose:
        print("Computing Lyapunov spectrum at solution...")

    lam_max,_,lam_max_dns,_ = compute_lyapunov_at_solution(
        model, X_train, Y_train,
        lr=lr,
        iterations=hvp_iters,
        k=hvp_k,
        batch_size=batch_size,
        renorm_every=10,seed =seed,dns=True #was 5
    )


    print("HVP-based λ_max:", lam_max)
    # Compute maximum eigenvalue of the Hessian using full training batch X_train
    sigma_max, _ =  power_iteration_hessian(model, X_train, Y_train, iters=100, tol=1e-6)  #maximum eigenvalue of Hessian

    return lam_max, test_loss,train_loss,sigma_max,lam_max_dns

# ============================================================
# 8. Main sweep + correlation analysis
# ============================================================

if __name__ == "__main__":
    n, m = 13, 1  # Input features dimension, target output dimension
    N = 506       # size of the Boston housing dataset
    data_seed = 0 # fix seed for generating data
    bs = 32         #batch size
    save_results  = True  #Save results to npz file

    #WIDTHS = [20] #20, 50, 100, 200 # width of the hidden layer
    width = 50 # width of the hidden layer 20 is best
    LRS    = [1e-3] #[1e-5,1e-4,1e-3,1e-2,5e-2] #5e-5, 1e-4, 3e-4 # fixed learning rates
    RUNS_PER_CONFIG = 100 # number of runs

    # Fixed dataset
    X, Y =  generate_data(seed=data_seed)
    split = int(0.8 * N)
    X_train, Y_train = X[:split], Y[:split]
    X_test,  Y_test  = X[split:], Y[split:]

    results = []

  #  for width in WIDTHS:
    for lr in LRS:
        for r in range(RUNS_PER_CONFIG):
            seed = 1000 * width + 100 * int(lr * 1e5) + r
            print(f"\n=== Config: seed={seed}, lr={lr:.1e}, run={r} ===")
            lam_max, test_loss,train_loss,sigma_max,lam_max_dns = run_single_config(
                X_train, Y_train, X_test, Y_test,
                n, m, width, lr, seed,
                max_steps=80000,
                batch_size=bs,
                hvp_iters=10000,
                hvp_k=10,
                verbose=True,
                train_loss_plots=True
            )

            print(f"λmax={lam_max:.4e},train_loss={train_loss:.4e},test_loss={test_loss:.4e},sigma_max={sigma_max:.4e},gen_gap={np.abs(train_loss - test_loss):.4e}")
            results.append({
                "width": width,
                "lr": lr,
                "seed": seed,
                "lambda_max": lam_max,
                "test_loss": test_loss,
                "train_loss":train_loss,
                "lambda_max_dns": lam_max_dns,
                "gen_gap": np.abs(train_loss - test_loss),
                "sigma_max":sigma_max
            })


    if save_results:
        # Convert list-of-dicts into a dict-of-lists
        results_np = {key: np.array([d[key] for d in results])
                      for key in results[0].keys()}
        # Save as NPZ
        np.savez("results_random_weight.npz", **results_np)
        print("Saved results to results_random_weight.npz")

