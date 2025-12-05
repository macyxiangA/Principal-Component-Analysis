from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def load_and_center_dataset(filename):
    """
    Load dataset from .npy file and center it by subtracting the mean.

    Args:
        filename (str): Path to the .npy file

    Returns:
        numpy.ndarray: Centered dataset (n x d matrix)
    """
    X = np.load(filename)
    mean = np.mean(X, axis=0)
    X_centered = X - mean 
    return X_centered

def get_covariance(dataset):
    """
    Calculate the sample covariance matrix of the dataset.

    Args:
        dataset (numpy.ndarray): Centered dataset (n x d matrix)

    Returns:
        numpy.ndarray: Covariance matrix (d x d matrix)
    """
    X = np.asarray(dataset, dtype=float)
    n = X.shape[0]
    S = (X.T @ X) / (n - 1)
    return S

def get_eig(S, k):
    """
    Get the k largest eigenvalues and corresponding eigenvectors.

    Args:
        S (numpy.ndarray): Covariance matrix (d x d)
        k (int): Number of largest eigenvalues/eigenvectors to return

    Returns:
        tuple: (Lambda, U) where Lambda is diagonal matrix of eigenvalues
               and U is matrix of corresponding eigenvectors as columns
    """
    S = np.asarray(S, dtype=float)
    d = S.shape[0]
    w, U = eigh(S, subset_by_index=[d - k, d - 1])
    w = w[::-1]
    U = U[:, ::-1]
    Lambda = np.diag(w)
    return Lambda, U

def get_eig_prop(S, prop):
    """
    Get eigenvalues and eigenvectors that explain more than prop proportion of variance.

    Args:
        S (numpy.ndarray): Covariance matrix (d x d)
        prop (float): Minimum proportion of variance to explain (0 <= prop <= 1)

    Returns:
        tuple: (Lambda, U) where Lambda is diagonal matrix of eigenvalues
               and U is matrix of corresponding eigenvectors as columns
    """
    S = np.asarray(S, dtype=float)
    d = S.shape[0]
    w, U = eigh(S)
    total = w.sum()
    if total <= 0:
        return np.zeros((0, 0)), np.zeros((d, 0))
    ratios = w / total
    sel = np.where(ratios > prop)[0]
    if sel.size == 0:
        return np.zeros((0, 0)), np.zeros((d, 0))
    sel = sel[::-1]
    wSel = w[sel]       
    USel = U[:, sel] 
    Lambda = np.diag(wSel)

    return Lambda, USel


def project_and_reconstruct_image(image, U):
    """
    Project image to PCA subspace and reconstruct it back to original dimension.

    Args:
        image (numpy.ndarray): Flattened image vector (d x 1)
        U (numpy.ndarray): Matrix of eigenvectors (d x m)

    Returns:
        numpy.ndarray: Reconstructed image as flattened d x 1 vector
    """
    x = np.asarray(image, dtype=float).reshape(-1) 
    U = np.asarray(U, dtype=float)
    alpha = U.T @ x
    x_hat = U @ alpha 
    return x_hat

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    """
    Display three images side by side: original high-res, original, and reconstructed.

    Args:
        im_orig_fullres (numpy.ndarray): Original high-resolution image
        im_orig (numpy.ndarray): Original low-resolution image
        im_reconstructed (numpy.ndarray): Reconstructed image from PCA

    Returns:
        tuple: (fig, ax1, ax2, ax3) matplotlib figure and axes objects
    """

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()
    
    img_hr = im_orig_fullres
    if img_hr.ndim == 1 and img_hr.size == 218*178*3:
        img_hr = img_hr.reshape(218, 178, 3)
    ax1.imshow(img_hr)
    ax1.set_title("Original High Res")
    ax1.axis("off") 
    
    im_orig = im_orig.reshape(60, 50)
    im2 = ax2.imshow(im_orig, aspect="equal", cmap="gray")
    ax2.set_title("Original")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2)
    
    im_recon = im_reconstructed.reshape(60, 50)
    im3 = ax3.imshow(im_recon, aspect="equal", cmap="gray")
    ax3.set_title("Reconstructed")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3)

    return fig, ax1, ax2, ax3


# =============================================================================
# N-GRAM LANGUAGE MODEL FUNCTIONS
# =============================================================================

class NGramCharLM:
    """
    N-gram Character Language Model
    
    This class implements an n-gram character-level language model.
    Students should implement the missing methods.
    """
    
    def __init__(self, n=5):
        """
        Initialize the n-gram language model.
        
        Args:
            n (int): The order of the n-gram model (must be >= 1)
        """
        assert n >= 1, "n must be at least 1"
        self.n = n
        self.counts = {}  # dict[str, dict[str, int]] - context -> {char: count}
        self.vocab = set()  # set of all characters seen during training
        self.trained = False
    
    def fit(self, text: str):
        """
        Train the n-gram model on the given text.
        
        Args:
            text (str): Training text
            
        Returns:
            NGramCharLM: self (for method chaining)


        Examples:
        If n=3 and text="banana", then counts should look like:
        {
            ""   : {"b": 1},         # at i=0, ctx="", ch="b"
            "b"  : {"a": 1},         # at i=1, ctx="b", ch="a"
            "ba" : {"n": 1},         # at i=2, ctx="ba", ch="n"
            "an" : {"a": 2},         # at i=3,5, ctx="an", ch="a"
            "na" : {"n": 1}          # at i=4, ctx="na", ch="n"
        }

        If n=2 and text="abab", then counts should look like:
        {
            ""  : {"a": 1},          # at i=0, ctx="", ch="a"
            "a" : {"b": 2},          # at i=1,3, ctx="a", ch="b"
            "b" : {"a": 1}           # at i=2, ctx="b", ch="a"
        }

        Note: Context is always the last (n-1) characters before current position.
          For positions near the beginning, context may be shorter than (n-1).
        """
        text = "" if text is None else str(text)
        self.vocab = set(text)
        self.counts = {}
        n = self.n
        
        for i, ch in enumerate(text):
            start = max(0, i - (n - 1))
            ctx = text[start:i]     
            bucket = self.counts.setdefault(ctx, {})
            bucket[ch] = bucket.get(ch, 0) + 1
            
        self.trained = True
        return self
    
    def _probs_for_context(self, context: str):
        """
        Get probability distribution over next characters for a given context.
        
        Args:
            context (str): The context string
            
        Returns:
            dict[str, float]: Dictionary mapping characters to probabilities
        """
        #if not self.trained:
            #raise RuntimeError("Model not trained. Call fit(text) first.")
        ctx = context[-(self.n - 1):] if self.n > 1 else ""
        ctxCounts = self.counts.get(ctx, {})
        total = sum(ctxCounts.values())
        if total == 0:
            return {ch: 0.0 for ch in self.vocab}
        probs = {ch: ctxCounts.get(ch, 0) / total for ch in self.vocab}
        
        return probs
    
    def prob(self, s: str) -> float:
        """
        Calculate the probability of a string.
        
        Args:
            s (str): String to calculate probability for
            
        Returns:
            float: Probability of the string
        """
        return math.exp(self.logprob(s))
    
    def logprob(self, s: str) -> float:
        """
        Calculate the log-probability of a string.
        
        Args:
            s (str): String to calculate log-probability for
            
        Returns:
            float: Log-probability of the string
        """
        lp = 0.0
        for i in range(len(s)):
            # Get context for character at position i
            ctx = s[max(0, i - (self.n - 1)):i]
            
            # Get probability of character given context
            p_next = self._probs_for_context(ctx).get(s[i], 0.0)
            
            if p_next <= 0.0:
                return float("-inf")
            
            lp += math.log(p_next)
        
        return lp
    
    def next_char_distribution(self, context: str):
        """
        Get the probability distribution over next characters for given context.
        
        Args:
            context (str): Context string
            
        Returns:
            dict[str, float]: Dictionary mapping characters to probabilities,
                            sorted by probability in descending order
        """
        d = self._probs_for_context(context)
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    def generate(self, num_chars: int, seed: str = "") -> str:
        """
        Generate text using the trained model.

        Args:
            num_chars (int): Number of characters to generate
            seed (str): Initial string to start generation

        Returns:
            str: Generated text (seed + num_chars new characters)
        """
        out = list(seed)

        for _ in range(num_chars):
            # Get probability distribution for current context
            dist = self._probs_for_context("".join(out))

            if not dist:
                break

            # Extract characters and probabilities
            chars, probs = zip(*dist.items())

            # Cumulative sampling
            r = random.random()
            s = 0.0
            pick = chars[-1]  # fallback to last character

            for ch, p in zip(chars, probs):
                s += p
                if r <= s:
                    pick = ch
                    break

            out.append(pick)

        return "".join(out)

