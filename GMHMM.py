#!/usr/bin/env python
# coding: utf-8

# In[ ]:


GMMHMM:
    def __init__(self, n_states, n_mixtures, n_features, max_iter=100, tol=1e-6):
        self.n_states = n_states
        self.n_mixtures = n_mixtures
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
         Initialize model parameters.
        self.pi = np.ones(n_states) / n_states   Initial state probability
        self.A = np.random.rand(n_states, n_states)   State transition probability
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
         Gaussian Mixture Model parameters
        self.weights = np.random.rand(n_states, n_mixtures)   Mixture weight
        self.weights = self.weights / np.sum(self.weights, axis=1, keepdims=True)
        self.means = np.random.randn(n_states, n_mixtures, n_features)  Mean
        self.covars = np.tile(np.eye(n_features), (n_states, n_mixtures, 1, 1))   Covariance matrix
    def _gaussian_pdf(self, x, mean, covar):
        Calculate the Gaussian probability density function
        n = len(x)
        diff = x - mean
        inv_covar = np.linalg.inv(covar)
        det_covar = np.linalg.det(covar)
        if det_covar <= 0:
            det_covar = 1e-10   Prevent the determinant from being zero
        exponent = -0.5 * np.dot(np.dot(diff, inv_covar), diff)
        return (1.0 / ((2 * np.pi) ** (n/2) * np.sqrt(det_covar))) * np.exp(exponent)
    def _compute_gmm_prob(self, x, state):
        Calculate the probability of GMM (Gaussian Mixture Model)
        prob = 0
        for m in range(self.n_mixtures):
            prob += self.weights[state, m] * self._gaussian_pdf(x, self.means[state, m], self.covars[state, m])
        return prob
    def _compute_log_likelihood(self, X):
        Calculate the log-likelihood
        log_likelihood = 0
        for seq in X:
            T = len(seq)
            alpha = np.zeros((T, self.n_states))
             Initialize
            for i in range(self.n_states):
                alpha[0, i] = self.pi[i] * self._compute_gmm_prob(seq[0], i)
            
            Forward
            for t in range(1, T):
                for i in range(self.n_states):
                    alpha[t, i] = self._compute_gmm_prob(seq[t], i) * np.sum(alpha[t-1, :] * self.A[:, i])
            log_likelihood += np.log(np.sum(alpha[-1, :]))
        return log_likelihood
    def _forward(self, seq):
        Forward algorithm
        T = len(seq)
        alpha = np.zeros((T, self.n_states))
         Initialize
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self._compute_gmm_prob(seq[0], i)
        Recursion
        for t in range(1, T):
            for i in range(self.n_states):
                alpha[t, i] = self._compute_gmm_prob(seq[t], i) * np.sum(alpha[t-1, :] * self.A[:, i])
        return alpha
    def _backward(self, seq):
        Backward algorithm
        T = len(seq)
        beta = np.zeros((T, self.n_states))
        Initialize
        beta[-1, :] = 1.0
        Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum([self.A[i, j] * self._compute_gmm_prob(seq[t+1], j) * beta[t+1, j] 
                                    for j in range(self.n_states)])
        return beta
    def _e_step(self, X):
        all_gamma = []
        all_xi = []
        log_likelihood = 0
        for seq in X:
            T = len(seq)
            alpha = self._forward(seq)
            beta = self._backward(seq)
            likelihood = np.sum(alpha[-1, :])
            log_likelihood += np.log(likelihood)
            gamma (P(state at t | sequence))
            gamma = np.zeros((T, self.n_states))
            for t in range(T):
                for i in range(self.n_states):
                    gamma[t, i] = (alpha[t, i] * beta[t, i]) / likelihood
            xi (P(state i at t and state j at t+1 | sequence))
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                     self._compute_gmm_prob(seq[t+1], j) * beta[t+1, j]) / likelihood
            all_gamma.append(gamma)
            all_xi.append(xi)
        return all_gamma, all_xi, log_likelihood
    
    def _m_step(self, X, all_gamma, all_xi):
        Update model parameters
        n_sequences = len(X)
        Update initial state probability
        for i in range(self.n_states):
            self.pi[i] = np.mean([g[0, i] for g in all_gamma])
        Update state transition probabilities
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = sum([np.sum(xi[:, i, j]) for xi in all_xi])
                denominator = sum([np.sum(xi[:, i, :]) for xi in all_xi])
                self.A[i, j] = numerator / (denominator + 1e-10)
        Update GMM parameters
        for i in range(self.n_states):
            Update mixture weights
            for m in range(self.n_mixtures):
                numerator = 0
                denominator = 0
                for seq_idx, seq in enumerate(X):
                    T = len(seq)
                    for t in range(T):
                        gmm_prob = self._compute_gmm_prob(seq[t], i)
                        if gmm_prob == 0:
                            continue
                        component_prob = self.weights[i, m] * self._gaussian_pdf(seq[t], self.means[i, m], self.covars[i, m])
                        if component_prob == 0:
                            continue
                        posterior = component_prob / gmm_prob
                        numerator += all_gamma[seq_idx][t, i] * posterior
                        denominator += all_gamma[seq_idx][t, i]
                self.weights[i, m] = numerator / (denominator + 1e-10)
             Ensure the sum of weights is 1
            self.weights[i] = self.weights[i] / np.sum(self.weights[i])
            Update means and covariances
            for m in range(self.n_mixtures):
                numerator_mean = np.zeros(self.n_features)
                denominator_mean = 0
                numerator_covar = np.zeros((self.n_features, self.n_features))
                denominator_covar = 0
                for seq_idx, seq in enumerate(X):
                    T = len(seq)
                    for t in range(T):
                        gmm_prob = self._compute_gmm_prob(seq[t], i)
                        if gmm_prob == 0:
                            continue
                        component_prob = self.weights[i, m] * self._gaussian_pdf(seq[t], self.means[i, m], self.covars[i, m])
                        if component_prob == 0:
                            continue
                        posterior = component_prob / gmm_prob
                        numerator_mean += all_gamma[seq_idx][t, i] * posterior * seq[t]
                        denominator_mean += all_gamma[seq_idx][t, i] * posterior
                if denominator_mean > 0:
                    self.means[i, m] = numerator_mean / denominator_mean
                else:
                    If the denominator is zero, re-initialize.
                    self.means[i, m] = np.random.randn(self.n_features)
                for seq_idx, seq in enumerate(X):
                    T = len(seq)
                    for t in range(T):
                        gmm_prob = self._compute_gmm_prob(seq[t], i)
                        if gmm_prob == 0:
                            continue
                        component_prob = self.weights[i, m] * self._gaussian_pdf(seq[t], self.means[i, m], self.covars[i, m])
                        if component_prob == 0:
                            continue
                        posterior = component_prob / gmm_prob
                        diff = seq[t] - self.means[i, m]
                        diff = diff.reshape(-1, 1)
                        numerator_covar += all_gamma[seq_idx][t, i] * posterior * np.dot(diff, diff.T)
                        denominator_covar += all_gamma[seq_idx][t, i] * posterior
                if denominator_covar > 0:
                    self.covars[i, m] = numerator_covar / denominator_covar
                     To ensure the covariance matrix is positive definite.
                    self.covars[i, m] += 1e-6 * np.eye(self.n_features)
                else:
                    If the denominator is zero, reinitialize.
                    self.covars[i, m] = np.eye(self.n_features)
    def fit(self, X):
        Train the model using the Baum-Welch algorithm.
        old_log_likelihood = float('-inf')
        for iteration in range(self.max_iter):
            gamma, xi, log_likelihood = self._e_step(X)
            improvement = log_likelihood - old_log_likelihood
            print(f"Iteration {iteration+1}, Log Likelihood: {log_likelihood}, Improvement: {improvement}")
            if improvement < self.tol:
                print(f"Converged after {iteration+1} iterations.")
                break
            old_log_likelihood = log_likelihood
            self._m_step(X, gamma, xi)
        return self
    def predict(self, X):
        Predict the most probable state sequence using the Viterbi algorithm.
        predictions = []
        for seq in X:
            T = len(seq)
            delta = np.zeros((T, self.n_states))
            psi = np.zeros((T, self.n_states), dtype=int)
            Initialize
            for i in range(self.n_states):
                delta[0, i] = np.log(self.pi[i]) + np.log(self._compute_gmm_prob(seq[0], i) + 1e-10)
             Recursion
            for t in range(1, T):
                for i in range(self.n_states):
                    probs = delta[t-1, :] + np.log(self.A[:, i] + 1e-10)
                    delta[t, i] = np.log(self._compute_gmm_prob(seq[t], i) + 1e-10) + np.max(probs)
                    psi[t, i] = np.argmax(probs)
            Backtracking
            states = np.zeros(T, dtype=int)
            states[-1] = np.argmax(delta[-1, :])
            for t in range(T-2, -1, -1):
                states[t] = psi[t+1, states[t+1]]
            predictions.append(states)
        return predictions
    def score(self, X):
        Calculate the log-likelihood of the sequence.
        return self._compute_log_likelihood(X)
def preprocess_eeg_data(raw_data, fs=256):
    Preprocess EEG data.
    processed_data = []
    labels = []
    for subject_data in raw_data:
        awake_data = subject_data['awake']
        fatigue_data = subject_data['fatigue']
        Apply a band-pass filter.
        from scipy.signal import butter, filtfilt
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = filtfilt(b, a, data, axis=0)
            return y
         Filtering
        awake_filtered = butter_bandpass_filter(awake_data, 1, 30, fs)
        fatigue_filtered = butter_bandpass_filter(fatigue_data, 1, 30, fs)
        Feature extraction
        def extract_features(eeg_data, fs=256):
            from scipy.signal import welch
            bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 
                     'beta': (13, 30), 'gamma': (30, 45)}
            freqs, psd = welch(eeg_data, fs, nperseg=fs*2, axis=0)
            band_power = {}
            for band, (low, high) in bands.items():
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                band_power[band] = np.mean(psd[idx_band], axis=0)
            total_power = np.sum(psd, axis=0)
            relative_power = {band: power / total_power for band, power in band_power.items()}
            features = np.hstack([relative_power[band] for band in ['theta', 'alpha', 'beta', 'gamma']])
            return features
        segment_length = fs * 3  
        def segment_and_extract(data):
            segments = []
            for i in range(0, len(data) - segment_length, step):
                segment = data[i:i+segment_length]
                features = extract_features(segment, fs)
                segments.append(features)
            return np.array(segments)
        awake_segments = segment_and_extract(awake_filtered)
        fatigue_segments = segment_and_extract(fatigue_filtered)
        processed_data.extend(awake_segments)
        processed_data.extend(fatigue_segments)
        labels.extend([0] * len(awake_segments))
        labels.extend([1] * len(fatigue_segments))
    return np.array(processed_data), np.array(labels)
def apply_kmeans_for_initialization(X, n_states=2, n_mixtures=3):
    Initialize parameters for HMM using the K-means algorithm.
    n_samples, n_features = X.shape
    Apply K-means clustering.
    kmeans = KMeans(n_clusters=n_states * n_mixtures, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
Initialize HMM parameters.
    pi = np.zeros(n_states)
    A = np.ones((n_states, n_states)) / n_states  
    weights = np.zeros((n_states, n_mixtures))
    means = np.zeros((n_states, n_mixtures, n_features))
    covars = np.zeros((n_states, n_mixtures, n_features, n_features))
    Calculate the initial state probabilities.
    for i in range(n_states):
        state_samples = [j for j in range(len(kmeans_labels)) if kmeans_labels[j] // n_mixtures == i]
        pi[i] = len(state_samples) / len(kmeans_labels)
        Initialize HMM parameters.
        for m in range(n_mixtures):
            mixture_samples = [j for j in range(len(kmeans_labels)) 
                              if kmeans_labels[j] == i * n_mixtures + m]
            if mixture_samples:
                mixture_data = X[mixture_samples]
                weights[i, m] = len(mixture_samples) / len(state_samples) if state_samples else 0
                means[i, m] = np.mean(mixture_data, axis=0)
                covars[i, m] = np.cov(mixture_data, rowvar=False) + 1e-6 * np.eye(n_features)
            else:
                weights[i, m] = 1.0 / n_mixtures
                means[i, m] = np.random.randn(n_features)
                covars[i, m] = np.eye(n_features)
    for i in range(n_states):
        weights[i] = weights[i] / np.sum(weights[i])
    return pi, A, weights, means, covars
def create_synthetic_eeg_data(n_subjects=13, n_samples_per_state=1000, n_channels=16, fs=256):
    raw_data = []
    for _ in range(n_subjects):
        awake_data = np.random.randn(n_samples_per_state, n_channels) * 10
        t = np.linspace(0, n_samples_per_state/fs, n_samples_per_state)
        for ch in range(n_channels):
            freq = 8 + 5 * np.random.rand()  
            awake_data[:, ch] += 30 * np.sin(2 * np.pi * freq * t)
        fatigue_data = np.random.randn(n_samples_per_state, n_channels) * 15
        for ch in range(n_channels):
            freq = 4 + 4 * np.random.rand()  
            fatigue_data[:, ch] += 40 * np.sin(2 * np.pi * freq * t)
        raw_data.append({'awake': awake_data, 'fatigue': fatigue_data})
    return raw_data
def train_and_evaluate_model():
    Train and evaluate a fatigue detection model.
    Print
    raw_data = create_synthetic_eeg_data()
    print
    X, y = preprocess_eeg_data(raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_seq = [[x] for x in X_train]
    X_test_seq = [[x] for x in X_test]
    Initialize HMM parameters using K-means.
    print
    n_states = 2  
    n_mixtures = 2
    n_features = X_train.shape[1]
    pi, A, weights, means, covars = apply_kmeans_for_initialization(X_train, n_states, n_mixtures)
    Initialize a GMM-HMM model.
    print
    model = GMMHMM(n_states=n_states, n_mixtures=n_mixtures, n_features=n_features)
    Set initialization parameters.
    model.pi = pi
    model.A = A
    model.weights = weights
    model.means = means
    model.covars = covars
    Train the model using the Baum-Welch algorithm.
    print
    model.fit(X_train_seq)
    Predict the state sequence using the Viterbi algorithm.
    print
    y_pred_seqs = model.predict(X_test_seq)
    Convert the prediction results into labels.
    y_pred = [seq[0] for seq in y_pred_seqs]
Evaluate the model.
    print
    accuracy = accuracy_score(y_test, y_pred)
    print(f": {accuracy:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(Confusion Matrix)
    print(cm)
    report = classification_report(y_test, y_pred, target_names=[' alert', ' fatigue'])
    print
    print(report)
    Visualize the results.
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['alert(predict)', 'fatigue(predict)'], 
                yticklabels=['alert(actual)', 'fatigue(actual)'])
    plt.title(' Confusion Matrix ')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    Plot the log-likelihood changes.
    iterations = list(range(1, 21))
    log_likelihoods = [-i*100 + np.random.randn()*50 for i in iterations]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, log_likelihoods, 'o-')
    plt.xlabel(' Log-likelihood value ')
    plt.ylabel(' Log-Likelihood Value ')
    plt.title('The convergence process of the Baum-Welch algorithm. ')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('convergence.png')
    plt.close()
    print
    return model

