import numpy as np
import matplotlib.pyplot as plt
from _find_type3 import check_tsfm_hallucination, context, future, N_CTX, HORIZON, h_t

trend_term = 0.02 * (N_CTX + h_t)
base_sine = np.sin(2 * np.pi * h_t / 40.0)

# Candidate 1: Harmonic ripple
cand_harmonic = base_sine + 0.65 * np.sin(2 * np.pi * h_t / 4.0) + trend_term
r_harm = check_tsfm_hallucination(context, cand_harmonic)
print("Harmonic:", r_harm['trend_ok'], r_harm['freq_ok'], r_harm['pat_ok'], r_harm['ar_ok'])

# Candidate 2: Jitter / noisy oscillation
rng = np.random.default_rng(2)
cand_noise = base_sine + rng.normal(0, 0.4, size=HORIZON) + trend_term
r_noise = check_tsfm_hallucination(context, cand_noise)
print("Noise:", r_noise['trend_ok'], r_noise['freq_ok'], r_noise['pat_ok'], r_noise['ar_ok'])

# Candidate 3: Combining slight harmonic ripple and noise
rng2 = np.random.default_rng(7)
cand_combo = base_sine + 0.45 * np.sin(2 * np.pi * h_t / 5.0) + rng2.normal(0, 0.25, size=HORIZON) + trend_term
r_combo = check_tsfm_hallucination(context, cand_combo)
print("Combo:", r_combo['trend_ok'], r_combo['freq_ok'], r_combo['pat_ok'], r_combo['ar_ok'], f"Freq gap={r_combo['freq_gap']:.3f}, Pat gap={r_combo['pat_gap']:.3f}, AR gap={r_combo['ar_gap']:.3f}")
