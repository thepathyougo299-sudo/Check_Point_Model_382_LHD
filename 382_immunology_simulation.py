# ================================================================
# Immune Checkpoint Blockade ODE Model  –  Zheng & Kim (2021)
# + Extension 1: Monte Carlo population simulation (n=500)
# + Extension 2: Time-dependent dosing schedules
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import Counter
import matplotlib 
matplotlib.use('Agg') 
import os
# Save figures next to wherever this script lives
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Baseline parameters (Table 2, Zheng & Kim 2021) ─────────
BASE = dict(
    rC=1,       rmax=0.09,  Cstar=1e3,  kappa=1.2,
    rA=0.5,     dA=0.8,     rI=0.4,     dI=3.0,
    rE=1.0,     Estar=5.0,  rS=1.0,     Sstar=5.0,
    beta=0.009, gamma=37.414
)

# ── ODE system  (Equations 1–5 from paper) ───────────────────
def model(t, y, p):
    C, A, I, E, S = [max(v, 0) for v in y]
    fC = min(p['rC'] * (1 - C / p['Cstar']), p['rmax'])   # logistic, capped at rmax
    dCdt = fC*C            - p['kappa']*C*E
    dAdt = p['rA']*C       - p['dA']*A
    dIdt = p['rI']*C*E     - p['dI']*I
    dEdt = -p['rE']*(E - p['Estar']) + p['beta']*A*I*E*S  - p['gamma']*E*S
    dSdt = -p['rS']*(S - p['Sstar']) - p['beta']*A*I*E*S  + p['gamma']*E*S
    return [dCdt, dAdt, dIdt, dEdt, dSdt]

def run(params, t_end=365, n_pts=800):
    """Solve the ODE system.  LSODA handles the stiffness well."""
    y0  = [params['Cstar'], 0, 0, 0, params['Sstar']]      # Table 3 ICs
    sol = solve_ivp(model, [0, t_end], y0, args=(params,),
                    t_eval=np.linspace(0, t_end, n_pts),
                    method='LSODA', rtol=1e-6, atol=1e-8)
    return sol.t, sol.y

def classify(t, C, Cstar):
    """Label a cancer trajectory as one of four response types."""
    Cn   = C / Cstar
    c30  = Cn[np.searchsorted(t, 30)]
    c90  = Cn[np.searchsorted(t, 90)]
    c180 = Cn[min(np.searchsorted(t, 180), len(Cn)-1)]
    if   c30 > 0.85 and c90 > 0.50:   return 'No Response'
    elif c30 < 0.15:                   return 'Quick Full'
    elif c30 > 0.85 and c180 < 0.50:  return 'Delayed'
    else:                              return 'Quick Partial'

# ================================================================
# FIGURE 1 – Four response types  (reproduces Fig 4 of paper)
# ================================================================
print("Generating Fig 1 – Four Response Types...")
cases = {
    'No Response':            dict(beta=0.0089988, gamma=37.4168, Estar=5.0, rmax=0.09),
    'Quick Full Response':    dict(beta=0.009,     gamma=37.4168, Estar=5.5, rmax=0.09),
    'Quick Partial Response': dict(beta=0.0089988, gamma=37.414,  Estar=5.0, rmax=1.0),
    'Delayed Response':       dict(beta=0.009,     gamma=37.414,  Estar=5.0, rmax=0.09),
}
colors1 = ['#d62728', '#2ca02c', '#9467bd', '#1f77b4']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, ov), col in zip(axes.flat, cases.items(), colors1):
    t, y = run({**BASE, **ov})
    ax.plot(t, y[0] / BASE['Cstar'], color=col, lw=2.5)
    ax.axhline(1, ls='--', color='gray', alpha=0.4)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (days)');  ax.set_ylabel('Cancer burden (C / C*)')
    ax.set_ylim(0, 1.2)
plt.suptitle('Fig 1 – Four Response Types  (Zheng & Kim 2021)', fontsize=13)
plt.tight_layout()
plt.savefig('fig1_response_types.png', dpi=150);  plt.show()
print("  → fig1_response_types.png saved.\n")

# ================================================================
# FIGURE 2 – Delayed responses with individual vs combo inhibitors
#             (reproduces Fig 5 of paper)
# ================================================================
print("Generating Fig 2 – Delayed Responses...")
delayed_cases = {
    'No Treatment':              dict(beta=0.0089988, gamma=37.4168),
    'Anti-CTLA-4 only (~5 mo)': dict(beta=0.009,     gamma=37.4168),
    'Anti-PD-1 only (~4 mo)':   dict(beta=0.0089988, gamma=37.414),
    'Combination (~2 mo)':      dict(beta=0.009,     gamma=37.414),
}
colors2 = ['gray', '#ff7f0e', '#9467bd', '#d62728']
ls2     = ['-', '--', '-.', ':']

fig, ax = plt.subplots(figsize=(10, 5))
for (name, ov), col, ls in zip(delayed_cases.items(), colors2, ls2):
    t, y = run({**BASE, **ov})
    ax.plot(t, y[0] / BASE['Cstar'], color=col, ls=ls, lw=2.5, label=name)
ax.set_title('Fig 2 – Effect of Inhibitor Type on Delay Length', fontweight='bold')
ax.set_xlabel('Time (days)');  ax.set_ylabel('Cancer burden (C / C*)')
ax.set_ylim(0, 1.15);  ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig2_delayed_responses.png', dpi=150);  plt.show()
print("  → fig2_delayed_responses.png saved.\n")

# ================================================================
# EXTENSION 1 – Monte Carlo: 500 virtual patients
# ================================================================
# We sample beta and gamma uniformly across the clinically
# relevant range surrounding the response-type thresholds.
# This lets us estimate how rare each outcome is in a population.
# ================================================================
print("Running Extension 1 – Monte Carlo (n=500)...")
np.random.seed(42)
N = 500

# Uniform sampling across the beta-gamma threshold region
beta_s  = np.random.uniform(0.00895, 0.00910, N)   # straddles beta-hat
gamma_s = np.random.uniform(37.408,  37.422,  N)   # straddles gamma-hat
Estar_s = np.random.normal(5.0, 0.3, N).clip(min=0) # patient immune variability

outcomes, delay_lengths = [], []
traces = {k: [] for k in ['No Response', 'Quick Full', 'Quick Partial', 'Delayed']}

for i in range(N):
    pt  = {**BASE, 'beta': beta_s[i], 'gamma': gamma_s[i], 'Estar': Estar_s[i]}
    t, y = run(pt, n_pts=400)
    label = classify(t, y[0], BASE['Cstar'])
    outcomes.append(label)
    if len(traces[label]) < 25:
        traces[label].append((t, y[0] / BASE['Cstar']))
    if label == 'Delayed':
        idx = np.where(y[0] / BASE['Cstar'] < 0.5)[0]
        if len(idx): delay_lengths.append(t[idx[0]])

counts = Counter(outcomes)
print(f"  Outcome counts: {dict(counts)}")
print(f"  Delayed responses: {counts['Delayed']}/{N}  = {100*counts['Delayed']/N:.1f}% of patients")
if delay_lengths:
    print(f"  Delay length: mean={np.mean(delay_lengths):.0f} d "
          f"({np.mean(delay_lengths)/30:.1f} mo), "
          f"range={np.min(delay_lengths):.0f}–{np.max(delay_lengths):.0f} d")

# ── Figure 3: three-panel Monte Carlo summary ────────────────
labels_mc = ['No Response', 'Delayed', 'Quick Partial', 'Quick Full']
colors_mc  = ['#d62728',    '#ff7f0e', '#9467bd',       '#2ca02c']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A – bar chart of outcome frequencies
vals = [counts[l] for l in labels_mc]
bars = axes[0].bar(labels_mc, vals, color=colors_mc, edgecolor='k', alpha=0.85)
axes[0].set_title('Outcome Distribution', fontweight='bold')
axes[0].set_ylabel('Number of Patients (out of 500)')
axes[0].set_xlabel('Response Type')
axes[0].tick_params(axis='x', rotation=12)
for bar, v in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 3,
                 f'{v}\n({100*v/N:.0f}%)', ha='center', fontsize=9)

# Panel B – spaghetti trajectories coloured by outcome
for label, color in zip(labels_mc, colors_mc):
    for t_tr, C_tr in traces[label]:
        axes[1].plot(t_tr, C_tr, color=color, alpha=0.22, lw=0.9)
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([0],[0], color=c, lw=2, label=l)
                         for l, c in zip(labels_mc, colors_mc)], fontsize=8)
axes[1].set_title('Cancer Trajectories by Outcome', fontweight='bold')
axes[1].set_xlabel('Time (days)');  axes[1].set_ylabel('Cancer burden (C / C*)')

# Panel C – parameter space scatter: where do outcomes fall in (beta, gamma)?
outcome_colors = {'No Response': '#d62728', 'Quick Full': '#2ca02c',
                  'Delayed': '#ff7f0e',     'Quick Partial': '#9467bd'}
for label in labels_mc:
    mask = [o == label for o in outcomes]
    axes[2].scatter(np.array(beta_s)[mask], np.array(gamma_s)[mask],
                    c=outcome_colors[label], s=18, alpha=0.6, label=label)
axes[2].set_title('Outcome in Parameter Space (β vs γ)', fontweight='bold')
axes[2].set_xlabel('β  (CTLA-4 activation)');  axes[2].set_ylabel('γ  (PD-1 suppression)')
axes[2].legend(fontsize=8, markerscale=1.5)

plt.suptitle(f'Extension 1 – Monte Carlo Population Simulation  (n={N})', fontsize=13)
plt.tight_layout()
plt.savefig('fig3_monte_carlo.png', dpi=150);  plt.show()
print("  → fig3_monte_carlo.png saved.\n")

# ================================================================
# EXTENSION 2 – Time-Dependent Dosing Schedules
# ================================================================
# A patient starts in the no-response region.  Each drug dose
# shifts beta up (anti-CTLA-4) and gamma down (anti-PD-1),
# moving them through parameter space.  We compare how dosing
# frequency and total number of doses changes the outcome.
# ================================================================
print("Generating Extension 2 – Dosing Schedules...")

NO_RESP = {**BASE, 'beta': 0.0089988, 'gamma': 37.4168}
DBETA   = 0.00000040   # beta increase per dose  (anti-CTLA-4 effect)
DGAMMA  = 0.00095      # gamma decrease per dose  (anti-PD-1 effect)

def model_dosing(t, y, base_p, dose_times):
    """ODE with beta/gamma updated after each administered dose."""
    n = sum(1 for td in dose_times if t >= td)
    p = {**base_p,
         'beta':  base_p['beta']  + n * DBETA,
         'gamma': max(base_p['gamma'] - n * DGAMMA, 0)}
    return model(t, y, p)

scenarios = {
    'No Treatment':             [],
    '4 doses – every 21 days':  [21, 42, 63, 84],
    '4 doses – every 42 days':  [42, 84, 126, 168],
    '8 doses – every 21 days':  [21, 42, 63, 84, 105, 126, 147, 168],
}
ls_list    = ['-', '--', '-.', ':']
color_list = ['gray', '#1f77b4', '#ff7f0e', '#d62728']

y0     = [NO_RESP['Cstar'], 0, 0, 0, NO_RESP['Sstar']]
t_eval = np.linspace(0, 400, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for (label, dose_times), ls, col in zip(scenarios.items(), ls_list, color_list):
    if not dose_times:
        sol = solve_ivp(model, [0, 400], y0, args=(NO_RESP,),
                        t_eval=t_eval, method='LSODA', rtol=1e-6, atol=1e-8)
    else:
        sol = solve_ivp(model_dosing, [0, 400], y0,
                        args=(NO_RESP, dose_times),
                        t_eval=t_eval, method='LSODA', rtol=1e-6, atol=1e-8)
    ax1.plot(sol.t, sol.y[0] / BASE['Cstar'], ls=ls, color=col, lw=2.5, label=label)
    ax2.plot(sol.t, sol.y[3],                 ls=ls, color=col, lw=2.5, label=label)

# Vertical lines marking the 8-dose schedule
for td in [21, 42, 63, 84, 105, 126, 147, 168]:
    ax1.axvline(td, color='gray', alpha=0.13, lw=1)
    ax2.axvline(td, color='gray', alpha=0.13, lw=1)

ax1.set_title('Cancer Burden vs Dosing Schedule', fontweight='bold')
ax1.set_xlabel('Time (days)');  ax1.set_ylabel('Cancer burden (C / C*)')
ax1.set_ylim(0, 1.15);  ax1.legend(fontsize=9)

ax2.set_title('Effector T Cell Response (E)', fontweight='bold')
ax2.set_xlabel('Time (days)');  ax2.set_ylabel('E cells (cells/nL)')
ax2.legend(fontsize=9)

plt.suptitle('Extension 2 – Effect of Dosing Schedule on Tumour Response', fontsize=13)
plt.tight_layout()
plt.savefig('fig4_dosing.png', dpi=150);  plt.show()
print("  → fig4_dosing.png saved.\n")

print("=" * 50)
print("ALL DONE.  Four figures generated:")
print("  fig1_response_types.png")
print("  fig2_delayed_responses.png")
print("  fig3_monte_carlo.png")
print("  fig4_dosing.png")

figures_to_merge = [
    'fig1_response_types.png',
    'fig2_delayed_responses.png',
    'fig3_monte_carlo.png',
    'fig4_dosing.png',
]

with PdfPages('simulation_results.pdf') as pdf:
    for fname in figures_to_merge:
        fig = plt.figure(figsize=(12, 8))
        img = plt.imread(fname)
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print("  → simulation_results.pdf saved.")