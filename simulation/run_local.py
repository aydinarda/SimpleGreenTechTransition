"""
Local batch simulation for RL policy optimization.
No HTTP – game logic runs in-process.

Outputs (per results/ directory):
  transitions.csv     — (state, action, reward, next_state, done) per player per round
  episodes.csv        — episode-level summary
  param_configs.csv   — parameter configurations sampled
  strategy_stats.csv  — per-strategy performance across all configs
  sim_summary.json    — aggregate statistics

State vector (17 dims):
  share, round_frac, g_min, g_max, g_mid, n_players_norm,
  alpha, beta, pi_p_norm, pi_r_norm, pi_q_norm, c_u_norm, c_o_norm,
  cum_score_norm, is_major, major_share, major_player_frac

Action:
  investment_frac ∈ [0, 1]  (actual investment = frac * share)

Usage:
  python simulation/run_local.py                    # default: medium scale
  python simulation/run_local.py --scale large
  python simulation/run_local.py --scale xlarge
  python simulation/run_local.py --configs 100 --episodes 20 --players 8
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Scale presets ──────────────────────────────────────────────────────────────
SCALES = {
    "small":  dict(n_configs=50,   n_episodes=20,  n_players=6,  n_rounds=6),
    "medium": dict(n_configs=200,  n_episodes=50,  n_players=10, n_rounds=6),
    "large":  dict(n_configs=500,  n_episodes=100, n_players=15, n_rounds=6),
    "xlarge": dict(n_configs=1000, n_episodes=200, n_players=20, n_rounds=6),
}

# ── Parameter ranges (uniform sampling) ───────────────────────────────────────
PARAM_RANGES = {
    "g_min":             (0.05, 0.45),
    "g_gap":             (0.30, 0.70),   # g_max = g_min + g_gap  →  ensures g_max < 1
    "alpha":             (0.0,  1.0),
    "beta":              (0.0,  1.0),
    "pi_p":              (2.0,  8.0),
    "pi_r":              (1.0,  5.0),
    "pi_q":              (3.0, 12.0),
    "c_u":               (4.0, 20.0),
    "c_o":               (0.5,  5.0),
    "major_share":       (0.5,  0.9),   # fraction of market held by major players
    "major_player_frac": (0.1,  0.4),   # fraction of players designated as major
}

# Normalization midpoints (for state encoding)
PI_MAX  = 12.0
C_U_MAX = 20.0
C_O_MAX = 5.0

# ── Investment strategies ──────────────────────────────────────────────────────
# Each strategy: fn(share, g_min, g_max, rng) → investment amount
STRATEGIES = {
    "zero":           lambda s, lo, hi, rng: 0.0,
    "tenth":          lambda s, lo, hi, rng: s * 0.10,
    "quarter":        lambda s, lo, hi, rng: s * 0.25,
    "half":           lambda s, lo, hi, rng: s * 0.50,
    "three_quarter":  lambda s, lo, hi, rng: s * 0.75,
    "full":           lambda s, lo, hi, rng: s * 1.00,
    "thresh_low":     lambda s, lo, hi, rng: np.clip(s * lo,             0, s),
    "thresh_mid":     lambda s, lo, hi, rng: np.clip(s * (lo + hi) / 2,  0, s),
    "thresh_high":    lambda s, lo, hi, rng: np.clip(s * hi,             0, s),
    "uniform_rand":   lambda s, lo, hi, rng: s * rng.random(),
    "beta_center":    lambda s, lo, hi, rng: s * rng.beta(2, 2),
    "beta_low":       lambda s, lo, hi, rng: s * rng.beta(1, 3),
    "beta_high":      lambda s, lo, hi, rng: s * rng.beta(3, 1),
}
STRATEGY_NAMES = list(STRATEGIES.keys())
N_STRATEGIES   = len(STRATEGY_NAMES)


# ── Game logic (inlined from backend – no FastAPI dependency) ──────────────────

def _generate_market(n: int, major_share: float, major_player_frac: float,
                     np_rng: np.random.Generator) -> tuple:
    """Two-group Dirichlet. Returns (shares, is_major_array)."""
    n_major = max(1, round(n * major_player_frac))
    n_minor = n - n_major
    major_part = np_rng.dirichlet(np.ones(n_major)) * major_share
    if n_minor > 0:
        minor_part = np_rng.dirichlet(np.ones(n_minor)) * (1 - major_share)
        shares   = np.concatenate([major_part, minor_part])
        is_major = np.array([True] * n_major + [False] * n_minor)
    else:
        shares   = major_part
        is_major = np.ones(n_major, dtype=bool)
    perm = np_rng.permutation(n)
    return shares[perm].tolist(), is_major[perm].tolist()


def _capped_softmax(cap: np.ndarray, weight: np.ndarray,
                    total: float, tol: float = 1e-9) -> np.ndarray:
    alloc     = np.zeros_like(cap, dtype=float)
    remaining = min(total, float(cap.sum()))
    active    = cap > tol
    while remaining > tol and active.any():
        w    = np.where(active, weight, 0.0)
        wsum = float(w.sum())
        if wsum > tol:
            proposal = remaining * w / wsum
        else:
            proposal = np.where(active, remaining / float(active.sum()), 0.0)
        room = cap - alloc
        hit  = active & (proposal >= room - tol)
        if hit.any():
            add        = np.where(hit, room, 0.0)
            alloc     += add
            remaining -= float(add.sum())
            active     = active & ((cap - alloc) > tol)
        else:
            alloc    += np.where(active, proposal, 0.0)
            remaining = 0.0
    return alloc


def _compute_round(players: dict, submissions: dict,
                   g: float, params: dict) -> dict:
    """Pure computation; returns per-player result dict."""
    alpha, beta  = params["alpha"],  params["beta"]
    pi_p, pi_r   = params["pi_p"],   params["pi_r"]
    pi_q         = params["pi_q"]
    c_u,  c_o    = params["c_u"],    params["c_o"]

    results = {}
    for pid, player in players.items():
        s_i  = player["share"]
        a_i  = float(np.clip(submissions.get(pid, 0.0), 0.0, s_i))
        gs_i = g * s_i
        results[pid] = {
            "s_i": s_i, "a_i": a_i, "gs_i": gs_i, "pos_i": a_i - gs_i,
            "lost": 0.0, "gained": 0.0, "overage": 0.0,
        }

    plus  = {p: r for p, r in results.items() if r["pos_i"] >  1e-9}
    minus = {p: r for p, r in results.items() if r["pos_i"] < -1e-9}

    if plus and minus:
        pp = list(plus.keys());  pm = list(minus.keys())
        ex  = np.array([results[p]["pos_i"]   for p in pp], dtype=float)
        sh  = np.array([-results[p]["pos_i"]  for p in pm], dtype=float)
        sp  = np.array([results[p]["s_i"]     for p in pp], dtype=float)
        sm  = np.array([results[p]["s_i"]     for p in pm], dtype=float)
        T   = min(float(ex.sum()), float(sh.sum()))
        wp  = np.exp((1 - beta)  * sp + beta  * np.log1p(ex))
        wm  = np.exp((1 - alpha) * sm + alpha * np.log1p(sh))
        win  = _capped_softmax(ex, wp, T)
        lost = _capped_softmax(sh, wm, T)
        for pid, w in zip(pp, win):
            results[pid]["gained"]  = float(w)
            results[pid]["overage"] = float(results[pid]["pos_i"] - w)
        for pid, l in zip(pm, lost):
            results[pid]["lost"] = float(l)

    for r in results.values():
        r["s_new"] = r["s_i"] - r["lost"] + r["gained"]

    A_g = (sum(min(r["gs_i"], r["a_i"]) for r in results.values())
           + sum(r["gained"]             for r in results.values()))

    for r in results.values():
        own              = min(r["gs_i"], r["a_i"])
        second           = r["s_new"] * max(0.0, g - A_g)
        r["reward_own"]  = pi_p * own
        r["reward_2nd"]  = pi_r * second
        r["reward_steal"]= pi_q * r["gained"]
        r["penalty"]     = -c_u * r["lost"] - c_o * r["overage"]
        r["score"]       = r["reward_own"] + r["reward_2nd"] + r["reward_steal"] + r["penalty"]
        r["A_g"]         = A_g
        r["g"]           = g
    return results


# ── Parameter sampling ─────────────────────────────────────────────────────────

def sample_params(rng: random.Random) -> dict:
    g_min = rng.uniform(*PARAM_RANGES["g_min"])
    g_gap = rng.uniform(*PARAM_RANGES["g_gap"])
    g_max = min(g_min + g_gap, 0.99)
    return {
        "g_min":             round(g_min, 4),
        "g_max":             round(g_max, 4),
        "alpha":             round(rng.uniform(*PARAM_RANGES["alpha"]), 4),
        "beta":              round(rng.uniform(*PARAM_RANGES["beta"]),  4),
        "pi_p":              round(rng.uniform(*PARAM_RANGES["pi_p"]),  4),
        "pi_r":              round(rng.uniform(*PARAM_RANGES["pi_r"]),  4),
        "pi_q":              round(rng.uniform(*PARAM_RANGES["pi_q"]),  4),
        "c_u":               round(rng.uniform(*PARAM_RANGES["c_u"]),   4),
        "c_o":               round(rng.uniform(*PARAM_RANGES["c_o"]),   4),
        "major_share":       round(rng.uniform(*PARAM_RANGES["major_share"]),       4),
        "major_player_frac": round(rng.uniform(*PARAM_RANGES["major_player_frac"]), 4),
    }


# ── State encoding (14-dim float vector) ──────────────────────────────────────

def encode_state(share: float, round_num: int, n_rounds: int,
                 params: dict, cum_score: float, n_players: int,
                 score_scale: float, is_major: bool = False) -> list:
    g_min = params["g_min"]
    g_max = params["g_max"]
    return [
        share,
        round_num / n_rounds,
        g_min,
        g_max,
        (g_min + g_max) / 2.0,
        n_players / 20.0,
        params["alpha"],
        params["beta"],
        params["pi_p"] / PI_MAX,
        params["pi_r"] / PI_MAX,
        params["pi_q"] / PI_MAX,
        params["c_u"]  / C_U_MAX,
        params["c_o"]  / C_O_MAX,
        cum_score / score_scale if score_scale > 0 else 0.0,
        float(is_major),
        params["major_share"],
        params["major_player_frac"],
    ]

STATE_COLS = [
    "s_share", "s_round_frac", "s_g_min", "s_g_max", "s_g_mid",
    "s_n_players_norm", "s_alpha", "s_beta",
    "s_pi_p_norm", "s_pi_r_norm", "s_pi_q_norm",
    "s_c_u_norm", "s_c_o_norm", "s_cum_score_norm",
    "s_is_major", "s_major_share", "s_major_player_frac",
]
NEXT_STATE_COLS = [c.replace("s_", "ns_", 1) for c in STATE_COLS]
N_STATE_DIMS = len(STATE_COLS)  # 17


# ── Single episode ─────────────────────────────────────────────────────────────

def run_episode(ep_id: int, config_id: int, params: dict,
                n_players: int, n_rounds: int,
                strategy_assignment: list, rng: random.Random,
                np_rng: np.random.Generator) -> tuple:
    """
    Returns:
        transitions : list of dicts  (one per player per round)
        ep_summary  : dict
    """
    g_min, g_max = params["g_min"], params["g_max"]

    # Initial market shares — two-group Dirichlet (major/minor)
    initial_shares, is_major_list = _generate_market(
        n_players, params["major_share"], params["major_player_frac"], np_rng
    )
    pids = [f"p{i}" for i in range(n_players)]

    players = {pid: {"share": initial_shares[i]} for i, pid in enumerate(pids)}
    cum_scores = {pid: 0.0 for pid in pids}

    # Rough score scale for normalization (max possible score per round)
    score_scale = (params["pi_q"] * 1.0 + params["pi_p"] * 1.0) * n_rounds

    transitions = []

    for rnd in range(1, n_rounds + 1):
        g_realized = np_rng.uniform(g_min, g_max)

        # Each player decides investment
        submissions = {}
        for i, pid in enumerate(pids):
            strat_name = strategy_assignment[i]
            strat_fn   = STRATEGIES[strat_name]
            share      = initial_shares[i]          # always initial share
            players[pid]["share"] = share            # reset share each round
            inv = float(np.clip(strat_fn(share, g_min, g_max, rng), 0.0, share))
            submissions[pid] = inv

        # Compute round
        results = _compute_round(players, submissions, g_realized, params)

        # Collect transitions
        done = (rnd == n_rounds)
        for i, pid in enumerate(pids):
            r        = results[pid]
            share    = initial_shares[i]
            im       = is_major_list[i]
            strat    = strategy_assignment[i]
            cum      = cum_scores[pid]

            state       = encode_state(share, rnd, n_rounds, params, cum, n_players, score_scale, im)
            action_frac = submissions[pid] / share if share > 1e-9 else 0.0
            reward      = r["score"]
            next_cum    = cum + reward
            next_round  = rnd + 1 if not done else rnd
            next_state  = encode_state(r["s_new"], next_round, n_rounds,
                                       params, next_cum, n_players, score_scale, im)

            row = {
                "episode_id":    ep_id,
                "config_id":     config_id,
                "round":         rnd,
                "player_idx":    i,
                "strategy":      strat,
                # raw state fields
                **{STATE_COLS[j]:      state[j]      for j in range(N_STATE_DIMS)},
                # action
                "action_frac":   round(action_frac,  6),
                "action_abs":    round(submissions[pid], 6),
                # reward
                "reward":        round(reward,        4),
                # next state
                **{NEXT_STATE_COLS[j]: next_state[j] for j in range(N_STATE_DIMS)},
                "done":          int(done),
                # diagnostics
                "g_realized":    round(g_realized,  4),
                "pos_i":         round(r["pos_i"],  4),
                "lost":          round(r["lost"],   4),
                "gained":        round(r["gained"], 4),
                "score_own":     round(r["reward_own"],   4),
                "score_2nd":     round(r["reward_2nd"],   4),
                "score_steal":   round(r["reward_steal"], 4),
                "penalty":       round(r["penalty"],      4),
            }
            transitions.append(row)
            cum_scores[pid] = next_cum

    # Episode summary
    final_scores  = {pid: cum_scores[pid] for pid in pids}
    winner_idx    = max(range(n_players), key=lambda i: final_scores[pids[i]])
    avg_score     = sum(final_scores.values()) / n_players

    ep_summary = {
        "episode_id":      ep_id,
        "config_id":       config_id,
        "n_players":       n_players,
        "n_rounds":        n_rounds,
        "winner_strategy": strategy_assignment[winner_idx],
        "winner_score":    round(final_scores[pids[winner_idx]], 3),
        "avg_score":       round(avg_score, 3),
        "min_score":       round(min(final_scores.values()), 3),
        "max_score":       round(max(final_scores.values()), 3),
        **{f"cfg_{k}": v for k, v in params.items()},
    }
    return transitions, ep_summary


# ── Worker (called by multiprocessing pool) ────────────────────────────────────

def _worker(args):
    config_id, params, n_episodes, n_players, n_rounds, base_seed = args

    rng    = random.Random(base_seed + config_id * 997)
    np_rng = np.random.default_rng(base_seed + config_id * 997 + 1)

    all_transitions = []
    all_summaries   = []

    for ep_idx in range(n_episodes):
        ep_id = config_id * n_episodes + ep_idx
        # Assign strategies round-robin + shuffle
        strats = [STRATEGY_NAMES[i % N_STRATEGIES] for i in range(n_players)]
        rng.shuffle(strats)

        trans, summary = run_episode(
            ep_id, config_id, params, n_players, n_rounds,
            strats, rng, np_rng,
        )
        all_transitions.extend(trans)
        all_summaries.append(summary)

    return all_transitions, all_summaries


# ── CSV helpers ────────────────────────────────────────────────────────────────

TRANSITION_FIELDS = (
    ["episode_id", "config_id", "round", "player_idx", "strategy"]
    + STATE_COLS
    + ["action_frac", "action_abs", "reward"]
    + NEXT_STATE_COLS
    + ["done", "g_realized", "pos_i", "lost", "gained",
       "score_own", "score_2nd", "score_steal", "penalty"]
)

EPISODE_FIELDS = (
    ["episode_id", "config_id", "n_players", "n_rounds",
     "winner_strategy", "winner_score", "avg_score", "min_score", "max_score"]
    + [f"cfg_{k}" for k in ["g_min","g_max","alpha","beta",
                             "pi_p","pi_r","pi_q","c_u","c_o"]]
)


def write_csv(path: str, rows: list, fields: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local RL simulation")
    parser.add_argument("--scale",    choices=SCALES.keys(), default="medium")
    parser.add_argument("--configs",  type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--players",  type=int, default=None)
    parser.add_argument("--rounds",   type=int, default=None)
    parser.add_argument("--workers",  type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    preset     = SCALES[args.scale]
    n_configs  = args.configs  or preset["n_configs"]
    n_episodes = args.episodes or preset["n_episodes"]
    n_players  = args.players  or preset["n_players"]
    n_rounds   = args.rounds   or preset["n_rounds"]
    seed       = args.seed
    n_workers  = args.workers

    total_episodes    = n_configs * n_episodes
    total_transitions = total_episodes * n_players * n_rounds

    print("=" * 64)
    print("LOCAL RL SIMULATION")
    print("=" * 64)
    print(f"  Scale           : {args.scale}")
    print(f"  Param configs   : {n_configs}")
    print(f"  Episodes/config : {n_episodes}")
    print(f"  Players/episode : {n_players}")
    print(f"  Rounds/episode  : {n_rounds}")
    print(f"  Workers         : {n_workers}")
    print(f"  Seed            : {seed}")
    print(f"  Total episodes  : {total_episodes:,}")
    print(f"  Total transitions: {total_transitions:,}")
    print(f"  Output dir      : {OUT_DIR}/")
    print("=" * 64)

    # Sample parameter configurations
    cfg_rng = random.Random(seed)
    configs = [sample_params(cfg_rng) for _ in range(n_configs)]

    # Save param configs
    cfg_path = os.path.join(OUT_DIR, "param_configs.csv")
    with open(cfg_path, "w", newline="") as f:
        fields = ["config_id"] + list(configs[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, cfg in enumerate(configs):
            w.writerow({"config_id": i, **cfg})
    print(f"  Saved {cfg_path}")

    # Build worker args
    worker_args = [
        (cfg_id, configs[cfg_id], n_episodes, n_players, n_rounds, seed)
        for cfg_id in range(n_configs)
    ]

    # Run simulation
    t0 = time.perf_counter()
    all_transitions = []
    all_summaries   = []
    done_configs    = 0

    print(f"\n  Running... (progress every 10 configs)")

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for trans, summ in pool.imap_unordered(_worker, worker_args, chunksize=5):
                all_transitions.extend(trans)
                all_summaries.extend(summ)
                done_configs += 1
                if done_configs % 10 == 0 or done_configs == n_configs:
                    elapsed = time.perf_counter() - t0
                    pct     = done_configs / n_configs * 100
                    rate    = done_configs / elapsed
                    eta     = (n_configs - done_configs) / rate if rate > 0 else 0
                    print(f"  [{pct:5.1f}%] {done_configs}/{n_configs} configs | "
                          f"{len(all_transitions):,} transitions | "
                          f"ETA {eta:.0f}s", flush=True)
    else:
        for wa in worker_args:
            trans, summ = _worker(wa)
            all_transitions.extend(trans)
            all_summaries.extend(summ)
            done_configs += 1
            if done_configs % 10 == 0 or done_configs == n_configs:
                elapsed = time.perf_counter() - t0
                pct     = done_configs / n_configs * 100
                print(f"  [{pct:5.1f}%] {done_configs}/{n_configs} configs | "
                      f"{len(all_transitions):,} transitions | "
                      f"{elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n  Simulation done in {elapsed:.1f}s")
    print(f"  Writing {len(all_transitions):,} transitions to CSV...", flush=True)

    # Write transitions
    trans_path = os.path.join(OUT_DIR, "transitions.csv")
    write_csv(trans_path, all_transitions, TRANSITION_FIELDS)
    trans_mb = os.path.getsize(trans_path) / 1e6
    print(f"  Saved {trans_path}  ({trans_mb:.1f} MB)")

    # Write episodes
    ep_path = os.path.join(OUT_DIR, "episodes.csv")
    write_csv(ep_path, all_summaries, EPISODE_FIELDS)
    print(f"  Saved {ep_path}")

    # ── Strategy analysis ──────────────────────────────────────────────────────
    from collections import defaultdict
    strat_wins   = defaultdict(int)
    strat_scores = defaultdict(list)
    strat_counts = defaultdict(int)

    for row in all_transitions:
        strat = row["strategy"]
        strat_scores[strat].append(row["reward"])
        strat_counts[strat] += 1

    for ep in all_summaries:
        strat_wins[ep["winner_strategy"]] += 1

    strat_rows = []
    for name in STRATEGY_NAMES:
        sc = strat_scores.get(name, [0])
        strat_rows.append({
            "strategy":       name,
            "win_count":      strat_wins.get(name, 0),
            "win_rate":       round(strat_wins.get(name, 0) / total_episodes, 4),
            "n_transitions":  strat_counts.get(name, 0),
            "avg_reward":     round(sum(sc) / len(sc), 4),
            "min_reward":     round(min(sc), 4),
            "max_reward":     round(max(sc), 4),
        })
    strat_rows.sort(key=lambda r: r["win_rate"], reverse=True)

    strat_path = os.path.join(OUT_DIR, "strategy_stats.csv")
    write_csv(strat_path, strat_rows,
              ["strategy","win_count","win_rate","n_transitions",
               "avg_reward","min_reward","max_reward"])
    print(f"  Saved {strat_path}")

    # ── Summary JSON ───────────────────────────────────────────────────────────
    all_rewards = [r["reward"] for r in all_transitions]
    summary = {
        "run_utc":           datetime.utcnow().isoformat(),
        "scale":             args.scale,
        "n_configs":         n_configs,
        "n_episodes":        n_episodes,
        "n_players":         n_players,
        "n_rounds":          n_rounds,
        "seed":              seed,
        "n_workers":         n_workers,
        "elapsed_s":         round(elapsed, 2),
        "total_transitions": len(all_transitions),
        "total_episodes":    len(all_summaries),
        "reward_mean":       round(float(np.mean(all_rewards)), 4),
        "reward_std":        round(float(np.std(all_rewards)),  4),
        "reward_min":        round(float(np.min(all_rewards)),  4),
        "reward_max":        round(float(np.max(all_rewards)),  4),
        "state_dims":        14,
        "action_space":      "[0, 1]  (investment fraction)",
        "strategy_ranking":  [r["strategy"] for r in strat_rows],
        "best_strategy":     strat_rows[0]["strategy"],
        "files": {
            "transitions":  "transitions.csv",
            "episodes":     "episodes.csv",
            "param_configs":"param_configs.csv",
            "strategies":   "strategy_stats.csv",
            "summary":      "sim_summary.json",
        },
        "state_vector_schema": {
            col: i for i, col in enumerate(STATE_COLS)
        },
    }

    summary_path = os.path.join(OUT_DIR, "sim_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}")

    # ── Final report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  Transitions    : {len(all_transitions):>10,}  ->  {trans_path}")
    print(f"  Episodes       : {len(all_summaries):>10,}  ->  {ep_path}")
    print(f"  File size      : {trans_mb:>10.1f} MB")
    print(f"  Elapsed        : {elapsed:>10.1f}s")
    print(f"  Throughput     : {len(all_transitions)/elapsed:>10,.0f} transitions/s")
    print()
    print("  Reward stats across all transitions:")
    print(f"    mean = {summary['reward_mean']:>8.3f}")
    print(f"    std  = {summary['reward_std']:>8.3f}")
    print(f"    min  = {summary['reward_min']:>8.3f}")
    print(f"    max  = {summary['reward_max']:>8.3f}")
    print()
    print("  Strategy ranking by win rate:")
    for r in strat_rows:
        bar = "#" * int(r["win_rate"] * 40)
        print(f"    {r['strategy']:<16} {r['win_rate']:.3f}  {bar}")
    print()
    print("  State vector (14 dims):")
    for i, col in enumerate(STATE_COLS):
        print(f"    [{i:2d}] {col}")
    print()
    print("  RL Usage:")
    print("    import pandas as pd")
    print("    df = pd.read_csv('simulation/results/transitions.csv')")
    print("    states  = df[STATE_COLS].values         # (N, 14)")
    print("    actions = df['action_frac'].values      # (N,)")
    print("    rewards = df['reward'].values           # (N,)")
    print("    next_s  = df[NEXT_STATE_COLS].values    # (N, 14)")
    print("    dones   = df['done'].values             # (N,)")
    print("=" * 64)


if __name__ == "__main__":
    main()
