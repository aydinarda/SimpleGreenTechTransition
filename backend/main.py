from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import uuid
import os
from typing import Dict, Optional
from database import database, init_db, db_insert_room, db_update_room_start, db_insert_player, db_insert_round

rooms: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await database.connect()
    await init_db()
    yield
    await database.disconnect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ─── Pydantic models ──────────────────────────────────────────────────────────

class CreateRoomRequest(BaseModel):
    g_min: float
    g_max: float
    num_rounds: int = 6
    alpha: float = 0.0          # underinvestor weight: 0=market share, 1=effort (log shortage)
    beta: float = 0.0           # overinvestor weight:  0=market share, 1=effort (log excess)
    pi_p: float = 4.0           # reward per green customer served (own allocation)
    pi_r: float = 3.0           # reward per green customer served (2nd stage residual)
    pi_q: float = 5.0           # reward per stolen green customer
    c_u: float = 10.0           # penalty per customer lost (underinvestors)
    c_o: float = 1.0            # penalty per unit of failed stealing (overinvestors)
    major_share: float = 0.8    # fraction of total market held by major players
    major_player_frac: float = 0.2  # fraction of players designated as major

class JoinRequest(BaseModel):
    player_name: str

class SubmitRequest(BaseModel):
    player_id: str
    investment: float

class AdminRequest(BaseModel):
    admin_token: str

class NextRoundRequest(BaseModel):
    admin_token: str
    # optional per-round param overrides
    g_min: Optional[float] = None
    g_max: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    pi_p: Optional[float] = None
    pi_r: Optional[float] = None
    pi_q: Optional[float] = None
    c_u: Optional[float] = None
    c_o: Optional[float] = None
    major_share: Optional[float] = None
    major_player_frac: Optional[float] = None


# ─── Core game logic ──────────────────────────────────────────────────────────

def _generate_market(n: int, major_share: float, major_player_frac: float) -> tuple:
    """Two-group Dirichlet: n_major players hold major_share of total market,
    n_minor hold (1 - major_share). Indices are shuffled so join order is neutral."""
    rng = np.random.default_rng()
    n_major = max(1, round(n * major_player_frac))
    n_minor = n - n_major
    major_part = rng.dirichlet(np.ones(n_major)) * major_share
    if n_minor > 0:
        minor_part = rng.dirichlet(np.ones(n_minor)) * (1 - major_share)
        shares   = np.concatenate([major_part, minor_part])
        is_major = np.array([True] * n_major + [False] * n_minor)
    else:
        shares   = major_part
        is_major = np.ones(n_major, dtype=bool)
    perm = rng.permutation(n)
    return shares[perm].tolist(), is_major[perm].tolist()


def _draw_g(params: dict) -> float:
    return float(np.random.uniform(params["g_min"], params["g_max"]))


def _capped_softmax_allocation(cap: np.ndarray, weight: np.ndarray, total: float, tol: float = 1e-9) -> np.ndarray:
    """Allocate `total` proportionally via softmax weights, respecting cap[i] per player.
    Guarantees alloc.sum() == min(total, cap.sum()) by iteratively redistributing
    the leftover from capped players to the remaining active players."""
    alloc     = np.zeros_like(cap, dtype=float)
    remaining = min(total, float(cap.sum()))
    active    = cap > tol

    while remaining > tol and active.any():
        w    = np.where(active, weight, 0.0)
        wsum = float(w.sum())
        proposal = (remaining * w / wsum) if wsum > tol else np.where(active, remaining / float(active.sum()), 0.0)

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


def compute_round(room: dict) -> tuple:
    params = room["params"]
    g     = room["g_realized"]
    alpha = params["alpha"]
    beta  = params["beta"]
    pi_p  = params["pi_p"]
    pi_r  = params["pi_r"]
    pi_q  = params["pi_q"]
    c_u   = params["c_u"]
    c_o   = params["c_o"]

    results = {}
    for pid, player in room["players"].items():
        s_i  = player["share"]
        a_i  = float(np.clip(room["submissions"].get(pid, 0.0), 0.0, s_i))
        gs_i = g * s_i
        pos  = a_i - gs_i
        results[pid] = {
            "name": player["name"],
            "s_i": s_i, "a_i": a_i, "gs_i": gs_i, "pos_i": pos,
            "lost": 0.0, "gained": 0.0, "overage": 0.0,
        }

    N_plus  = {pid: r for pid, r in results.items() if r["pos_i"] >  1e-9}
    N_minus = {pid: r for pid, r in results.items() if r["pos_i"] < -1e-9}

    if N_plus and N_minus:
        pids_plus    = list(N_plus.keys())
        pids_minus   = list(N_minus.keys())

        excess_arr   = np.array([results[pid]["pos_i"]   for pid in pids_plus],  dtype=float)
        shortage_arr = np.array([-results[pid]["pos_i"]  for pid in pids_minus], dtype=float)
        s_plus_arr   = np.array([results[pid]["s_i"]     for pid in pids_plus],  dtype=float)
        s_minus_arr  = np.array([results[pid]["s_i"]     for pid in pids_minus], dtype=float)

        T = min(float(excess_arr.sum()), float(shortage_arr.sum()))

        # exp((1-β)·s + β·log(1+excess)) = exp((1-β)·s) · (1+excess)^β
        weight_plus  = np.exp((1 - beta)  * s_plus_arr  + beta  * np.log1p(excess_arr))
        weight_minus = np.exp((1 - alpha) * s_minus_arr + alpha * np.log1p(shortage_arr))

        win_arr  = _capped_softmax_allocation(excess_arr,   weight_plus,  T)
        lost_arr = _capped_softmax_allocation(shortage_arr, weight_minus, T)

        for pid, gained in zip(pids_plus, win_arr):
            results[pid]["gained"]  = float(gained)
            results[pid]["overage"] = float(results[pid]["pos_i"] - gained)

        for pid, lost in zip(pids_minus, lost_arr):
            results[pid]["lost"] = float(lost)

    # New market shares: s̃_i = s_i + w_i - u_i
    for pid, r in results.items():
        r["s_new"] = r["s_i"] - r["lost"] + r["gained"]

    # A_g: total green customers served in stage 1
    A_g = (
        sum(min(r["gs_i"], r["a_i"]) for r in results.values())
        + sum(r["gained"] for r in results.values())
    )

    # Reward function (paper Section 2)
    for pid, r in results.items():
        own              = min(r["gs_i"], r["a_i"])
        second           = r["s_new"] * max(0.0, g - A_g)
        r["reward_own"]   = pi_p * own
        r["reward_2nd"]   = pi_r * second
        r["reward_steal"] = pi_q * r["gained"]
        r["penalty"]      = -c_u * r["lost"] - c_o * r["overage"]
        r["score"]        = r["reward_own"] + r["reward_2nd"] + r["reward_steal"] + r["penalty"]

    return results, A_g


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/rooms")
async def create_room(req: CreateRoomRequest):
    room_id     = str(uuid.uuid4())[:6].upper()
    admin_token = str(uuid.uuid4())
    params = {
        "g_min": req.g_min, "g_max": req.g_max, "num_rounds": req.num_rounds,
        "alpha": req.alpha, "beta": req.beta,
        "pi_p": req.pi_p, "pi_r": req.pi_r, "pi_q": req.pi_q,
        "c_u": req.c_u, "c_o": req.c_o,
        "major_share": req.major_share,
        "major_player_frac": req.major_player_frac,
    }
    rooms[room_id] = {
        "room_id": room_id,
        "admin_token": admin_token,
        "params": params,
        "g_realized": None,
        "players": {},
        "status": "lobby",
        "current_round": 0,
        "submissions": {},
        "history": [],
    }
    await db_insert_room(room_id, params)
    return {"room_id": room_id, "admin_token": admin_token}


@app.post("/api/rooms/{room_id}/join")
def join_room(room_id: str, req: JoinRequest):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "Room not found")
    if room["status"] != "lobby":
        raise HTTPException(400, "Game already started")
    player_id = str(uuid.uuid4())
    room["players"][player_id] = {
        "name": req.player_name, "share": 0.0, "total_score": 0.0,
    }
    return {"player_id": player_id, "player_name": req.player_name}


@app.post("/api/rooms/{room_id}/start")
async def start_game(room_id: str, req: AdminRequest):
    room = rooms.get(room_id)
    if not room: raise HTTPException(404, "Room not found")
    if req.admin_token != room["admin_token"]: raise HTTPException(403, "Not admin")
    if room["status"] != "lobby": raise HTTPException(400, "Already started")
    if len(room["players"]) < 2: raise HTTPException(400, "Need at least 2 players")

    n = len(room["players"])
    shares, is_major_list = _generate_market(
        n,
        room["params"]["major_share"],
        room["params"]["major_player_frac"],
    )
    initial_shares = {}
    for i, pid in enumerate(room["players"]):
        room["players"][pid]["share"]    = shares[i]
        room["players"][pid]["is_major"] = is_major_list[i]
        initial_shares[pid] = shares[i]
    room["initial_shares"] = initial_shares
    room["initial_is_major"] = {pid: is_major_list[i] for i, pid in enumerate(room["players"])}

    room["g_realized"] = _draw_g(room["params"])

    room["status"] = "playing"
    room["current_round"] = 1
    room["submissions"] = {}

    await db_update_room_start(room_id, room["g_realized"])
    for pid, share in initial_shares.items():
        await db_insert_player(pid, room_id, room["players"][pid]["name"], share)

    return {"status": "playing", "round": 1}


@app.post("/api/rooms/{room_id}/submit")
def submit_investment(room_id: str, req: SubmitRequest):
    room = rooms.get(room_id)
    if not room: raise HTTPException(404, "Room not found")
    if room["status"] != "playing": raise HTTPException(400, "Not in playing phase")
    if req.player_id not in room["players"]: raise HTTPException(404, "Player not found")

    s_i = room["players"][req.player_id]["share"]
    a_i = float(np.clip(req.investment, 0.0, s_i))
    room["submissions"][req.player_id] = a_i
    return {"submitted": a_i}


@app.post("/api/rooms/{room_id}/resolve")
async def resolve_round(room_id: str, req: AdminRequest):
    room = rooms.get(room_id)
    if not room: raise HTTPException(404, "Room not found")
    if req.admin_token != room["admin_token"]: raise HTTPException(403, "Not admin")
    if room["status"] != "playing": raise HTTPException(400, "Not in playing phase")

    for pid in room["players"]:
        room["submissions"].setdefault(pid, 0.0)

    results, A_g = compute_round(room)

    for pid, r in results.items():
        room["players"][pid]["share"] = r["s_new"]
        room["players"][pid]["total_score"] += r["score"]

    room["history"].append({
        "round": room["current_round"],
        "g_realized": room["g_realized"],
        "results": results,
        "A_g": A_g,
    })
    room["status"] = "round_results"

    await db_insert_round(room_id, room["current_round"], results, A_g)

    return {"status": "round_results", "g_realized": room["g_realized"]}


@app.post("/api/rooms/{room_id}/next_round")
def next_round(room_id: str, req: NextRoundRequest):
    room = rooms.get(room_id)
    if not room: raise HTTPException(404, "Room not found")
    if req.admin_token != room["admin_token"]: raise HTTPException(403, "Not admin")
    if room["status"] != "round_results": raise HTTPException(400, "Not in round_results phase")

    # Apply optional param overrides
    _UPDATABLE = ["g_min", "g_max", "alpha", "beta", "pi_p", "pi_r", "pi_q", "c_u", "c_o",
                  "major_share", "major_player_frac"]
    market_changed = False
    for key in _UPDATABLE:
        val = getattr(req, key)
        if val is not None:
            if key in ("major_share", "major_player_frac") and val != room["params"][key]:
                market_changed = True
            room["params"][key] = val

    room["current_round"] += 1
    room["submissions"] = {}

    if room["current_round"] > room["params"]["num_rounds"]:
        room["status"] = "finished"
    else:
        # Regenerate market if structure params changed
        if market_changed:
            n = len(room["players"])
            new_shares, new_is_major = _generate_market(
                n, room["params"]["major_share"], room["params"]["major_player_frac"]
            )
            pids = list(room["players"].keys())
            room["initial_shares"]   = {pid: new_shares[i]   for i, pid in enumerate(pids)}
            room["initial_is_major"] = {pid: new_is_major[i] for i, pid in enumerate(pids)}
            for pid in pids:
                room["players"][pid]["is_major"] = room["initial_is_major"][pid]

        # Reset shares to initial and re-draw g for this round
        for pid, share in room["initial_shares"].items():
            room["players"][pid]["share"] = share
        room["g_realized"] = _draw_g(room["params"])
        room["status"] = "playing"

    return {"status": room["status"], "round": room["current_round"], "g_realized": room["g_realized"]}


@app.get("/api/rooms/{room_id}/state")
def get_state(room_id: str, player_id: Optional[str] = None, admin_token: Optional[str] = None):
    room = rooms.get(room_id)
    if not room: raise HTTPException(404, "Room not found")
    is_admin = admin_token is not None and admin_token == room["admin_token"]

    players_info = {}
    for pid, p in room["players"].items():
        players_info[pid] = {
            "name": p["name"],
            "share": p["share"],
            "total_score": p["total_score"],
            "submitted": pid in room["submissions"],
            "is_me": pid == player_id,
            "is_major": p.get("is_major"),
        }

    return {
        "room_id": room_id,
        "status": room["status"],
        "current_round": room["current_round"],
        "num_rounds": room["params"]["num_rounds"],
        "g_min": room["params"]["g_min"],
        "g_max": room["params"]["g_max"],
        "g_realized": room["g_realized"] if is_admin else None,
        "params": room["params"],
        "players": players_info,
        "history": room["history"],
        "my_share": room["players"].get(player_id, {}).get("share") if player_id else None,
        "my_submitted": player_id in room["submissions"] if player_id else False,
    }


# ─── Serve frontend ───────────────────────────────────────────────────────────

frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
