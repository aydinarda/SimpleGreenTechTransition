import os
import databases

_RAW_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./game_data.db")

# Normalize Supabase / Heroku postgres:// → postgresql+asyncpg://
if _RAW_URL.startswith("postgres://"):
    _RAW_URL = _RAW_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif _RAW_URL.startswith("postgresql://") and "+asyncpg" not in _RAW_URL:
    _RAW_URL = _RAW_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

IS_PG = _RAW_URL.startswith("postgresql")
database = databases.Database(_RAW_URL)

# ─── Schema ───────────────────────────────────────────────────────────────────

_ROOMS = """
CREATE TABLE IF NOT EXISTS rooms (
    room_id      TEXT PRIMARY KEY,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    g_min        REAL, g_max REAL, g_realized REAL,
    num_rounds   INTEGER,
    alpha REAL, beta REAL,
    pi_p REAL, pi_r REAL, pi_q REAL, c_u REAL, c_o REAL
)"""

_PLAYERS = """
CREATE TABLE IF NOT EXISTS players (
    player_id     TEXT PRIMARY KEY,
    room_id       TEXT REFERENCES rooms(room_id),
    name          TEXT,
    initial_share REAL
)"""

_ROUND_RESULTS = """
CREATE TABLE IF NOT EXISTS round_results (
    id           {pk},
    room_id      TEXT,
    round_number INTEGER,
    player_id    TEXT,
    player_name  TEXT,
    s_i REAL, a_i REAL, gs_i REAL, pos_i REAL,
    lost REAL, gained REAL, overage REAL, s_new REAL,
    reward_own REAL, reward_2nd REAL, reward_steal REAL,
    penalty REAL, score REAL, A_g REAL,
    recorded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""


async def init_db():
    pk = "BIGSERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    for stmt in [_ROOMS, _PLAYERS, _ROUND_RESULTS.format(pk=pk)]:
        await database.execute(stmt)


# ─── Write helpers ────────────────────────────────────────────────────────────

def _insert(table: str, conflict_col: str) -> str:
    """Return INSERT … ON CONFLICT DO NOTHING (PG) or INSERT OR IGNORE (SQLite)."""
    if IS_PG:
        return f"INSERT INTO {table} {{cols}} VALUES {{vals}} ON CONFLICT ({conflict_col}) DO NOTHING"
    return f"INSERT OR IGNORE INTO {table} {{cols}} VALUES {{vals}}"


async def db_insert_room(room_id: str, params: dict):
    cols = "(room_id, g_min, g_max, num_rounds, alpha, beta, pi_p, pi_r, pi_q, c_u, c_o)"
    vals = "(:room_id,:g_min,:g_max,:num_rounds,:alpha,:beta,:pi_p,:pi_r,:pi_q,:c_u,:c_o)"
    await database.execute(
        _insert("rooms", "room_id").format(cols=cols, vals=vals),
        {"room_id": room_id, **{k: params[k] for k in
          ["g_min","g_max","num_rounds","alpha","beta","pi_p","pi_r","pi_q","c_u","c_o"]}},
    )


async def db_update_room_start(room_id: str, g_realized: float):
    await database.execute(
        "UPDATE rooms SET g_realized = :g WHERE room_id = :rid",
        {"g": g_realized, "rid": room_id},
    )


async def db_insert_player(player_id: str, room_id: str, name: str, initial_share: float):
    cols = "(player_id, room_id, name, initial_share)"
    vals = "(:player_id, :room_id, :name, :initial_share)"
    await database.execute(
        _insert("players", "player_id").format(cols=cols, vals=vals),
        {"player_id": player_id, "room_id": room_id, "name": name, "initial_share": initial_share},
    )


async def db_insert_round(room_id: str, round_number: int, results: dict, A_g: float):
    rows = [
        {
            "room_id": room_id, "round_number": round_number,
            "player_id": pid, "player_name": r["name"],
            "s_i": r["s_i"], "a_i": r["a_i"], "gs_i": r["gs_i"], "pos_i": r["pos_i"],
            "lost": r["lost"], "gained": r["gained"], "overage": r["overage"], "s_new": r["s_new"],
            "reward_own": r["reward_own"], "reward_2nd": r["reward_2nd"],
            "reward_steal": r["reward_steal"], "penalty": r["penalty"],
            "score": r["score"], "A_g": A_g,
        }
        for pid, r in results.items()
    ]
    await database.execute_many(
        """INSERT INTO round_results
           (room_id, round_number, player_id, player_name,
            s_i, a_i, gs_i, pos_i, lost, gained, overage, s_new,
            reward_own, reward_2nd, reward_steal, penalty, score, A_g)
           VALUES
           (:room_id, :round_number, :player_id, :player_name,
            :s_i, :a_i, :gs_i, :pos_i, :lost, :gained, :overage, :s_new,
            :reward_own, :reward_2nd, :reward_steal, :penalty, :score, :A_g)""",
        rows,
    )
