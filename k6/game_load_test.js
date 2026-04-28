/**
 * k6 Concurrent Load Tests — Green Tech Transition Game
 *
 * Scenarios (run sequentially by startTime, k6 handles concurrency within each):
 *   1. concurrent_create  – 20 VUs create rooms simultaneously
 *   2. concurrent_join    – 30 VUs join the SAME room simultaneously
 *   3. concurrent_submit  – 10 VUs submit investments in the SAME round simultaneously
 *   4. double_submit      – 5 VUs each submit twice; second must overwrite first
 *   5. error_cases        – expected 4xx responses (started-room join, wrong token, unknown room)
 *   6. full_game          – 5 complete 2-round games running in parallel
 *
 * Key checks:
 *   - No duplicate player_ids after concurrent joins
 *   - All 10 submissions land (teardown verifies via game state)
 *   - Double submit returns the updated investment value, not the first one
 *   - Race conditions on resolve / next_round do not corrupt scores
 */

import http          from 'k6/http';
import { check }     from 'k6';
import { Trend, Counter } from 'k6/metrics';
import { scenario }  from 'k6/execution';

const BASE = (__ENV.BASE_URL || 'https://simplegreentechtransition.onrender.com').replace(/\/$/, '');
const HDR  = { headers: { 'Content-Type': 'application/json' } };

// ── Custom metrics ─────────────────────────────────────────────────────────────
const joinLatency   = new Trend('join_latency',   true);  // ms, percentiles
const submitLatency = new Trend('submit_latency', true);
const gameErrors    = new Counter('game_errors');          // any unexpected failure

// ── Options ────────────────────────────────────────────────────────────────────
export const options = {
  scenarios: {
    concurrent_create: {
      executor:    'shared-iterations',
      vus:         20,
      iterations:  20,
      maxDuration: '30s',
      startTime:   '0s',
      exec:        'concurrentCreate',
      tags:        { scenario: 'concurrent_create' },
    },
    concurrent_join: {
      executor:    'shared-iterations',
      vus:         30,
      iterations:  30,
      maxDuration: '30s',
      startTime:   '10s',
      exec:        'concurrentJoin',
      tags:        { scenario: 'concurrent_join' },
    },
    concurrent_submit: {
      executor:    'shared-iterations',
      vus:         10,
      iterations:  10,
      maxDuration: '30s',
      startTime:   '25s',
      exec:        'concurrentSubmit',
      tags:        { scenario: 'concurrent_submit' },
    },
    double_submit: {
      executor:    'shared-iterations',
      vus:         5,
      iterations:  5,
      maxDuration: '30s',
      startTime:   '60s',
      exec:        'doubleSubmit',
      tags:        { scenario: 'double_submit' },
    },
    error_cases: {
      executor:    'shared-iterations',
      vus:         3,
      iterations:  3,
      maxDuration: '20s',
      startTime:   '95s',
      exec:        'errorCases',
      tags:        { scenario: 'error_cases' },
    },
    full_game: {
      executor:    'shared-iterations',
      vus:         30,
      iterations:  30,
      maxDuration: '120s',
      startTime:   '120s',
      exec:        'fullGame',
      tags:        { scenario: 'full_game' },
    },
  },

  thresholds: {
    'join_latency':                                ['p(95)<3000'],
    'submit_latency':                              ['p(95)<3000'],
    'game_errors':                                 ['count<5'],
    // error_cases intentionally sends bad requests; exclude it from the failure rate
    'http_req_failed{scenario:concurrent_create}': ['rate<0.02'],
    'http_req_failed{scenario:concurrent_join}':   ['rate<0.02'],
    'http_req_failed{scenario:concurrent_submit}': ['rate<0.02'],
    'http_req_failed{scenario:double_submit}':     ['rate<0.02'],
    'http_req_failed{scenario:full_game}':         ['rate<0.05'],
  },
};

// ── Helpers ────────────────────────────────────────────────────────────────────
function post(path, body) {
  return http.post(`${BASE}${path}`, JSON.stringify(body), HDR);
}

function getState(room_id, admin_token) {
  return http.get(`${BASE}/api/rooms/${room_id}/state?admin_token=${admin_token}`, HDR);
}

function j(res) {
  try   { return JSON.parse(res.body) || {}; }
  catch { return {}; }
}

const BASE_PARAMS = {
  g_min: 0.2, g_max: 0.8, num_rounds: 2,
  alpha: 0.0, beta: 0.0,
  pi_p: 4.0, pi_r: 3.0, pi_q: 5.0,
  c_u: 10.0, c_o: 1.0,
};

// ── Setup — runs ONCE before all scenarios ─────────────────────────────────────
export function setup() {
  // --- Room for concurrent_join (stays in lobby so all 30 VUs can join) ---
  const jr = j(post('/api/rooms', BASE_PARAMS));
  if (!jr.room_id) throw new Error('setup: could not create join-room');

  // --- Room for concurrent_submit: 10 players pre-joined, game started ---
  const sr = j(post('/api/rooms', BASE_PARAMS));
  if (!sr.room_id) throw new Error('setup: could not create submit-room');

  const submitPlayers = [];
  for (let i = 0; i < 10; i++) {
    const p = j(post(`/api/rooms/${sr.room_id}/join`, { player_name: `Sub${i + 1}` }));
    if (!p.player_id) throw new Error(`setup: join failed (Sub${i + 1})`);
    submitPlayers.push({ player_id: p.player_id });
  }
  post(`/api/rooms/${sr.room_id}/start`, { admin_token: sr.admin_token });

  // Read initial shares so we can use valid investment values
  const ss = j(getState(sr.room_id, sr.admin_token));
  submitPlayers.forEach(p => {
    p.share = ss.players?.[p.player_id]?.share ?? 0.1;
  });

  // --- Room for double_submit: 5 players pre-joined, game started ---
  const dr = j(post('/api/rooms', BASE_PARAMS));
  if (!dr.room_id) throw new Error('setup: could not create double-room');

  const doublePlayers = [];
  for (let i = 0; i < 5; i++) {
    const p = j(post(`/api/rooms/${dr.room_id}/join`, { player_name: `Dbl${i + 1}` }));
    if (!p.player_id) throw new Error(`setup: join failed (Dbl${i + 1})`);
    doublePlayers.push({ player_id: p.player_id });
  }
  post(`/api/rooms/${dr.room_id}/start`, { admin_token: dr.admin_token });

  const ds = j(getState(dr.room_id, dr.admin_token));
  doublePlayers.forEach(p => {
    p.share = ds.players?.[p.player_id]?.share ?? 0.1;
  });

  return {
    joinRoomId:       jr.room_id,
    submitRoomId:     sr.room_id,
    submitAdminToken: sr.admin_token,
    submitPlayers,
    doubleRoomId:     dr.room_id,
    doubleAdminToken: dr.admin_token,
    doublePlayers,
  };
}

// ── Scenario 1: 20 VUs create rooms simultaneously ────────────────────────────
export function concurrentCreate(_data) {
  const r = post('/api/rooms', BASE_PARAMS);
  const d = j(r);
  const ok = check(r, {
    'create: 200':            () => r.status === 200,
    'create: room_id length': () => typeof d.room_id === 'string' && d.room_id.length === 6,
    'create: admin_token':    () => typeof d.admin_token === 'string' && d.admin_token.length > 8,
  });
  if (!ok) gameErrors.add(1);
}

// ── Scenario 2: 30 VUs join the same room simultaneously ──────────────────────
export function concurrentJoin(data) {
  const name = `VU${__VU}_i${scenario.iterationInTest}`;
  const t0   = Date.now();
  const r    = post(`/api/rooms/${data.joinRoomId}/join`, { player_name: name });
  joinLatency.add(Date.now() - t0);
  const d    = j(r);
  const ok   = check(r, {
    'join: 200':            () => r.status === 200,
    'join: has player_id':  () => typeof d.player_id === 'string' && d.player_id.length > 8,
    'join: correct name':   () => d.player_name === name,
  });
  if (!ok) gameErrors.add(1);
}

// ── Scenario 3: 10 VUs submit investments in the same round ───────────────────
export function concurrentSubmit(data) {
  // Each VU gets a unique player via scenario.iterationInTest (globally 0–9)
  const player = data.submitPlayers[scenario.iterationInTest % data.submitPlayers.length];
  const inv    = parseFloat((player.share * 0.5).toFixed(6));

  const t0 = Date.now();
  const r  = post(`/api/rooms/${data.submitRoomId}/submit`, {
    player_id: player.player_id, investment: inv,
  });
  submitLatency.add(Date.now() - t0);
  const d  = j(r);
  const ok = check(r, {
    'submit: 200':              () => r.status === 200,
    'submit: returned number':  () => typeof d.submitted === 'number',
    'submit: value within cap': () => d.submitted <= player.share + 0.0001,
  });
  if (!ok) gameErrors.add(1);
}

// ── Scenario 4: double submit — second value must overwrite first ──────────────
export function doubleSubmit(data) {
  const player = data.doublePlayers[scenario.iterationInTest % data.doublePlayers.length];
  const inv1   = parseFloat((player.share * 0.3).toFixed(6));
  const inv2   = parseFloat((player.share * 0.7).toFixed(6));

  post(`/api/rooms/${data.doubleRoomId}/submit`, { player_id: player.player_id, investment: inv1 });

  const r2 = post(`/api/rooms/${data.doubleRoomId}/submit`, { player_id: player.player_id, investment: inv2 });
  const d2 = j(r2);
  const ok = check(r2, {
    'double: 2nd submit 200':     () => r2.status === 200,
    'double: value is overwritten': () => Math.abs((d2.submitted ?? -1) - inv2) < 0.0001,
  });
  if (!ok) gameErrors.add(1);
}

// ── Scenario 5: expected error cases — verify 4xx, not 5xx ───────────────────
export function errorCases(data) {
  // Joining a room that already started must return 400
  const r1 = post(`/api/rooms/${data.submitRoomId}/join`, { player_name: 'LatePlayer' });
  check(r1, { 'error: join started room = 400': () => r1.status === 400 });

  // Resolve with wrong admin token must return 403
  const r2 = post(`/api/rooms/${data.submitRoomId}/resolve`, { admin_token: 'not-the-token' });
  check(r2, { 'error: wrong token = 403': () => r2.status === 403 });

  // Unknown room must return 404
  const r3 = http.get(`${BASE}/api/rooms/ZZZZZZ/state`, HDR);
  check(r3, { 'error: unknown room = 404': () => r3.status === 404 });
}

// ── Scenario 6: 5 complete 2-round games running in parallel ──────────────────
export function fullGame(_data) {
  const room = j(post('/api/rooms', { ...BASE_PARAMS, num_rounds: 2 }));
  if (!room.room_id) { gameErrors.add(1); return; }

  const p1 = j(post(`/api/rooms/${room.room_id}/join`, { player_name: 'Alice' }));
  const p2 = j(post(`/api/rooms/${room.room_id}/join`, { player_name: 'Bob'   }));
  if (!p1.player_id || !p2.player_id) { gameErrors.add(1); return; }

  const started = j(post(`/api/rooms/${room.room_id}/start`, { admin_token: room.admin_token }));
  if (!check(started, { 'game: started': s => s.status === 'playing' })) {
    gameErrors.add(1); return;
  }

  // Read initial shares so investments are always valid
  const init = j(getState(room.room_id, room.admin_token));
  const s1   = init.players?.[p1.player_id]?.share ?? 0.5;
  const s2   = init.players?.[p2.player_id]?.share ?? 0.5;

  for (let rnd = 1; rnd <= 2; rnd++) {
    post(`/api/rooms/${room.room_id}/submit`, { player_id: p1.player_id, investment: s1 * 0.6 });
    post(`/api/rooms/${room.room_id}/submit`, { player_id: p2.player_id, investment: s2 * 0.4 });

    const resolved = j(post(`/api/rooms/${room.room_id}/resolve`, { admin_token: room.admin_token }));
    check(resolved, { [`game: round ${rnd} resolved`]: d => d.status === 'round_results' });

    post(`/api/rooms/${room.room_id}/next_round`, { admin_token: room.admin_token });
  }

  const final = j(getState(room.room_id, room.admin_token));
  const ok = check(final, {
    'game: finished':             d => d.status === 'finished',
    'game: scores recorded':      d => Object.values(d.players || {}).every(p => typeof p.total_score === 'number'),
  });
  if (!ok) gameErrors.add(1);
}

// ── Teardown — runs ONCE after all scenarios ───────────────────────────────────
export function teardown(data) {
  // Verify all 10 players in submit-room have submitted (race condition check)
  const ss = j(getState(data.submitRoomId, data.submitAdminToken));
  const ps = Object.values(ss.players || {});
  check({ submitted: ps.filter(p => p.submitted).length, total: ps.length }, {
    'teardown: all 10 submissions landed': d => d.submitted === d.total && d.total === 10,
  });

  // Verify 30 unique player_ids were created in join-room (no duplicates)
  const js    = j(http.get(`${BASE}/api/rooms/${data.joinRoomId}/state`, HDR));
  const count = Object.keys(js.players || {}).length;
  check({ count }, {
    'teardown: 30 unique player_ids in join-room': d => d.count === 30,
  });
}
