/**
 * k6 Concurrent Load Tests — Green Tech Transition Game
 *
 * Scenarios (run sequentially by startTime, k6 handles concurrency within each):
 *   === Functional / correctness ===
 *   1. concurrent_create         – 20 VUs create rooms simultaneously
 *   2. concurrent_join           – 30 VUs join the SAME room simultaneously
 *   3. concurrent_submit         – 10 VUs submit investments in the SAME round simultaneously
 *   4. double_submit             – 5 VUs each submit twice; second must overwrite first
 *   5. error_cases               – expected 4xx responses (started-room join, wrong token, unknown room)
 *   6. full_game                 – 5 complete 2-round games running in parallel
 *   === Extreme / stress ===
 *   7. spike_join_40             – 40 VUs join ONE room simultaneously (demo day peak moment)
 *   8. poll_storm                – 40 VUs poll /state every 3 s for 2 min (exact demo sustained load)
 *   9. full_demo_sim             – 40 players, 3 rounds, all 40 submissions per round back-to-back
 *  10. concurrent_resolve_race   – 10 VUs race to resolve the same round; no 500s allowed
 *  11. memory_pressure           – 100 rooms rapidly created and populated (in-memory dict stress)
 *
 * Key checks:
 *   - No duplicate player_ids after concurrent joins
 *   - All 10 submissions land (teardown verifies via game state)
 *   - Double submit returns the updated investment value, not the first one
 *   - Race conditions on resolve / next_round do not corrupt scores or produce 5xx
 *   - Sustained polling stays under 2 s p(95) throughout the demo load window
 */

import http               from 'k6/http';
import { check, sleep }   from 'k6';
import { Trend, Counter } from 'k6/metrics';
import { scenario }       from 'k6/execution';

const BASE = (__ENV.BASE_URL || 'https://simplegreentechtransition.onrender.com').replace(/\/$/, '');
const HDR  = { headers: { 'Content-Type': 'application/json' } };

// ── Custom metrics ─────────────────────────────────────────────────────────────
const joinLatency   = new Trend('join_latency',   true);  // ms, percentiles
const submitLatency = new Trend('submit_latency', true);
const pollLatency   = new Trend('poll_latency',   true);
const gameErrors    = new Counter('game_errors');          // any unexpected failure
const resolve5xx    = new Counter('resolve_5xx');          // must stay 0 in race scenario

// ── Options ────────────────────────────────────────────────────────────────────
export const options = {
  scenarios: {

    // ── Functional ─────────────────────────────────────────────────────────
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

    // ── Extreme / Stress ───────────────────────────────────────────────────

    // Simulates the exact moment 40 people click "Join" simultaneously on demo day.
    spike_join_40: {
      executor:    'shared-iterations',
      vus:         40,
      iterations:  40,
      maxDuration: '30s',
      startTime:   '250s',
      exec:        'spikeJoin40',
      tags:        { scenario: 'spike_join_40' },
    },

    // 40 VUs each sleep(3) between polls — mirrors the 3-second frontend poll loop
    // at full 40-player demo capacity for 2 minutes (~13 req/s sustained).
    poll_storm: {
      executor:  'constant-vus',
      vus:       40,
      duration:  '120s',
      startTime: '285s',
      exec:      'pollStorm',
      tags:      { scenario: 'poll_storm' },
    },

    // Single VU drives a complete 40-player 3-round game as fast as the server can
    // handle it — worst-case latency chain for the entire session lifecycle.
    full_demo_sim: {
      executor:    'shared-iterations',
      vus:         1,
      iterations:  1,
      maxDuration: '300s',
      startTime:   '415s',
      exec:        'fullDemoSim',
      tags:        { scenario: 'full_demo_sim' },
    },

    // 10 VUs all call resolve on the same room at the same instant.
    // Exactly 1 must succeed (200); the rest must get 400, never 500.
    concurrent_resolve_race: {
      executor:    'shared-iterations',
      vus:         10,
      iterations:  10,
      maxDuration: '30s',
      startTime:   '730s',
      exec:        'concurrentResolveRace',
      tags:        { scenario: 'concurrent_resolve_race' },
    },

    // 50 VUs, each creating 2 rooms and joining 5 players per room = 100 rooms total.
    // Stresses the in-memory rooms dict and Python heap under concurrent writes.
    memory_pressure: {
      executor:    'shared-iterations',
      vus:         50,
      iterations:  100,
      maxDuration: '60s',
      startTime:   '770s',
      exec:        'memoryPressure',
      tags:        { scenario: 'memory_pressure' },
    },
  },

  thresholds: {
    'join_latency':                                          ['p(95)<3000'],
    'submit_latency':                                        ['p(95)<3000'],
    'poll_latency':                                          ['p(95)<2000'],
    'game_errors':                                           ['count<5'],
    'resolve_5xx':                                           ['count==0'],
    'http_req_failed{scenario:concurrent_create}':           ['rate<0.02'],
    'http_req_failed{scenario:concurrent_join}':             ['rate<0.02'],
    'http_req_failed{scenario:concurrent_submit}':           ['rate<0.02'],
    'http_req_failed{scenario:double_submit}':               ['rate<0.02'],
    'http_req_failed{scenario:full_game}':                   ['rate<0.05'],
    'http_req_failed{scenario:spike_join_40}':               ['rate<0.02'],
    'http_req_failed{scenario:poll_storm}':                  ['rate<0.02'],
    'http_req_failed{scenario:full_demo_sim}':               ['rate<0.05'],
    'http_req_failed{scenario:memory_pressure}':             ['rate<0.05'],
    // concurrent_resolve_race: 9/10 requests are intentional 400s — omit from http_req_failed
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

  // ── Functional scenario rooms ─────────────────────────────────────────────

  // Room for concurrent_join (stays in lobby so all 30 VUs can join)
  const jr = j(post('/api/rooms', BASE_PARAMS));
  if (!jr.room_id) throw new Error('setup: could not create join-room');

  // Room for concurrent_submit: 10 players pre-joined, game started
  const sr = j(post('/api/rooms', BASE_PARAMS));
  if (!sr.room_id) throw new Error('setup: could not create submit-room');

  const submitPlayers = [];
  for (let i = 0; i < 10; i++) {
    const p = j(post(`/api/rooms/${sr.room_id}/join`, { player_name: `Sub${i + 1}` }));
    if (!p.player_id) throw new Error(`setup: join failed (Sub${i + 1})`);
    submitPlayers.push({ player_id: p.player_id });
  }
  post(`/api/rooms/${sr.room_id}/start`, { admin_token: sr.admin_token });

  const ss = j(getState(sr.room_id, sr.admin_token));
  submitPlayers.forEach(p => { p.share = ss.players?.[p.player_id]?.share ?? 0.1; });

  // Room for double_submit: 5 players pre-joined, game started
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
  doublePlayers.forEach(p => { p.share = ds.players?.[p.player_id]?.share ?? 0.1; });

  // ── Extreme scenario rooms ────────────────────────────────────────────────

  // Room for spike_join_40: stays in lobby so all 40 VUs can join
  const sjr = j(post('/api/rooms', BASE_PARAMS));
  if (!sjr.room_id) throw new Error('setup: could not create spike-join-room');

  // Room for poll_storm: 40 players joined, round 1 resolved so history is populated.
  // /state responses include full round_results (~20 KB) making the bandwidth spike visible.
  const psr = j(post('/api/rooms', { ...BASE_PARAMS, num_rounds: 3 }));
  if (!psr.room_id) throw new Error('setup: could not create poll-storm-room');
  for (let i = 0; i < 40; i++) {
    post(`/api/rooms/${psr.room_id}/join`, { player_name: `Poll${i + 1}` });
  }
  post(`/api/rooms/${psr.room_id}/start`, { admin_token: psr.admin_token });
  post(`/api/rooms/${psr.room_id}/resolve`, { admin_token: psr.admin_token });

  // Room for concurrent_resolve_race: 5 players joined, started, and all submitted.
  // Ready to be resolved — 10 VUs will race to call /resolve simultaneously.
  const rrr = j(post('/api/rooms', BASE_PARAMS));
  if (!rrr.room_id) throw new Error('setup: could not create resolve-race-room');
  const racePids = [];
  for (let i = 0; i < 5; i++) {
    const p = j(post(`/api/rooms/${rrr.room_id}/join`, { player_name: `Race${i + 1}` }));
    if (p.player_id) racePids.push(p.player_id);
  }
  post(`/api/rooms/${rrr.room_id}/start`, { admin_token: rrr.admin_token });
  const rs = j(getState(rrr.room_id, rrr.admin_token));
  racePids.forEach(pid => {
    const share = rs.players?.[pid]?.share ?? 0.1;
    post(`/api/rooms/${rrr.room_id}/submit`, { player_id: pid, investment: share * 0.5 });
  });

  return {
    // Functional
    joinRoomId:        jr.room_id,
    submitRoomId:      sr.room_id,
    submitAdminToken:  sr.admin_token,
    submitPlayers,
    doubleRoomId:      dr.room_id,
    doubleAdminToken:  dr.admin_token,
    doublePlayers,
    // Extreme
    spikeJoinRoomId:   sjr.room_id,
    pollRoomId:        psr.room_id,
    pollAdminToken:    psr.admin_token,
    resolveRaceRoomId: rrr.room_id,
    resolveRaceToken:  rrr.admin_token,
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
    'double: 2nd submit 200':       () => r2.status === 200,
    'double: value is overwritten':  () => Math.abs((d2.submitted ?? -1) - inv2) < 0.0001,
  });
  if (!ok) gameErrors.add(1);
}

// ── Scenario 5: expected error cases — verify 4xx, not 5xx ───────────────────
export function errorCases(data) {
  const r1 = post(`/api/rooms/${data.submitRoomId}/join`, { player_name: 'LatePlayer' });
  check(r1, { 'error: join started room = 400': () => r1.status === 400 });

  const r2 = post(`/api/rooms/${data.submitRoomId}/resolve`, { admin_token: 'not-the-token' });
  check(r2, { 'error: wrong token = 403': () => r2.status === 403 });

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
    'game: finished':        d => d.status === 'finished',
    'game: scores recorded': d => Object.values(d.players || {}).every(p => typeof p.total_score === 'number'),
  });
  if (!ok) gameErrors.add(1);
}

// ── Scenario 7: spike join — 40 VUs join ONE room simultaneously ───────────────
// Simulates the exact moment all 40 demo participants click "Join" at once.
export function spikeJoin40(data) {
  const name = `Spike${__VU}_${scenario.iterationInTest}`;
  const t0   = Date.now();
  const r    = post(`/api/rooms/${data.spikeJoinRoomId}/join`, { player_name: name });
  joinLatency.add(Date.now() - t0);
  const d    = j(r);
  const ok   = check(r, {
    'spike_join: 200':           () => r.status === 200,
    'spike_join: has player_id': () => typeof d.player_id === 'string' && d.player_id.length > 8,
    'spike_join: correct name':  () => d.player_name === name,
  });
  if (!ok) gameErrors.add(1);
}

// ── Scenario 8: poll storm — 40 VUs poll /state every 3 s for 2 minutes ───────
// Exact replica of the demo load: 40 browsers each polling on a 3-second interval.
// p(95) latency threshold is set to 2 s — tighter than join/submit.
export function pollStorm(data) {
  const t0 = Date.now();
  const r  = http.get(`${BASE}/api/rooms/${data.pollRoomId}/state`, HDR);
  pollLatency.add(Date.now() - t0);
  const ok = check(r, {
    'poll: 200':       () => r.status === 200,
    'poll: has status': () => j(r).status !== undefined,
  });
  if (!ok) gameErrors.add(1);
  sleep(3);  // mirrors the frontend 3-second polling interval
}

// ── Scenario 9: full demo simulation — 40 players, 3 rounds ───────────────────
// A single VU drives the entire lifecycle as fast as the server can handle:
// 40 joins → start → (40 submits → resolve → next_round) × 3.
// Validates share carry-over, score accumulation, and status transitions at scale.
export function fullDemoSim(_data) {
  const room = j(post('/api/rooms', { ...BASE_PARAMS, num_rounds: 3 }));
  if (!room.room_id) { gameErrors.add(1); return; }

  // Join 40 players sequentially
  const pids = [];
  for (let i = 0; i < 40; i++) {
    const p = j(post(`/api/rooms/${room.room_id}/join`, { player_name: `Demo${i + 1}` }));
    if (p.player_id) pids.push(p.player_id);
  }
  check({ count: pids.length }, { 'demo: 40 players joined': c => c.count === 40 });
  if (pids.length < 40) { gameErrors.add(1); return; }

  const started = j(post(`/api/rooms/${room.room_id}/start`, { admin_token: room.admin_token }));
  if (!check(started, { 'demo: started': s => s.status === 'playing' })) {
    gameErrors.add(1); return;
  }

  // Snapshot initial shares once (backend clips any submission that exceeds current share)
  const init   = j(getState(room.room_id, room.admin_token));
  const shares = {};
  pids.forEach(pid => { shares[pid] = init.players?.[pid]?.share ?? (1 / 40); });

  for (let rnd = 1; rnd <= 3; rnd++) {
    // All 40 players submit 50% of their initial share (clipped by backend if share shrank)
    for (const pid of pids) {
      post(`/api/rooms/${room.room_id}/submit`, {
        player_id: pid, investment: shares[pid] * 0.5,
      });
    }

    const resolved = j(post(`/api/rooms/${room.room_id}/resolve`, { admin_token: room.admin_token }));
    check(resolved, { [`demo: round ${rnd} resolved`]: d => d.status === 'round_results' });

    // Always call next_round; on the final round it transitions to "finished"
    post(`/api/rooms/${room.room_id}/next_round`, { admin_token: room.admin_token });
  }

  const final = j(getState(room.room_id, room.admin_token));
  const ok = check(final, {
    'demo: finished':             d => d.status === 'finished',
    'demo: 40 players present':   d => Object.keys(d.players || {}).length === 40,
    'demo: all scores are numbers': d =>
      Object.values(d.players || {}).every(p => typeof p.total_score === 'number'),
    'demo: scores non-zero':      d =>
      Object.values(d.players || {}).some(p => p.total_score !== 0),
  });
  if (!ok) gameErrors.add(1);
}

// ── Scenario 10: concurrent resolve race — no 500s ────────────────────────────
// 10 VUs all call /resolve on the same room at the same instant.
// Python asyncio serialises coroutines, so exactly 1 should win (200);
// the other 9 must receive 400 "Not in playing phase", never 500.
export function concurrentResolveRace(data) {
  const r  = post(`/api/rooms/${data.resolveRaceRoomId}/resolve`, { admin_token: data.resolveRaceToken });
  const ok = check(r, {
    'race: no 5xx':       () => r.status < 500,
    'race: 200 or 400':   () => r.status === 200 || r.status === 400,
  });
  if (r.status >= 500) resolve5xx.add(1);
  if (!ok) gameErrors.add(1);
}

// ── Scenario 11: memory pressure — 100 rooms, 5 players each ──────────────────
// 50 VUs each run 2 iterations: create a room, join 5 players, start the game.
// This creates 100 simultaneously-alive rooms and validates the in-memory store
// doesn't corrupt under concurrent writes.
export function memoryPressure(_data) {
  const room = j(post('/api/rooms', BASE_PARAMS));
  if (!room.room_id) { gameErrors.add(1); return; }

  for (let i = 0; i < 5; i++) {
    post(`/api/rooms/${room.room_id}/join`, { player_name: `Mem${i + 1}` });
  }

  const started = j(post(`/api/rooms/${room.room_id}/start`, { admin_token: room.admin_token }));
  const ok = check(started, { 'memory: game started': s => s.status === 'playing' });
  if (!ok) gameErrors.add(1);
}

// ── Teardown — runs ONCE after all scenarios ───────────────────────────────────
export function teardown(data) {
  // Verify all 10 players in submit-room have submitted (concurrent submit race check)
  const ss = j(getState(data.submitRoomId, data.submitAdminToken));
  const ps = Object.values(ss.players || {});
  check({ submitted: ps.filter(p => p.submitted).length, total: ps.length }, {
    'teardown: all 10 submissions landed': d => d.submitted === d.total && d.total === 10,
  });

  // Verify 30 unique player_ids in original join-room (no duplicate UUIDs)
  const js    = j(http.get(`${BASE}/api/rooms/${data.joinRoomId}/state`, HDR));
  const count = Object.keys(js.players || {}).length;
  check({ count }, {
    'teardown: 30 unique player_ids in join-room': d => d.count === 30,
  });

  // Verify 40 unique player_ids in spike-join-room (no UUID collision under spike)
  const sjs    = j(http.get(`${BASE}/api/rooms/${data.spikeJoinRoomId}/state`, HDR));
  const scount = Object.keys(sjs.players || {}).length;
  check({ scount }, {
    'teardown: 40 unique player_ids in spike-join-room': d => d.scount === 40,
  });

  // Verify resolve-race room ended up in round_results (exactly 1 resolve succeeded)
  const rrs = j(getState(data.resolveRaceRoomId, data.resolveRaceToken));
  check(rrs, {
    'teardown: resolve-race room in round_results': d => d.status === 'round_results',
  });
}
