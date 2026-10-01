// Game state and UI. Rules mirror tools/othello.py; the engine runs in engine.worker.js.
// Board: 64 chars, row-major from the top-left, '.' empty, '0' black, '1' white. Black moves first.

const START = ".".repeat(27) + "10" + ".".repeat(6) + "01" + ".".repeat(27);
const DIRECTIONS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

function flips(board, player, index) {
  if (board[index] !== ".") return [];
  const opponent = player === "0" ? "1" : "0";
  const row = Math.floor(index / 8), col = index % 8;
  const result = [];
  for (const [dr, dc] of DIRECTIONS) {
    const line = [];
    let r = row + dr, c = col + dc;
    while (r >= 0 && r < 8 && c >= 0 && c < 8 && board[r * 8 + c] === opponent) {
      line.push(r * 8 + c);
      r += dr; c += dc;
    }
    if (line.length && r >= 0 && r < 8 && c >= 0 && c < 8 && board[r * 8 + c] === player) result.push(...line);
  }
  return result;
}

const legalMoves = (board, player) => [...Array(64).keys()].filter((i) => flips(board, player, i).length);
const other = (player) => (player === "0" ? "1" : "0");
const count = (board, player) => [...board].filter((c) => c === player).length;
const toName = (index) => "abcdefgh"[index % 8] + (Math.floor(index / 8) + 1);
const fromName = (name) => (Number(name.slice(1)) - 1) * 8 + (name.charCodeAt(0) - 97);

function play(board, player, index) {
  const cells = [...board];
  const flipped = flips(board, player, index);
  for (const i of [...flipped, index]) cells[i] = player;
  return { board: cells.join(""), flipped };
}

// ---- state ----
const state = {
  board: START,
  toMove: "0",
  human: "0",
  history: [],          // previous {board, toMove, lastMove} for undo
  lastMove: null,
  flipped: [],
  thinking: false,
  engineReady: false,
  requestId: 0,
  message: "",
};

const els = {
  board: document.getElementById("board"),
  status: document.getElementById("status"),
  colour: document.getElementById("colour"),
  think: document.getElementById("think"),
  newGame: document.getElementById("new-game"),
  undo: document.getElementById("undo"),
  scoreBlack: document.getElementById("score-black"),
  scoreWhite: document.getElementById("score-white"),
  evalText: document.getElementById("eval-text"),
  evalFill: document.getElementById("eval-fill"),
};

const cells = [];
for (let i = 0; i < 64; i++) {
  const cell = document.createElement("button");
  cell.type = "button";
  cell.className = "cell";
  cell.setAttribute("role", "gridcell");
  cell.addEventListener("click", () => humanMove(i));
  els.board.appendChild(cell);
  cells.push(cell);
}

const worker = new Worker("engine.worker.js");
worker.onmessage = ({ data }) => {
  if (data.type === "ready") {
    state.engineReady = true;
    advance();
  } else if (data.type === "move" && data.id === state.requestId) {
    state.thinking = false;
    showEval(data.value);
    if (data.move !== "pass") applyMove(fromName(data.move));
    advance();
  }
};
worker.onerror = () => {
  state.message = "The engine failed to load. Try reloading the page.";
  render();
};

function gameOver(board) {
  return !legalMoves(board, "0").length && !legalMoves(board, "1").length;
}

function applyMove(index) {
  state.history.push({ board: state.board, toMove: state.toMove, lastMove: state.lastMove });
  const result = play(state.board, state.toMove, index);
  state.board = result.board;
  state.flipped = result.flipped;
  state.lastMove = index;
  state.toMove = other(state.toMove);
}

// Handle passes and game end, then hand the turn to whoever should move.
function advance() {
  state.message = "";
  if (gameOver(state.board)) {
    render();
    return;
  }
  if (!legalMoves(state.board, state.toMove).length) {
    const who = state.toMove === state.human ? "You have" : "The engine has";
    state.message = `${who} no legal move, so the turn passes.`;
    state.toMove = other(state.toMove);
  }
  if (state.toMove !== state.human && state.engineReady) {
    state.thinking = true;
    state.requestId++;
    worker.postMessage({ id: state.requestId, position: state.board + state.toMove, timeMs: Number(els.think.value) });
  }
  render();
}

function humanMove(index) {
  if (state.thinking || state.toMove !== state.human || !flips(state.board, state.human, index).length) return;
  applyMove(index);
  advance();
}

function newGame() {
  Object.assign(state, {
    board: START, toMove: "0", human: els.colour.value, history: [],
    lastMove: null, flipped: [], thinking: false, message: "",
  });
  state.requestId++;   // ignore any search still running for the old game
  showEval(null);
  advance();
}

function undo() {
  // step back to the human's previous turn
  while (state.history.length) {
    Object.assign(state, state.history.pop(), { flipped: [] });
    if (state.toMove === state.human) break;
  }
  state.thinking = false;
  state.requestId++;
  advance();
}

function showEval(value) {
  if (value === null || Number.isNaN(value)) {
    els.evalText.textContent = "–";
    els.evalFill.style.width = "50%";
    return;
  }
  const black = Math.round(((value + 1) / 2) * 100);
  els.evalText.textContent = black >= 50 ? `Black ${black}%` : `White ${100 - black}%`;
  els.evalFill.style.width = `${black}%`;
}

function statusText() {
  if (!state.engineReady) return "Loading engine…";
  if (gameOver(state.board)) {
    const b = count(state.board, "0"), w = count(state.board, "1");
    const humanDiscs = state.human === "0" ? b : w, engineDiscs = state.human === "0" ? w : b;
    if (humanDiscs === engineDiscs) return `Draw, ${b}–${w}.`;
    return humanDiscs > engineDiscs ? `You win ${humanDiscs}–${engineDiscs}!` : `The engine wins ${engineDiscs}–${humanDiscs}.`;
  }
  const prefix = state.message ? state.message + " " : "";
  return prefix + (state.thinking ? "Engine is thinking…" : "Your move.");
}

function render() {
  const legal = !state.thinking && state.toMove === state.human && state.engineReady
    ? new Set(legalMoves(state.board, state.human)) : new Set();
  const flipped = new Set(state.flipped);
  cells.forEach((cell, i) => {
    const value = state.board[i];
    cell.className = "cell" + (legal.has(i) ? " legal" : "") + (state.lastMove === i ? " last" : "");
    cell.setAttribute("aria-label", `${toName(i)}: ${value === "0" ? "black" : value === "1" ? "white" : legal.has(i) ? "legal move" : "empty"}`);
    let piece = cell.firstChild;
    if (value === ".") {
      if (piece) piece.remove();
      return;
    }
    if (!piece) {
      piece = document.createElement("span");
      cell.appendChild(piece);
    }
    piece.className = "piece " + (value === "0" ? "black" : "white") + (flipped.has(i) ? " flipped" : "");
  });
  state.flipped = [];

  els.scoreBlack.querySelector(".count").textContent = count(state.board, "0");
  els.scoreWhite.querySelector(".count").textContent = count(state.board, "1");
  els.scoreBlack.querySelector(".who").textContent = state.human === "0" ? "You" : "Engine";
  els.scoreWhite.querySelector(".who").textContent = state.human === "1" ? "You" : "Engine";
  const over = gameOver(state.board);
  els.scoreBlack.classList.toggle("to-move", !over && state.toMove === "0");
  els.scoreWhite.classList.toggle("to-move", !over && state.toMove === "1");
  els.status.textContent = statusText();
  els.undo.disabled = !state.history.length;
}

els.newGame.addEventListener("click", newGame);
els.undo.addEventListener("click", undo);
els.colour.addEventListener("change", newGame);
render();
