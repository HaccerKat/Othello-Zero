// Runs the WebAssembly engine off the main thread so the page stays responsive while it searches.
importScripts("engine.js");

const ready = createEngine().then((module) =>
  module.cwrap("best_move", "string", ["string", "number", "number"])
);

ready.then(() => postMessage({ type: "ready" }));

onmessage = async ({ data }) => {
  const bestMove = await ready;
  const [move, value] = bestMove(data.position, data.timeMs, 0).split(" ");
  postMessage({ type: "move", id: data.id, move, value: parseFloat(value) });
};
