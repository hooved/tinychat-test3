const kernelsReady = (async () => {
  // can't get browser to use updated versions except with cache-busting query string
  const exports = await import(`./net_clang.js?version=${Date.now()}`);
  Object.assign(self, exports);
})();

//const parts = [];
const files = [];
const order = [14, 13, 7, 6, 5, 4, 3, 2, 1, 0, 12, 11, 10, 9, 8, 63];

async function initStateDict(event) {
  await kernelsReady;
  self.model = await self.transformer(event.data);
  self.inputPtr = self.model.wasm._malloc(4);
  self.outputPtr = self.model.wasm._malloc(4);
  self.addEventListener("message", loadStateDict);
  self.removeEventListener("message", initStateDict);
  self.postMessage(self.model.state_dict);
  delete self.model.state_dict;
}

async function loadStateDict(event) {
  if (event.data === "done") {
    let delay = 100;
    for (let i=0; i<files.length; i++) {
      const ptr = self.model.wasm._malloc(files[i].bytes.length);
      self.model.wasm.HEAPU8.set(files[i].bytes, ptr);
      files[i].bytes = null;
      //if (i > 25) delay = 150;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    self.addEventListener("message", inference);
    self.removeEventListener("message", loadStateDict);
  }
  else {
    const file = event.data;
    files.push(file);
    //self.model.wasm.HEAPU8.set(part.bytes, part.target_start_pos);
  }
  self.postMessage("success");
}

function inference(event) {
  const [tok, start_pos] = event.data;
  const int32tok = new Int32Array([tok]);
  const uint8tok = new Uint8Array(int32tok.buffer);
  self.model.wasm.HEAPU8.set(uint8tok, self.inputPtr);
  self.model.wasm._net(self.outputPtr, self.inputPtr, start_pos);
  const uint8nextTok = self.model.wasm.HEAPU8.slice(self.outputPtr, self.outputPtr + 4);
  const int32nextTok = new Int32Array(uint8nextTok.buffer);
  self.postMessage(int32nextTok[0]);
}

self.addEventListener('message', initStateDict);