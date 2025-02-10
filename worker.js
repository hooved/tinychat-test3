const kernelsReady = (async () => {
  // can't get browser to use updated versions except with cache-busting query string
  const exports = await import(`./net_clang.js?version=${Date.now()}`);
  Object.assign(self, exports);
})();

async function initStateDict(event) {
  await kernelsReady;
  self.model = await self.transformer(event.data);
  self.addEventListener("message", loadStateDict);
  self.removeEventListener("message", initStateDict);
  self.postMessage(self.model.state_dict);
  delete self.model.state_dict;
}

function loadStateDict(event) {
  if (event.data === "done") {
    self.addEventListener("message", inference);
    self.removeEventListener("message", loadStateDict);
  }
  else {
    const part = event.data;
    for (const [wasm_idx, wasm_offset] of part.wasm_offsets) {
      //self.model.wasm[wasm_idx].HEAPU8.set(part.bytes, wasm_offset);
      const filePath = '/persistent/myData.bin';
      self.model.wasm[wasm_idx].FS.writeFile(filePath, part.bytes);
      self.model.wasm[wasm_idx]._load_buf(wasm_offset);
      if (false) {
      //if (wasm_idx === 0) {
        // Suppose instance is your instantiated module (with FS mounted, etc.)
        const filePath = '/persistent/myData.bin';
        // Write the binary data into the Emscripten file system.
        // FS.writeFile can accept a Uint8Array directly.
        self.model.wasm[0].FS.writeFile(filePath, part.bytes);
        console.log("write successful");
        //self.model.wasm[0]._start();
        //self.model.wasm[0]._asdf();
      }
    }
  }
  self.postMessage("success");
}

function inference(event) {
  const [tok, start_pos] = event.data;
  const int32tok = new Int32Array([tok]);
  const uint8tok = new Uint8Array(int32tok.buffer);
  const uint8nextTok = self.model.run(uint8tok, start_pos)[0];
  const int32nextTok = new Int32Array(uint8nextTok.buffer);
  self.postMessage(int32nextTok[0]);
}

self.addEventListener('message', initStateDict);