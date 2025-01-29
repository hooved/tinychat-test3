import transformerModule from './transformer.js'
import q6k_to_f32Module from './q6k_to_f32.js'
import q6k_to_int8_2048_2048Module from './q6k_to_int8_2048_2048.js'
import q6k_to_int8_512_2048Module from './q6k_to_int8_512_2048.js'
import q6k_to_int8_8192_2048Module from './q6k_to_int8_8192_2048.js'
import q6k_to_int8_2048_8192Module from './q6k_to_int8_2048_8192.js'

var transformer = async function(metadata, progress) {

  const wasm = await transformerModule();
  const weightNames = ["tok_embeddings.arange", "output.weight", "layers.0.attention_norm.weight", "layers.0.attention.wk.weight", "layers.0.attention.wk.scale", "layers.0.attention.wv.weight", "layers.0.attention.wv.scale", "layers.0.attention.cache_kv", "freqs_cis", "layers.0.attention.wq.weight", "layers.0.attention.wq.scale", "layers.0.attention.wo.weight", "layers.0.attention.wo.scale", "layers.0.ffn_norm.weight", "layers.0.feed_forward.w3.weight", "layers.0.feed_forward.w3.scale", "layers.0.feed_forward.w1.weight", "layers.0.feed_forward.w1.scale", "layers.0.feed_forward.w2.weight", "layers.0.feed_forward.w2.scale", "layers.1.attention_norm.weight", "layers.1.attention.wk.weight", "layers.1.attention.wk.scale", "layers.1.attention.wv.weight", "layers.1.attention.wv.scale", "layers.1.attention.cache_kv", "layers.1.attention.wq.weight", "layers.1.attention.wq.scale", "layers.1.attention.wo.weight", "layers.1.attention.wo.scale", "layers.1.ffn_norm.weight", "layers.1.feed_forward.w3.weight", "layers.1.feed_forward.w3.scale", "layers.1.feed_forward.w1.weight", "layers.1.feed_forward.w1.scale", "layers.1.feed_forward.w2.weight", "layers.1.feed_forward.w2.scale", "layers.2.attention_norm.weight", "layers.2.attention.wk.weight", "layers.2.attention.wk.scale", "layers.2.attention.wv.weight", "layers.2.attention.wv.scale", "layers.2.attention.cache_kv", "layers.2.attention.wq.weight", "layers.2.attention.wq.scale", "layers.2.attention.wo.weight", "layers.2.attention.wo.scale", "layers.2.ffn_norm.weight", "layers.2.feed_forward.w3.weight", "layers.2.feed_forward.w3.scale", "layers.2.feed_forward.w1.weight", "layers.2.feed_forward.w1.scale", "layers.2.feed_forward.w2.weight", "layers.2.feed_forward.w2.scale", "layers.3.attention_norm.weight", "layers.3.attention.wk.weight", "layers.3.attention.wk.scale", "layers.3.attention.wv.weight", "layers.3.attention.wv.scale", "layers.3.attention.cache_kv", "layers.3.attention.wq.weight", "layers.3.attention.wq.scale", "layers.3.attention.wo.weight", "layers.3.attention.wo.scale", "layers.3.ffn_norm.weight", "layers.3.feed_forward.w3.weight", "layers.3.feed_forward.w3.scale", "layers.3.feed_forward.w1.weight", "layers.3.feed_forward.w1.scale", "layers.3.feed_forward.w2.weight", "layers.3.feed_forward.w2.scale", "layers.4.attention_norm.weight", "layers.4.attention.wk.weight", "layers.4.attention.wk.scale", "layers.4.attention.wv.weight", "layers.4.attention.wv.scale", "layers.4.attention.cache_kv", "layers.4.attention.wq.weight", "layers.4.attention.wq.scale", "layers.4.attention.wo.weight", "layers.4.attention.wo.scale", "layers.4.ffn_norm.weight", "layers.4.feed_forward.w3.weight", "layers.4.feed_forward.w3.scale", "layers.4.feed_forward.w1.weight", "layers.4.feed_forward.w1.scale", "layers.4.feed_forward.w2.weight", "layers.4.feed_forward.w2.scale", "layers.5.attention_norm.weight", "layers.5.attention.wk.weight", "layers.5.attention.wk.scale", "layers.5.attention.wv.weight", "layers.5.attention.wv.scale", "layers.5.attention.cache_kv", "layers.5.attention.wq.weight", "layers.5.attention.wq.scale", "layers.5.attention.wo.weight", "layers.5.attention.wo.scale", "layers.5.ffn_norm.weight", "layers.5.feed_forward.w3.weight", "layers.5.feed_forward.w3.scale", "layers.5.feed_forward.w1.weight", "layers.5.feed_forward.w1.scale", "layers.5.feed_forward.w2.weight", "layers.5.feed_forward.w2.scale", "layers.6.attention_norm.weight", "layers.6.attention.wk.weight", "layers.6.attention.wk.scale", "layers.6.attention.wv.weight", "layers.6.attention.wv.scale", "layers.6.attention.cache_kv", "layers.6.attention.wq.weight", "layers.6.attention.wq.scale", "layers.6.attention.wo.weight", "layers.6.attention.wo.scale", "layers.6.ffn_norm.weight", "layers.6.feed_forward.w3.weight", "layers.6.feed_forward.w3.scale", "layers.6.feed_forward.w1.weight", "layers.6.feed_forward.w1.scale", "layers.6.feed_forward.w2.weight", "layers.6.feed_forward.w2.scale", "layers.7.attention_norm.weight", "layers.7.attention.wk.weight", "layers.7.attention.wk.scale", "layers.7.attention.wv.weight", "layers.7.attention.wv.scale", "layers.7.attention.cache_kv", "layers.7.attention.wq.weight", "layers.7.attention.wq.scale", "layers.7.attention.wo.weight", "layers.7.attention.wo.scale", "layers.7.ffn_norm.weight", "layers.7.feed_forward.w3.weight", "layers.7.feed_forward.w3.scale", "layers.7.feed_forward.w1.weight", "layers.7.feed_forward.w1.scale", "layers.7.feed_forward.w2.weight", "layers.7.feed_forward.w2.scale", "layers.8.attention_norm.weight", "layers.8.attention.wk.weight", "layers.8.attention.wk.scale", "layers.8.attention.wv.weight", "layers.8.attention.wv.scale", "layers.8.attention.cache_kv", "layers.8.attention.wq.weight", "layers.8.attention.wq.scale", "layers.8.attention.wo.weight", "layers.8.attention.wo.scale", "layers.8.ffn_norm.weight", "layers.8.feed_forward.w3.weight", "layers.8.feed_forward.w3.scale", "layers.8.feed_forward.w1.weight", "layers.8.feed_forward.w1.scale", "layers.8.feed_forward.w2.weight", "layers.8.feed_forward.w2.scale", "layers.9.attention_norm.weight", "layers.9.attention.wk.weight", "layers.9.attention.wk.scale", "layers.9.attention.wv.weight", "layers.9.attention.wv.scale", "layers.9.attention.cache_kv", "layers.9.attention.wq.weight", "layers.9.attention.wq.scale", "layers.9.attention.wo.weight", "layers.9.attention.wo.scale", "layers.9.ffn_norm.weight", "layers.9.feed_forward.w3.weight", "layers.9.feed_forward.w3.scale", "layers.9.feed_forward.w1.weight", "layers.9.feed_forward.w1.scale", "layers.9.feed_forward.w2.weight", "layers.9.feed_forward.w2.scale", "layers.10.attention_norm.weight", "layers.10.attention.wk.weight", "layers.10.attention.wk.scale", "layers.10.attention.wv.weight", "layers.10.attention.wv.scale", "layers.10.attention.cache_kv", "layers.10.attention.wq.weight", "layers.10.attention.wq.scale", "layers.10.attention.wo.weight", "layers.10.attention.wo.scale", "layers.10.ffn_norm.weight", "layers.10.feed_forward.w3.weight", "layers.10.feed_forward.w3.scale", "layers.10.feed_forward.w1.weight", "layers.10.feed_forward.w1.scale", "layers.10.feed_forward.w2.weight", "layers.10.feed_forward.w2.scale", "layers.11.attention_norm.weight", "layers.11.attention.wk.weight", "layers.11.attention.wk.scale", "layers.11.attention.wv.weight", "layers.11.attention.wv.scale", "layers.11.attention.cache_kv", "layers.11.attention.wq.weight", "layers.11.attention.wq.scale", "layers.11.attention.wo.weight", "layers.11.attention.wo.scale", "layers.11.ffn_norm.weight", "layers.11.feed_forward.w3.weight", "layers.11.feed_forward.w3.scale", "layers.11.feed_forward.w1.weight", "layers.11.feed_forward.w1.scale", "layers.11.feed_forward.w2.weight", "layers.11.feed_forward.w2.scale", "layers.12.attention_norm.weight", "layers.12.attention.wk.weight", "layers.12.attention.wk.scale", "layers.12.attention.wv.weight", "layers.12.attention.wv.scale", "layers.12.attention.cache_kv", "layers.12.attention.wq.weight", "layers.12.attention.wq.scale", "layers.12.attention.wo.weight", "layers.12.attention.wo.scale", "layers.12.ffn_norm.weight", "layers.12.feed_forward.w3.weight", "layers.12.feed_forward.w3.scale", "layers.12.feed_forward.w1.weight", "layers.12.feed_forward.w1.scale", "layers.12.feed_forward.w2.weight", "layers.12.feed_forward.w2.scale", "layers.13.attention_norm.weight", "layers.13.attention.wk.weight", "layers.13.attention.wk.scale", "layers.13.attention.wv.weight", "layers.13.attention.wv.scale", "layers.13.attention.cache_kv", "layers.13.attention.wq.weight", "layers.13.attention.wq.scale", "layers.13.attention.wo.weight", "layers.13.attention.wo.scale", "layers.13.ffn_norm.weight", "layers.13.feed_forward.w3.weight", "layers.13.feed_forward.w3.scale", "layers.13.feed_forward.w1.weight", "layers.13.feed_forward.w1.scale", "layers.13.feed_forward.w2.weight", "layers.13.feed_forward.w2.scale", "layers.14.attention_norm.weight", "layers.14.attention.wk.weight", "layers.14.attention.wk.scale", "layers.14.attention.wv.weight", "layers.14.attention.wv.scale", "layers.14.attention.cache_kv", "layers.14.attention.wq.weight", "layers.14.attention.wq.scale", "layers.14.attention.wo.weight", "layers.14.attention.wo.scale", "layers.14.ffn_norm.weight", "layers.14.feed_forward.w3.weight", "layers.14.feed_forward.w3.scale", "layers.14.feed_forward.w1.weight", "layers.14.feed_forward.w1.scale", "layers.14.feed_forward.w2.weight", "layers.14.feed_forward.w2.scale", "layers.15.attention_norm.weight", "layers.15.attention.wk.weight", "layers.15.attention.wk.scale", "layers.15.attention.wv.weight", "layers.15.attention.wv.scale", "layers.15.attention.cache_kv", "layers.15.attention.wq.weight", "layers.15.attention.wq.scale", "layers.15.attention.wo.weight", "layers.15.attention.wo.scale", "layers.15.ffn_norm.weight", "layers.15.feed_forward.w3.weight", "layers.15.feed_forward.w3.scale", "layers.15.feed_forward.w1.weight", "layers.15.feed_forward.w1.scale", "layers.15.feed_forward.w2.weight", "layers.15.feed_forward.w2.scale", "norm.weight"];
  for (const [i, name] of weightNames.entries()) {
    const bufPtr = wasm._malloc(metadata[name].size);
    wasm.HEAPU8.set(metadata[name].bytes, bufPtr);
    metadata[name].bytes = null;
    wasm._set_buf(i, bufPtr);
  }

  return {
    run: (input0,start_pos) => {
      const inputPtr0 = wasm._malloc(4);
      const outputPtr0 = wasm._malloc(4);
      wasm.HEAPU8.set(input0, inputPtr0);
      wasm._net(outputPtr0, inputPtr0, start_pos);
      const output0 = wasm.HEAPU8.slice(outputPtr0, outputPtr0 + 4);
      wasm._free(outputPtr0);
      wasm._free(inputPtr0);
      return [output0];
    },
    wasm: wasm
  }
}

var q6k_to_f32 = async function() {

  const wasm = await q6k_to_f32Module();
  

  return {
    run: (input0) => {
      const inputPtr0 = wasm._malloc(430080);
      const outputPtr0 = wasm._malloc(2097152);
      wasm.HEAPU8.set(input0, inputPtr0);
      wasm._net(outputPtr0, inputPtr0);
      const output0 = wasm.HEAPU8.slice(outputPtr0, outputPtr0 + 2097152);
      wasm._free(outputPtr0);
      wasm._free(inputPtr0);
      return [output0];
    },
    wasm: wasm
  }
}

var q6k_to_int8_2048_2048 = async function() {

  const wasm = await q6k_to_int8_2048_2048Module();
  

  return {
    run: (input0) => {
      const inputPtr0 = wasm._malloc(3440640);
      const outputPtr0 = wasm._malloc(8192);
      const outputPtr1 = wasm._malloc(4194304);
      wasm.HEAPU8.set(input0, inputPtr0);
      wasm._net(outputPtr0, outputPtr1, inputPtr0);
      const output0 = wasm.HEAPU8.slice(outputPtr0, outputPtr0 + 8192);
      const output1 = wasm.HEAPU8.slice(outputPtr1, outputPtr1 + 4194304);
      wasm._free(outputPtr0);
      wasm._free(outputPtr1);
      wasm._free(inputPtr0);
      return [output0, output1];
    },
    wasm: wasm
  }
}

var q6k_to_int8_512_2048 = async function() {

  const wasm = await q6k_to_int8_512_2048Module();
  

  return {
    run: (input0) => {
      const inputPtr0 = wasm._malloc(860160);
      const outputPtr0 = wasm._malloc(2048);
      const outputPtr1 = wasm._malloc(1048576);
      wasm.HEAPU8.set(input0, inputPtr0);
      wasm._net(outputPtr0, outputPtr1, inputPtr0);
      const output0 = wasm.HEAPU8.slice(outputPtr0, outputPtr0 + 2048);
      const output1 = wasm.HEAPU8.slice(outputPtr1, outputPtr1 + 1048576);
      wasm._free(outputPtr0);
      wasm._free(outputPtr1);
      wasm._free(inputPtr0);
      return [output0, output1];
    },
    wasm: wasm
  }
}

var q6k_to_int8_8192_2048 = async function() {

  const wasm = await q6k_to_int8_8192_2048Module();
  

  return {
    run: (input0) => {
      const inputPtr0 = wasm._malloc(13762560);
      const outputPtr0 = wasm._malloc(32768);
      const outputPtr1 = wasm._malloc(16777216);
      wasm.HEAPU8.set(input0, inputPtr0);
      wasm._net(outputPtr0, outputPtr1, inputPtr0);
      const output0 = wasm.HEAPU8.slice(outputPtr0, outputPtr0 + 32768);
      const output1 = wasm.HEAPU8.slice(outputPtr1, outputPtr1 + 16777216);
      wasm._free(outputPtr0);
      wasm._free(outputPtr1);
      wasm._free(inputPtr0);
      return [output0, output1];
    },
    wasm: wasm
  }
}

var q6k_to_int8_2048_8192 = async function() {

  const wasm = await q6k_to_int8_2048_8192Module();
  

  return {
    run: (input0) => {
      const inputPtr0 = wasm._malloc(13762560);
      const outputPtr0 = wasm._malloc(8192);
      const outputPtr1 = wasm._malloc(16777216);
      wasm.HEAPU8.set(input0, inputPtr0);
      wasm._net(outputPtr0, outputPtr1, inputPtr0);
      const output0 = wasm.HEAPU8.slice(outputPtr0, outputPtr0 + 8192);
      const output1 = wasm.HEAPU8.slice(outputPtr1, outputPtr1 + 16777216);
      wasm._free(outputPtr0);
      wasm._free(outputPtr1);
      wasm._free(inputPtr0);
      return [output0, output1];
    },
    wasm: wasm
  }
}
export {transformer, q6k_to_f32, q6k_to_int8_2048_2048, q6k_to_int8_512_2048, q6k_to_int8_8192_2048, q6k_to_int8_2048_8192};