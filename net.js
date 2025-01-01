
    window.MODEL_BASE_URL= ".";
    const getTensorMetadata = (safetensorBuffer) => {
      const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
      const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
      return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
    };

  const getTensorBuffer = (safetensorParts, tensorMetadata, key) => {
    let selectedPart = 0;
    let counter = 0;
    let partStartOffsets = [595639560, 1149320456, 1703001352, 2298641672, 2852322568, 3406003464, 4943273224, 5994459400, 5994983688];
    let correctedOffsets = tensorMetadata.data_offsets;
    let prev_offset = 0;

    for (let start of partStartOffsets) {
      prev_offset = (counter == 0) ? 0 : partStartOffsets[counter-1];

      if (tensorMetadata.data_offsets[0] < start) {
        selectedPart = counter;
        correctedOffsets = [correctedOffsets[0]-prev_offset, correctedOffsets[1]-prev_offset];
        break;
      }

      counter++;
    }

    return safetensorParts[selectedPart].subarray(...correctedOffsets);
  }

  const getWeight = (safetensors, key) => {
    let uint8Data = getTensorBuffer(safetensors, getTensorMetadata(safetensors[0])[key], key);
    return new Float32Array(uint8Data.buffer, uint8Data.byteOffset, uint8Data.byteLength / Float32Array.BYTES_PER_ELEMENT);
  }

  const createEmptyBuf = (device, size) => {
      return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  };

  const createWeightBuf = (device, size, data) => {
    const buf = device.createBuffer({ mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE });
    new Uint8Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  };

  const addComputePass = (device, commandEncoder, pipeline, bufs, workgroup) => {
    const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: bufs.map((buffer, index) => ({ binding: index, resource: { buffer } }))});
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...workgroup);
    passEncoder.end();
  };
    var transformer = function() {

    const E_8_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<i32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var val0 = data1[alu0];
  var alu1 = (alu0+1);
  var val1 = data1[alu1];
  var alu2 = (alu0+2);
  var val2 = data1[alu2];
  var alu3 = (alu0+3);
  var val3 = data1[alu3];
  data0[alu1] = select(0.0f,(f32(-inf(1.0))),((val1!=1000000)!=true));
  data0[alu2] = select(0.0f,(f32(-inf(1.0))),((val2!=1000000)!=true));
  data0[alu3] = select(0.0f,(f32(-inf(1.0))),((val3!=1000000)!=true));
  data0[alu0] = select(0.0f,(f32(-inf(1.0))),((val0!=1000000)!=true));
}`;

const E_128_16_8_16_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 128 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data1[(lidx0+(gidx1<<3))];
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var alu2 = (alu0+(gidx1*4086)+(lidx0*4093)+alu1);
  var alu3 = (alu0+(gidx1<<13)+(lidx0<<10)+alu1);
  data0[(alu3+2)] = (val0+select(0.0f,(f32(-inf(1.0))),(((alu2+1)%4094)<2047)));
  data0[(alu3+3)] = (val0+select(0.0f,(f32(-inf(1.0))),(((alu2+2)%4094)<2047)));
  data0[alu3] = (val0+select(0.0f,(f32(-inf(1.0))),(((alu2+4093)%4094)<2047)));
  data0[(alu3+1)] = (val0+select(0.0f,(f32(-inf(1.0))),((alu2%4094)<2047)));
}`;

const E_n7 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<u32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var val0 = data0[0];
  data0[0] = (val0+1u);
}`;

const E_8_32_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<atomic<u32>>;
@group(0) @binding(1)var<storage,read_write>data1:array<i32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var val0 = data1[alu0];
  var alu1 = (alu0+1);
  var val1 = data1[alu1];
  var alu2 = (alu0+2);
  var val2 = data1[alu2];
  var alu3 = (alu0+3);
  var val3 = data1[alu3];
  var val4 = atomicLoad(&data0[(alu1>>2)]);
  var val5 = atomicLoad(&data0[(alu2>>2)]);
  var val6 = atomicLoad(&data0[(alu3>>2)]);
  var val7 = atomicLoad(&data0[(alu0>>2)]);
  var alu4 = (((u32(alu1))&3u)<<3u);
  var alu5 = (((u32(alu2))&3u)<<3u);
  var alu6 = (((u32(alu3))&3u)<<3u);
  var alu7 = (((u32(alu0))&3u)<<3u);
  atomicAnd(&data0[(alu1>>2)],((255u<<alu4)^4294967295u));
  atomicAdd(&data0[(alu1>>2)],((u32(((val1!=1000000)!=true)))<<alu4));
  atomicAnd(&data0[(alu2>>2)],((255u<<alu5)^4294967295u));
  atomicAdd(&data0[(alu2>>2)],((u32(((val2!=1000000)!=true)))<<alu5));
  atomicAnd(&data0[(alu3>>2)],((255u<<alu6)^4294967295u));
  atomicAdd(&data0[(alu3>>2)],((u32(((val3!=1000000)!=true)))<<alu6));
  atomicAnd(&data0[(alu0>>2)],((255u<<alu7)^4294967295u));
  atomicAdd(&data0[(alu0>>2)],((u32(((val0!=1000000)!=true)))<<alu7));
}`;

const r_1024_16_64 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<i32, 16>;
@group(0) @binding(0)var<storage,read_write>data0:array<i32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx0+(lidx0<<6)+-959);
  var alu1 = -select(alu0,0,(alu0<0));
  temp0[lidx0] = -select(alu1,-64,(alu1<-64));
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0;
    for (var ridx0 = 0; ridx0 < 16; ridx0++) {
      var val0 = temp0[ridx0];
      acc0 = (acc0+val0);
    }
    data0[gidx0] = (acc0+-1);
  }
}`;

const E_n5 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<u32>;
@group(0) @binding(1)var<storage,read_write>data1:array<u32>;
@group(0) @binding(2)var<storage,read_write>data2:array<u32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var val0 = data1[0];
  var val1 = data2[0];
  var val2 = data2[1];
  var alu0 = (val0+val2);
  var alu1 = (val0+val1+alu0);
  var alu2 = (val1^val2^466688986u);
  var alu3 = ((alu1+1u)^((val0<<13u)+(val2<<13u)+((alu0+1u)>>19u)+8192u));
  var alu4 = (alu1+alu3);
  var alu5 = ((alu4+1u)^((alu3<<15u)+(alu3>>17u)));
  var alu6 = (alu4+alu5);
  var alu7 = ((alu6+1u)^((alu5<<26u)+(alu5>>6u)));
  var alu8 = (alu6+alu7);
  var alu9 = ((alu8+1u)^((alu7<<6u)+(alu7>>26u)));
  var alu10 = (alu9+alu2);
  var alu11 = (val2+alu8+alu10);
  var alu12 = ((alu11+2u)^((alu9<<17u)+(alu2<<17u)+((alu10+1u)>>15u)+131072u));
  var alu13 = (alu11+alu12);
  var alu14 = ((alu13+2u)^((alu12<<29u)+(alu12>>3u)));
  var alu15 = (alu13+alu14);
  var alu16 = ((alu15+2u)^((alu14<<16u)+(alu14>>16u)));
  var alu17 = (alu15+alu16);
  var alu18 = ((alu17+2u)^((alu16<<24u)+(alu16>>8u)));
  var alu19 = (val1+alu18);
  var alu20 = (alu19+alu17+alu2);
  var alu21 = ((alu20+4u)^((val1<<13u)+(alu18<<13u)+((alu19+2u)>>19u)+16384u));
  var alu22 = (alu20+alu21);
  var alu23 = ((alu22+4u)^((alu21<<15u)+(alu21>>17u)));
  var alu24 = (alu22+alu23);
  var alu25 = ((alu24+4u)^((alu23<<26u)+(alu23>>6u)));
  var alu26 = (alu24+alu25);
  var alu27 = ((alu26+4u)^((alu25<<6u)+(alu25>>26u)));
  var alu28 = (val2+alu27);
  var alu29 = (val1+alu26+alu28);
  var alu30 = ((alu29+7u)^((val2<<17u)+(alu27<<17u)+((alu28+3u)>>15u)+393216u));
  var alu31 = (alu29+alu30);
  var alu32 = ((alu31+7u)^((alu30<<29u)+(alu30>>3u)));
  var alu33 = (alu31+alu32);
  var alu34 = ((alu33+7u)^((alu32<<16u)+(alu32>>16u)));
  var alu35 = (alu33+alu34);
  var alu36 = ((alu35+7u)^((alu34<<24u)+(alu34>>8u)));
  var alu37 = (alu36+alu2);
  var alu38 = (val2+alu35+alu37);
  var alu39 = ((alu38+11u)^((alu36<<13u)+(alu2<<13u)+((alu37+4u)>>19u)+32768u));
  var alu40 = (alu38+alu39);
  var alu41 = ((alu40+11u)^((alu39<<15u)+(alu39>>17u)));
  var alu42 = (alu40+alu41);
  var alu43 = ((alu42+11u)^((alu41<<26u)+(alu41>>6u)));
  data0[0] = (val1+((alu42+alu43+11u)^((alu43<<6u)+(alu43>>26u)))+5u);
}`;

const E_8_32_4n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<i32>;
@group(0) @binding(1)var<storage,read_write>data1:array<atomic<u32>>;
@group(0) @binding(2)var<storage,read_write>data2:array<i32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var val0 = data2[alu0];
  var alu1 = (alu0+1);
  var val1 = data2[alu1];
  var alu2 = (alu0+2);
  var val2 = data2[alu2];
  var alu3 = (alu0+3);
  var val3 = data2[alu3];
  var val4 = atomicLoad(&data1[(alu1>>2)]);
  var val5 = atomicLoad(&data1[(alu2>>2)]);
  var val6 = atomicLoad(&data1[(alu3>>2)]);
  var val7 = atomicLoad(&data1[(alu0>>2)]);
  data0[alu1] = select(val1,0,(bool(((val4>>(((u32(alu1))&3u)<<3u))&255u))));
  data0[alu2] = select(val2,0,(bool(((val5>>(((u32(alu2))&3u)<<3u))&255u))));
  data0[alu3] = select(val3,0,(bool(((val6>>(((u32(alu3))&3u)<<3u))&255u))));
  data0[alu0] = select(val0,0,(bool(((val7>>(((u32(alu0))&3u)<<3u))&255u))));
}`;

const r_4_256_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<i32, 16>;
@group(0) @binding(0)var<storage,read_write>data0:array<i32>;
@group(0) @binding(1)var<storage,read_write>data1:array<atomic<u32>>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx1<<8);
  var acc0 = 0;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var alu1 = (gidx0+(lidx0<<4)+ridx0);
    var alu2 = (alu1+alu0+-255);
    var val0 = select(0u, atomicLoad(&data1[(alu2>>2)]), ((alu1<255)!=true));
    acc0 = (acc0+(i32((bool(((val0>>(((u32(alu2))&3u)<<3u))&255u))))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[(gidx0+alu0)] = acc1;
  }
}`;

const E_n6 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<u32>;
@group(0) @binding(2)var<storage,read_write>data2:array<u32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var val0 = data1[0];
  var val1 = data2[0];
  var val2 = data2[1];
  var precast0 = 1.0f;
  var alu0 = (val0+val2);
  var alu1 = (val0+val1+alu0);
  var alu2 = (val1^val2^466688986u);
  var alu3 = ((alu1+1u)^((val0<<13u)+(val2<<13u)+((alu0+1u)>>19u)+8192u));
  var alu4 = (alu1+alu3);
  var alu5 = ((alu4+1u)^((alu3<<15u)+(alu3>>17u)));
  var alu6 = (alu4+alu5);
  var alu7 = ((alu6+1u)^((alu5<<26u)+(alu5>>6u)));
  var alu8 = (alu6+alu7);
  var alu9 = ((alu8+1u)^((alu7<<6u)+(alu7>>26u)));
  var alu10 = (alu9+alu2);
  var alu11 = (val2+alu8+alu10);
  var alu12 = ((alu11+2u)^((alu9<<17u)+(alu2<<17u)+((alu10+1u)>>15u)+131072u));
  var alu13 = (alu11+alu12);
  var alu14 = ((alu13+2u)^((alu12<<29u)+(alu12>>3u)));
  var alu15 = (alu13+alu14);
  var alu16 = ((alu15+2u)^((alu14<<16u)+(alu14>>16u)));
  var alu17 = (alu15+alu16);
  var alu18 = ((alu17+2u)^((alu16<<24u)+(alu16>>8u)));
  var alu19 = (val1+alu18);
  var alu20 = (alu19+alu17+alu2);
  var alu21 = ((alu20+4u)^((val1<<13u)+(alu18<<13u)+((alu19+2u)>>19u)+16384u));
  var alu22 = (alu20+alu21);
  var alu23 = ((alu22+4u)^((alu21<<15u)+(alu21>>17u)));
  var alu24 = (alu22+alu23);
  var alu25 = ((alu24+4u)^((alu23<<26u)+(alu23>>6u)));
  var alu26 = (alu24+alu25);
  var alu27 = ((alu26+4u)^((alu25<<6u)+(alu25>>26u)));
  var alu28 = (val2+alu27);
  var alu29 = (val1+alu26+alu28);
  var alu30 = ((alu29+7u)^((val2<<17u)+(alu27<<17u)+((alu28+3u)>>15u)+393216u));
  var alu31 = (alu29+alu30);
  var alu32 = ((alu31+7u)^((alu30<<29u)+(alu30>>3u)));
  var alu33 = (alu31+alu32);
  var alu34 = ((alu33+7u)^((alu32<<16u)+(alu32>>16u)));
  var alu35 = (alu33+alu34);
  var alu36 = ((alu35+7u)^((alu34<<24u)+(alu34>>8u)));
  var alu37 = (alu36+alu2);
  var alu38 = (val2+alu35+alu37);
  var alu39 = ((alu38+11u)^((alu36<<13u)+(alu2<<13u)+((alu37+4u)>>19u)+32768u));
  var alu40 = (alu38+alu39);
  var alu41 = ((alu40+11u)^((alu39<<15u)+(alu39>>17u)));
  var alu42 = (alu40+alu41);
  var precast1 = (bitcast<u32>(precast0)|((alu42+((alu42+11u)^((alu41<<26u)+(alu41>>6u)))+alu2+11u)>>9u));
  data0[0] = (bitcast<f32>(precast1)+-1.0f);
}`;

const r_32_32_8_16_32064_4_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<i32>;
@group(0) @binding(2)var<storage,read_write>data2:array<i32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = (lidx1<<2);
  var alu2 = ((gidx1<<5)+(lidx0<<2));
  var val0 = data2[alu2];
  var val1 = data2[(alu2+1)];
  var val2 = data2[(alu2+2)];
  var val3 = data2[(alu2+3)];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 32064; ridx0++) {
    var alu3 = (ridx0<<2);
    var val4 = data1[alu3];
    var val5 = data1[(alu3+1)];
    var val6 = data1[(alu3+2)];
    var val7 = data1[(alu3+3)];
    var alu4 = (alu0+alu1+(ridx0<<13));
    var val8 = data3[alu4];
    var val9 = data3[(alu4+1)];
    var val10 = data3[(alu4+2)];
    var val11 = data3[(alu4+3)];
    var val12 = data3[(alu4+2048)];
    var val13 = data3[(alu4+2049)];
    var val14 = data3[(alu4+2050)];
    var val15 = data3[(alu4+2051)];
    var val16 = data3[(alu4+4096)];
    var val17 = data3[(alu4+4097)];
    var val18 = data3[(alu4+4098)];
    var val19 = data3[(alu4+4099)];
    var val20 = data3[(alu4+6144)];
    var val21 = data3[(alu4+6145)];
    var val22 = data3[(alu4+6146)];
    var val23 = data3[(alu4+6147)];
    var cast0 = (f32(((val5!=val1)!=true)));
    var cast1 = (f32(((val5!=val2)!=true)));
    var cast2 = (f32(((val5!=val3)!=true)));
    var cast3 = (f32(((val5!=val0)!=true)));
    var cast4 = (f32(((val6!=val1)!=true)));
    var cast5 = (f32(((val6!=val2)!=true)));
    var cast6 = (f32(((val6!=val3)!=true)));
    var cast7 = (f32(((val6!=val0)!=true)));
    var cast8 = (f32(((val7!=val1)!=true)));
    var cast9 = (f32(((val7!=val2)!=true)));
    var cast10 = (f32(((val7!=val3)!=true)));
    var cast11 = (f32(((val7!=val0)!=true)));
    var cast12 = (f32(((val4!=val1)!=true)));
    var cast13 = (f32(((val4!=val2)!=true)));
    var cast14 = (f32(((val4!=val3)!=true)));
    var cast15 = (f32(((val4!=val0)!=true)));
    acc0 = (acc0+(cast3*val12)+(cast15*val8)+(cast7*val16)+(cast11*val20));
    acc1 = (acc1+(cast0*val12)+(cast12*val8)+(cast4*val16)+(cast8*val20));
    acc2 = (acc2+(cast1*val12)+(cast13*val8)+(cast5*val16)+(cast9*val20));
    acc3 = (acc3+(cast2*val12)+(cast14*val8)+(cast6*val16)+(cast10*val20));
    acc4 = (acc4+(cast3*val13)+(cast15*val9)+(cast7*val17)+(cast11*val21));
    acc5 = (acc5+(cast0*val13)+(cast12*val9)+(cast4*val17)+(cast8*val21));
    acc6 = (acc6+(cast1*val13)+(cast13*val9)+(cast5*val17)+(cast9*val21));
    acc7 = (acc7+(cast2*val13)+(cast14*val9)+(cast6*val17)+(cast10*val21));
    acc8 = (acc8+(cast3*val14)+(cast15*val10)+(cast7*val18)+(cast11*val22));
    acc9 = (acc9+(cast0*val14)+(cast12*val10)+(cast4*val18)+(cast8*val22));
    acc10 = (acc10+(cast1*val14)+(cast13*val10)+(cast5*val18)+(cast9*val22));
    acc11 = (acc11+(cast2*val14)+(cast14*val10)+(cast6*val18)+(cast10*val22));
    acc12 = (acc12+(cast3*val15)+(cast15*val11)+(cast7*val19)+(cast11*val23));
    acc13 = (acc13+(cast0*val15)+(cast12*val11)+(cast4*val19)+(cast8*val23));
    acc14 = (acc14+(cast1*val15)+(cast13*val11)+(cast5*val19)+(cast9*val23));
    acc15 = (acc15+(cast2*val15)+(cast14*val11)+(cast6*val19)+(cast10*val23));
  }
  var alu22 = (alu0+(gidx1<<16)+(lidx0<<13)+alu1);
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc4;
  data0[(alu22+2)] = acc8;
  data0[(alu22+3)] = acc12;
  data0[(alu22+2048)] = acc1;
  data0[(alu22+2049)] = acc5;
  data0[(alu22+2050)] = acc9;
  data0[(alu22+2051)] = acc13;
  data0[(alu22+4096)] = acc2;
  data0[(alu22+4097)] = acc6;
  data0[(alu22+4098)] = acc10;
  data0[(alu22+4099)] = acc14;
  data0[(alu22+6144)] = acc3;
  data0[(alu22+6145)] = acc7;
  data0[(alu22+6146)] = acc11;
  data0[(alu22+6147)] = acc15;
}`;

const r_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<i32>;
@group(0) @binding(1)var<storage,read_write>data1:array<i32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var val0 = data1[255];
  var val1 = data1[511];
  var val2 = data1[767];
  data0[0] = 0;
  data0[1] = val0;
  var alu2 = (val0+val1);
  data0[2] = alu2;
  data0[3] = (val2+alu2);
}`;

const r_64_4_8_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (lidx0<<5);
  var alu1 = (lidx0+(gidx0<<4));
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu2 = (lidx1+(gidx0<<15)+(lidx0<<11)+(ridx0<<3));
    var val0 = data1[alu2];
    var val1 = data1[(alu2+8192)];
    var val2 = data1[(alu2+16384)];
    var val3 = data1[(alu2+24576)];
    acc0 = (acc0+(val0*val0));
    acc1 = (acc1+(val1*val1));
    acc2 = (acc2+(val2*val2));
    acc3 = (acc3+(val3*val3));
  }
  var alu8 = (alu0+(lidx1<<2));
  temp0[alu8] = acc0;
  temp0[(alu8+1)] = acc1;
  temp0[(alu8+2)] = acc2;
  temp0[(alu8+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu15 = (alu0+(ridx1<<2));
      var val4 = temp0[(alu15+1)];
      var val5 = temp0[(alu15+2)];
      var val6 = temp0[(alu15+3)];
      var val7 = temp0[alu15];
      acc4 = (acc4+val7);
      acc5 = (acc5+val4);
      acc6 = (acc6+val5);
      acc7 = (acc7+val6);
    }
    data0[alu1] = sqrt((1/((acc4*0.00048828125f)+1e-05f)));
    data0[(alu1+4)] = sqrt((1/((acc5*0.00048828125f)+1e-05f)));
    data0[(alu1+8)] = sqrt((1/((acc6*0.00048828125f)+1e-05f)));
    data0[(alu1+12)] = sqrt((1/((acc7*0.00048828125f)+1e-05f)));
  }
}`;

const r_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<i32, 256>;
@group(0) @binding(0)var<storage,read_write>data0:array<i32>;
@group(0) @binding(1)var<storage,read_write>data1:array<i32>;
@group(0) @binding(2)var<storage,read_write>data2:array<i32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data2[(lidx0>>6)];
  var acc0 = 0;
  for (var ridx0 = 0; ridx0 < 4; ridx0++) {
    var val1 = data1[((lidx0<<2)+ridx0)];
    acc0 = (acc0+(i32(((bool((val1+val0)))!=true))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0;
    for (var ridx1 = 0; ridx1 < 256; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[0] = (select(0,1024,(acc1<1))+acc1+-1);
  }
}`;

const E_32_32_8_16_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = ((gidx1<<5)+(lidx0<<2));
  var val0 = data2[alu1];
  var val1 = data2[(alu1+1)];
  var val2 = data2[(alu1+2)];
  var val3 = data2[(alu1+3)];
  var alu2 = (lidx1<<2);
  var alu3 = (alu0+(gidx1<<16)+(lidx0<<13)+alu2);
  var val4 = data1[alu3];
  var alu4 = (alu3+1);
  var val5 = data1[alu4];
  var alu5 = (alu3+2);
  var val6 = data1[alu5];
  var alu6 = (alu3+3);
  var val7 = data1[alu6];
  var alu7 = (alu3+2048);
  var val8 = data1[alu7];
  var alu8 = (alu3+2049);
  var val9 = data1[alu8];
  var alu9 = (alu3+2050);
  var val10 = data1[alu9];
  var alu10 = (alu3+2051);
  var val11 = data1[alu10];
  var alu11 = (alu3+4096);
  var val12 = data1[alu11];
  var alu12 = (alu3+4097);
  var val13 = data1[alu12];
  var alu13 = (alu3+4098);
  var val14 = data1[alu13];
  var alu14 = (alu3+4099);
  var val15 = data1[alu14];
  var alu15 = (alu3+6144);
  var val16 = data1[alu15];
  var alu16 = (alu3+6145);
  var val17 = data1[alu16];
  var alu17 = (alu3+6146);
  var val18 = data1[alu17];
  var alu18 = (alu3+6147);
  var val19 = data1[alu18];
  var alu19 = (alu0+alu2);
  var val20 = data3[alu19];
  var val21 = data3[(alu19+1)];
  var val22 = data3[(alu19+2)];
  var val23 = data3[(alu19+3)];
  data0[alu4] = (val21*val5*val0);
  data0[alu8] = (val21*val9*val1);
  data0[alu12] = (val21*val13*val2);
  data0[alu16] = (val21*val17*val3);
  data0[alu5] = (val22*val6*val0);
  data0[alu9] = (val22*val10*val1);
  data0[alu13] = (val22*val14*val2);
  data0[alu17] = (val22*val18*val3);
  data0[alu6] = (val23*val7*val0);
  data0[alu10] = (val23*val11*val1);
  data0[alu14] = (val23*val15*val2);
  data0[alu18] = (val23*val19*val3);
  data0[alu7] = (val20*val8*val1);
  data0[alu11] = (val20*val12*val2);
  data0[alu15] = (val20*val16*val3);
  data0[alu3] = (val20*val4*val0);
}`;

const r_32_8_8_16_512_4_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0<<17)+(lidx1<<13)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+2048)];
    var val5 = data2[(alu1+2049)];
    var val6 = data2[(alu1+2050)];
    var val7 = data2[(alu1+2051)];
    var val8 = data2[(alu1+4096)];
    var val9 = data2[(alu1+4097)];
    var val10 = data2[(alu1+4098)];
    var val11 = data2[(alu1+4099)];
    var val12 = data2[(alu1+6144)];
    var val13 = data2[(alu1+6145)];
    var val14 = data2[(alu1+6146)];
    var val15 = data2[(alu1+6147)];
    var alu2 = ((gidx1<<16)+(lidx0<<13)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+2048)];
    var val21 = data1[(alu2+2049)];
    var val22 = data1[(alu2+2050)];
    var val23 = data1[(alu2+2051)];
    var val24 = data1[(alu2+4096)];
    var val25 = data1[(alu2+4097)];
    var val26 = data1[(alu2+4098)];
    var val27 = data1[(alu2+4099)];
    var val28 = data1[(alu2+6144)];
    var val29 = data1[(alu2+6145)];
    var val30 = data1[(alu2+6146)];
    var val31 = data1[(alu2+6147)];
    acc0 = (acc0+(val17*val1)+(val16*val0)+(val18*val2)+(val19*val3));
    acc1 = (acc1+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3));
    acc2 = (acc2+(val24*val0)+(val25*val1)+(val26*val2)+(val27*val3));
    acc3 = (acc3+(val28*val0)+(val29*val1)+(val30*val2)+(val31*val3));
    acc4 = (acc4+(val17*val5)+(val16*val4)+(val18*val6)+(val19*val7));
    acc5 = (acc5+(val20*val4)+(val21*val5)+(val22*val6)+(val23*val7));
    acc6 = (acc6+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7));
    acc7 = (acc7+(val28*val4)+(val29*val5)+(val30*val6)+(val31*val7));
    acc8 = (acc8+(val17*val9)+(val16*val8)+(val18*val10)+(val19*val11));
    acc9 = (acc9+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc10 = (acc10+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc11 = (acc11+(val28*val8)+(val29*val9)+(val30*val10)+(val31*val11));
    acc12 = (acc12+(val17*val13)+(val16*val12)+(val18*val14)+(val19*val15));
    acc13 = (acc13+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc14 = (acc14+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc15 = (acc15+(val28*val12)+(val29*val13)+(val30*val14)+(val31*val15));
  }
  var alu20 = ((gidx0<<6)+(gidx1<<14)+(lidx0<<11)+(lidx1<<2));
  data0[alu20] = acc0;
  data0[(alu20+1)] = acc4;
  data0[(alu20+2)] = acc8;
  data0[(alu20+3)] = acc12;
  data0[(alu20+512)] = acc1;
  data0[(alu20+513)] = acc5;
  data0[(alu20+514)] = acc9;
  data0[(alu20+515)] = acc13;
  data0[(alu20+1024)] = acc2;
  data0[(alu20+1025)] = acc6;
  data0[(alu20+1026)] = acc10;
  data0[(alu20+1027)] = acc14;
  data0[(alu20+1536)] = acc3;
  data0[(alu20+1537)] = acc7;
  data0[(alu20+1538)] = acc11;
  data0[(alu20+1539)] = acc15;
}`;

const r_32_32_8_16_512_4_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx1<<16);
  var alu1 = (lidx0<<13);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu2 = (ridx0<<2);
    var alu3 = ((gidx0<<17)+(lidx1<<13)+alu2);
    var val0 = data2[alu3];
    var val1 = data2[(alu3+1)];
    var val2 = data2[(alu3+2)];
    var val3 = data2[(alu3+3)];
    var val4 = data2[(alu3+2048)];
    var val5 = data2[(alu3+2049)];
    var val6 = data2[(alu3+2050)];
    var val7 = data2[(alu3+2051)];
    var val8 = data2[(alu3+4096)];
    var val9 = data2[(alu3+4097)];
    var val10 = data2[(alu3+4098)];
    var val11 = data2[(alu3+4099)];
    var val12 = data2[(alu3+6144)];
    var val13 = data2[(alu3+6145)];
    var val14 = data2[(alu3+6146)];
    var val15 = data2[(alu3+6147)];
    var alu4 = (alu0+alu1+alu2);
    var val16 = data1[alu4];
    var val17 = data1[(alu4+1)];
    var val18 = data1[(alu4+2)];
    var val19 = data1[(alu4+3)];
    var val20 = data1[(alu4+2048)];
    var val21 = data1[(alu4+2049)];
    var val22 = data1[(alu4+2050)];
    var val23 = data1[(alu4+2051)];
    var val24 = data1[(alu4+4096)];
    var val25 = data1[(alu4+4097)];
    var val26 = data1[(alu4+4098)];
    var val27 = data1[(alu4+4099)];
    var val28 = data1[(alu4+6144)];
    var val29 = data1[(alu4+6145)];
    var val30 = data1[(alu4+6146)];
    var val31 = data1[(alu4+6147)];
    acc0 = (acc0+(val17*val1)+(val16*val0)+(val18*val2)+(val19*val3));
    acc1 = (acc1+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3));
    acc2 = (acc2+(val24*val0)+(val25*val1)+(val26*val2)+(val27*val3));
    acc3 = (acc3+(val28*val0)+(val29*val1)+(val30*val2)+(val31*val3));
    acc4 = (acc4+(val17*val5)+(val16*val4)+(val18*val6)+(val19*val7));
    acc5 = (acc5+(val20*val4)+(val21*val5)+(val22*val6)+(val23*val7));
    acc6 = (acc6+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7));
    acc7 = (acc7+(val28*val4)+(val29*val5)+(val30*val6)+(val31*val7));
    acc8 = (acc8+(val17*val9)+(val16*val8)+(val18*val10)+(val19*val11));
    acc9 = (acc9+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc10 = (acc10+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc11 = (acc11+(val28*val8)+(val29*val9)+(val30*val10)+(val31*val11));
    acc12 = (acc12+(val17*val13)+(val16*val12)+(val18*val14)+(val19*val15));
    acc13 = (acc13+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc14 = (acc14+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc15 = (acc15+(val28*val12)+(val29*val13)+(val30*val14)+(val31*val15));
  }
  var alu22 = ((gidx0<<6)+alu0+alu1+(lidx1<<2));
  data0[alu22] = acc0;
  data0[(alu22+1)] = acc4;
  data0[(alu22+2)] = acc8;
  data0[(alu22+3)] = acc12;
  data0[(alu22+2048)] = acc1;
  data0[(alu22+2049)] = acc5;
  data0[(alu22+2050)] = acc9;
  data0[(alu22+2051)] = acc13;
  data0[(alu22+4096)] = acc2;
  data0[(alu22+4097)] = acc6;
  data0[(alu22+4098)] = acc10;
  data0[(alu22+4099)] = acc14;
  data0[(alu22+6144)] = acc3;
  data0[(alu22+6145)] = acc7;
  data0[(alu22+6146)] = acc11;
  data0[(alu22+6147)] = acc15;
}`;

const E_1024_2_8_16_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 1024 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<5);
  var alu1 = (lidx1<<1);
  var alu2 = (alu0+(gidx1<<11)+(lidx0<<8)+alu1);
  var val0 = data1[alu2];
  var alu3 = (alu2+1);
  var val1 = data1[alu3];
  var alu4 = (alu2+64);
  var val2 = data1[alu4];
  var alu5 = (alu2+65);
  var val3 = data1[alu5];
  var alu6 = (alu2+128);
  var val4 = data1[alu6];
  var alu7 = (alu2+129);
  var val5 = data1[alu7];
  var alu8 = (alu2+192);
  var val6 = data1[alu8];
  var alu9 = (alu2+193);
  var val7 = data1[alu9];
  var alu10 = (alu0+(gidx1<<6)+alu1);
  var val8 = data2[alu10];
  var val9 = data2[(alu10+1)];
  data0[alu5] = ((val2*val9)+(val3*val8));
  data0[alu7] = ((val4*val9)+(val5*val8));
  data0[alu9] = ((val6*val9)+(val7*val8));
  data0[alu3] = ((val1*val8)+(val0*val9));
  data0[alu4] = ((val2*val8)-(val3*val9));
  data0[alu6] = ((val4*val8)-(val5*val9));
  data0[alu8] = ((val6*val8)-(val7*val9));
  data0[alu2] = ((val0*val8)-(val1*val9));
}`;

const E_256_2_4_2_16_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(4,2,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 256 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 2 */
  var lidx2 = i32(lindex.z); /* 16 */
  var alu0 = (gidx0<<5);
  var alu1 = (lidx2<<1);
  var alu2 = (alu0+(gidx1<<11)+(lidx0<<9)+(lidx1<<8)+alu1);
  var val0 = data1[alu2];
  var alu3 = (alu2+1);
  var val1 = data1[alu3];
  var alu4 = (alu2+64);
  var val2 = data1[alu4];
  var alu5 = (alu2+65);
  var val3 = data1[alu5];
  var alu6 = (alu2+128);
  var val4 = data1[alu6];
  var alu7 = (alu2+129);
  var val5 = data1[alu7];
  var alu8 = (alu2+192);
  var val6 = data1[alu8];
  var alu9 = (alu2+193);
  var val7 = data1[alu9];
  var alu10 = (alu0+(gidx1<<8)+(lidx0<<6)+alu1);
  var val8 = data2[alu10];
  var val9 = data2[(alu10+1)];
  data0[alu5] = ((val2*val9)+(val3*val8));
  data0[alu7] = ((val4*val9)+(val5*val8));
  data0[alu9] = ((val6*val9)+(val7*val8));
  data0[alu3] = ((val1*val8)+(val0*val9));
  data0[alu4] = ((val2*val8)-(val3*val9));
  data0[alu6] = ((val4*val8)-(val5*val9));
  data0[alu8] = ((val6*val8)-(val7*val9));
  data0[alu2] = ((val0*val8)-(val1*val9));
}`;

const r_8_32_16_8_16_64_4_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 32 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  var acc16 = 0.0f;
  var acc17 = 0.0f;
  var acc18 = 0.0f;
  var acc19 = 0.0f;
  var acc20 = 0.0f;
  var acc21 = 0.0f;
  var acc22 = 0.0f;
  var acc23 = 0.0f;
  var acc24 = 0.0f;
  var acc25 = 0.0f;
  var acc26 = 0.0f;
  var acc27 = 0.0f;
  var acc28 = 0.0f;
  var acc29 = 0.0f;
  var acc30 = 0.0f;
  var acc31 = 0.0f;
  var acc32 = 0.0f;
  var acc33 = 0.0f;
  var acc34 = 0.0f;
  var acc35 = 0.0f;
  var acc36 = 0.0f;
  var acc37 = 0.0f;
  var acc38 = 0.0f;
  var acc39 = 0.0f;
  var acc40 = 0.0f;
  var acc41 = 0.0f;
  var acc42 = 0.0f;
  var acc43 = 0.0f;
  var acc44 = 0.0f;
  var acc45 = 0.0f;
  var acc46 = 0.0f;
  var acc47 = 0.0f;
  var acc48 = 0.0f;
  var acc49 = 0.0f;
  var acc50 = 0.0f;
  var acc51 = 0.0f;
  var acc52 = 0.0f;
  var acc53 = 0.0f;
  var acc54 = 0.0f;
  var acc55 = 0.0f;
  var acc56 = 0.0f;
  var acc57 = 0.0f;
  var acc58 = 0.0f;
  var acc59 = 0.0f;
  var acc60 = 0.0f;
  var acc61 = 0.0f;
  var acc62 = 0.0f;
  var acc63 = 0.0f;
  for (var ridx0 = 0; ridx0 < 64; ridx0++) {
    var alu0 = ((gidx0<<15)+(gidx2<<6)+(lidx1<<11)+ridx0);
    var val0 = data2[alu0];
    var val1 = data2[(alu0+512)];
    var val2 = data2[(alu0+1024)];
    var val3 = data2[(alu0+1536)];
    var alu1 = ((gidx1<<16)+(gidx2<<8)+(lidx0<<13)+ridx0);
    var val4 = data1[alu1];
    var val5 = data1[(alu1+64)];
    var val6 = data1[(alu1+128)];
    var val7 = data1[(alu1+192)];
    var val8 = data1[(alu1+2048)];
    var val9 = data1[(alu1+2112)];
    var val10 = data1[(alu1+2176)];
    var val11 = data1[(alu1+2240)];
    var val12 = data1[(alu1+4096)];
    var val13 = data1[(alu1+4160)];
    var val14 = data1[(alu1+4224)];
    var val15 = data1[(alu1+4288)];
    var val16 = data1[(alu1+6144)];
    var val17 = data1[(alu1+6208)];
    var val18 = data1[(alu1+6272)];
    var val19 = data1[(alu1+6336)];
    acc0 = (acc0+(val4*val0));
    acc1 = (acc1+(val5*val0));
    acc2 = (acc2+(val6*val0));
    acc3 = (acc3+(val7*val0));
    acc4 = (acc4+(val8*val0));
    acc5 = (acc5+(val9*val0));
    acc6 = (acc6+(val10*val0));
    acc7 = (acc7+(val11*val0));
    acc8 = (acc8+(val12*val0));
    acc9 = (acc9+(val13*val0));
    acc10 = (acc10+(val14*val0));
    acc11 = (acc11+(val15*val0));
    acc12 = (acc12+(val16*val0));
    acc13 = (acc13+(val17*val0));
    acc14 = (acc14+(val18*val0));
    acc15 = (acc15+(val19*val0));
    acc16 = (acc16+(val4*val1));
    acc17 = (acc17+(val5*val1));
    acc18 = (acc18+(val6*val1));
    acc19 = (acc19+(val7*val1));
    acc20 = (acc20+(val8*val1));
    acc21 = (acc21+(val9*val1));
    acc22 = (acc22+(val10*val1));
    acc23 = (acc23+(val11*val1));
    acc24 = (acc24+(val12*val1));
    acc25 = (acc25+(val13*val1));
    acc26 = (acc26+(val14*val1));
    acc27 = (acc27+(val15*val1));
    acc28 = (acc28+(val16*val1));
    acc29 = (acc29+(val17*val1));
    acc30 = (acc30+(val18*val1));
    acc31 = (acc31+(val19*val1));
    acc32 = (acc32+(val4*val2));
    acc33 = (acc33+(val5*val2));
    acc34 = (acc34+(val6*val2));
    acc35 = (acc35+(val7*val2));
    acc36 = (acc36+(val8*val2));
    acc37 = (acc37+(val9*val2));
    acc38 = (acc38+(val10*val2));
    acc39 = (acc39+(val11*val2));
    acc40 = (acc40+(val12*val2));
    acc41 = (acc41+(val13*val2));
    acc42 = (acc42+(val14*val2));
    acc43 = (acc43+(val15*val2));
    acc44 = (acc44+(val16*val2));
    acc45 = (acc45+(val17*val2));
    acc46 = (acc46+(val18*val2));
    acc47 = (acc47+(val19*val2));
    acc48 = (acc48+(val4*val3));
    acc49 = (acc49+(val5*val3));
    acc50 = (acc50+(val6*val3));
    acc51 = (acc51+(val7*val3));
    acc52 = (acc52+(val8*val3));
    acc53 = (acc53+(val9*val3));
    acc54 = (acc54+(val10*val3));
    acc55 = (acc55+(val11*val3));
    acc56 = (acc56+(val12*val3));
    acc57 = (acc57+(val13*val3));
    acc58 = (acc58+(val14*val3));
    acc59 = (acc59+(val15*val3));
    acc60 = (acc60+(val16*val3));
    acc61 = (acc61+(val17*val3));
    acc62 = (acc62+(val18*val3));
    acc63 = (acc63+(val19*val3));
  }
  var alu67 = (gidx0<<6);
  var alu68 = (gidx1<<15);
  var alu69 = (lidx0<<12);
  var alu70 = (lidx1<<2);
  var alu71 = (alu67+alu68+alu69+alu70);
  var val20 = data3[alu71];
  var val21 = data3[(alu71+1)];
  var val22 = data3[(alu71+2)];
  var val23 = data3[(alu71+3)];
  var val24 = data3[(alu71+1024)];
  var val25 = data3[(alu71+1025)];
  var val26 = data3[(alu71+1026)];
  var val27 = data3[(alu71+1027)];
  var val28 = data3[(alu71+2048)];
  var val29 = data3[(alu71+2049)];
  var val30 = data3[(alu71+2050)];
  var val31 = data3[(alu71+2051)];
  var val32 = data3[(alu71+3072)];
  var val33 = data3[(alu71+3073)];
  var val34 = data3[(alu71+3074)];
  var val35 = data3[(alu71+3075)];
  var alu72 = (alu68+(gidx2<<22)+alu67+alu69+alu70);
  data0[alu72] = (val20+(acc0*0.125f));
  data0[(alu72+1)] = (val21+(acc16*0.125f));
  data0[(alu72+2)] = (val22+(acc32*0.125f));
  data0[(alu72+3)] = (val23+(acc48*0.125f));
  data0[(alu72+1024)] = (val24+(acc4*0.125f));
  data0[(alu72+1025)] = (val25+(acc20*0.125f));
  data0[(alu72+1026)] = (val26+(acc36*0.125f));
  data0[(alu72+1027)] = (val27+(acc52*0.125f));
  data0[(alu72+2048)] = (val28+(acc8*0.125f));
  data0[(alu72+2049)] = (val29+(acc24*0.125f));
  data0[(alu72+2050)] = (val30+(acc40*0.125f));
  data0[(alu72+2051)] = (val31+(acc56*0.125f));
  data0[(alu72+3072)] = (val32+(acc12*0.125f));
  data0[(alu72+3073)] = (val33+(acc28*0.125f));
  data0[(alu72+3074)] = (val34+(acc44*0.125f));
  data0[(alu72+3075)] = (val35+(acc60*0.125f));
  data0[(alu72+1048576)] = (val20+(acc1*0.125f));
  data0[(alu72+1048577)] = (val21+(acc17*0.125f));
  data0[(alu72+1048578)] = (val22+(acc33*0.125f));
  data0[(alu72+1048579)] = (val23+(acc49*0.125f));
  data0[(alu72+1049600)] = (val24+(acc5*0.125f));
  data0[(alu72+1049601)] = (val25+(acc21*0.125f));
  data0[(alu72+1049602)] = (val26+(acc37*0.125f));
  data0[(alu72+1049603)] = (val27+(acc53*0.125f));
  data0[(alu72+1050624)] = (val28+(acc9*0.125f));
  data0[(alu72+1050625)] = (val29+(acc25*0.125f));
  data0[(alu72+1050626)] = (val30+(acc41*0.125f));
  data0[(alu72+1050627)] = (val31+(acc57*0.125f));
  data0[(alu72+1051648)] = (val32+(acc13*0.125f));
  data0[(alu72+1051649)] = (val33+(acc29*0.125f));
  data0[(alu72+1051650)] = (val34+(acc45*0.125f));
  data0[(alu72+1051651)] = (val35+(acc61*0.125f));
  data0[(alu72+2097152)] = (val20+(acc2*0.125f));
  data0[(alu72+2097153)] = (val21+(acc18*0.125f));
  data0[(alu72+2097154)] = (val22+(acc34*0.125f));
  data0[(alu72+2097155)] = (val23+(acc50*0.125f));
  data0[(alu72+2098176)] = (val24+(acc6*0.125f));
  data0[(alu72+2098177)] = (val25+(acc22*0.125f));
  data0[(alu72+2098178)] = (val26+(acc38*0.125f));
  data0[(alu72+2098179)] = (val27+(acc54*0.125f));
  data0[(alu72+2099200)] = (val28+(acc10*0.125f));
  data0[(alu72+2099201)] = (val29+(acc26*0.125f));
  data0[(alu72+2099202)] = (val30+(acc42*0.125f));
  data0[(alu72+2099203)] = (val31+(acc58*0.125f));
  data0[(alu72+2100224)] = (val32+(acc14*0.125f));
  data0[(alu72+2100225)] = (val33+(acc30*0.125f));
  data0[(alu72+2100226)] = (val34+(acc46*0.125f));
  data0[(alu72+2100227)] = (val35+(acc62*0.125f));
  data0[(alu72+3145728)] = (val20+(acc3*0.125f));
  data0[(alu72+3145729)] = (val21+(acc19*0.125f));
  data0[(alu72+3145730)] = (val22+(acc35*0.125f));
  data0[(alu72+3145731)] = (val23+(acc51*0.125f));
  data0[(alu72+3146752)] = (val24+(acc7*0.125f));
  data0[(alu72+3146753)] = (val25+(acc23*0.125f));
  data0[(alu72+3146754)] = (val26+(acc39*0.125f));
  data0[(alu72+3146755)] = (val27+(acc55*0.125f));
  data0[(alu72+3147776)] = (val28+(acc11*0.125f));
  data0[(alu72+3147777)] = (val29+(acc27*0.125f));
  data0[(alu72+3147778)] = (val30+(acc43*0.125f));
  data0[(alu72+3147779)] = (val31+(acc59*0.125f));
  data0[(alu72+3148800)] = (val32+(acc15*0.125f));
  data0[(alu72+3148801)] = (val33+(acc31*0.125f));
  data0[(alu72+3148802)] = (val34+(acc47*0.125f));
  data0[(alu72+3148803)] = (val35+(acc63*0.125f));
}`;

const r_1024_32_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = (f32(-inf(1.0)));
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu0 = ((gidx0<<15)+(lidx0<<10)+(ridx0<<2));
    var val0 = data1[alu0];
    var val1 = data1[(alu0+1)];
    var val2 = data1[(alu0+2)];
    var val3 = data1[(alu0+3)];
    var alu1 = select(val1,val0,(val1<val0));
    var alu2 = select(val2,alu1,(val2<alu1));
    var alu3 = select(val3,alu2,(val3<alu2));
    acc0 = select(acc0,alu3,(acc0<alu3));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_256_32_256_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var alu1 = (alu0+1);
  var alu2 = (alu0+2);
  var alu3 = (alu0+3);
  var val0 = data2[alu1];
  var val1 = data2[alu2];
  var val2 = data2[alu3];
  var val3 = data2[alu0];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu4 = ((gidx0<<17)+(lidx0<<12)+(ridx0<<2));
    var val4 = data1[alu4];
    var val5 = data1[(alu4+1)];
    var val6 = data1[(alu4+2)];
    var val7 = data1[(alu4+3)];
    var val8 = data1[(alu4+1024)];
    var val9 = data1[(alu4+1025)];
    var val10 = data1[(alu4+1026)];
    var val11 = data1[(alu4+1027)];
    var val12 = data1[(alu4+2048)];
    var val13 = data1[(alu4+2049)];
    var val14 = data1[(alu4+2050)];
    var val15 = data1[(alu4+2051)];
    var val16 = data1[(alu4+3072)];
    var val17 = data1[(alu4+3073)];
    var val18 = data1[(alu4+3074)];
    var val19 = data1[(alu4+3075)];
    acc0 = (acc0+exp2(((val7-val3)*1.4426950408889634f))+exp2(((val6-val3)*1.4426950408889634f))+exp2(((val5-val3)*1.4426950408889634f))+exp2(((val4-val3)*1.4426950408889634f)));
    acc1 = (acc1+exp2(((val11-val0)*1.4426950408889634f))+exp2(((val10-val0)*1.4426950408889634f))+exp2(((val8-val0)*1.4426950408889634f))+exp2(((val9-val0)*1.4426950408889634f)));
    acc2 = (acc2+exp2(((val15-val1)*1.4426950408889634f))+exp2(((val14-val1)*1.4426950408889634f))+exp2(((val12-val1)*1.4426950408889634f))+exp2(((val13-val1)*1.4426950408889634f)));
    acc3 = (acc3+exp2(((val19-val2)*1.4426950408889634f))+exp2(((val18-val2)*1.4426950408889634f))+exp2(((val16-val2)*1.4426950408889634f))+exp2(((val17-val2)*1.4426950408889634f)));
  }
  data0[alu1] = acc1;
  data0[alu2] = acc2;
  data0[alu3] = acc3;
  data0[alu0] = acc0;
}`;

const E_4096_16_8_16_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 4096 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx0+(gidx1<<3));
  var val0 = data2[alu0];
  var val1 = data3[alu0];
  var alu1 = ((gidx0<<6)+(gidx1<<13)+(lidx0<<10)+(lidx1<<2));
  var val2 = data1[alu1];
  var alu2 = (alu1+1);
  var val3 = data1[alu2];
  var alu3 = (alu1+2);
  var val4 = data1[alu3];
  var alu4 = (alu1+3);
  var val5 = data1[alu4];
  var alu5 = (1/val1);
  data0[alu2] = (exp2(((val3-val0)*1.4426950408889634f))*alu5);
  data0[alu3] = (exp2(((val4-val0)*1.4426950408889634f))*alu5);
  data0[alu4] = (exp2(((val5-val0)*1.4426950408889634f))*alu5);
  data0[alu1] = (exp2(((val2-val0)*1.4426950408889634f))*alu5);
}`;

const r_32_32_8_16_256_4_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx1<<2);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu1 = ((gidx0<<15)+(gidx1<<20)+(lidx0<<12)+(ridx0<<2));
    var val0 = data1[alu1];
    var val1 = data1[(alu1+1)];
    var val2 = data1[(alu1+2)];
    var val3 = data1[(alu1+3)];
    var val4 = data1[(alu1+1024)];
    var val5 = data1[(alu1+1025)];
    var val6 = data1[(alu1+1026)];
    var val7 = data1[(alu1+1027)];
    var val8 = data1[(alu1+2048)];
    var val9 = data1[(alu1+2049)];
    var val10 = data1[(alu1+2050)];
    var val11 = data1[(alu1+2051)];
    var val12 = data1[(alu1+3072)];
    var val13 = data1[(alu1+3073)];
    var val14 = data1[(alu1+3074)];
    var val15 = data1[(alu1+3075)];
    var alu2 = (((gidx1>>2)<<6)+(ridx0<<11)+alu0);
    var val16 = data2[alu2];
    var val17 = data2[(alu2+1)];
    var val18 = data2[(alu2+2)];
    var val19 = data2[(alu2+3)];
    var val20 = data2[(alu2+512)];
    var val21 = data2[(alu2+513)];
    var val22 = data2[(alu2+514)];
    var val23 = data2[(alu2+515)];
    var val24 = data2[(alu2+1024)];
    var val25 = data2[(alu2+1025)];
    var val26 = data2[(alu2+1026)];
    var val27 = data2[(alu2+1027)];
    var val28 = data2[(alu2+1536)];
    var val29 = data2[(alu2+1537)];
    var val30 = data2[(alu2+1538)];
    var val31 = data2[(alu2+1539)];
    acc0 = (acc0+(val1*val20)+(val0*val16)+(val2*val24)+(val3*val28));
    acc1 = (acc1+(val4*val16)+(val5*val20)+(val6*val24)+(val7*val28));
    acc2 = (acc2+(val8*val16)+(val9*val20)+(val10*val24)+(val11*val28));
    acc3 = (acc3+(val12*val16)+(val13*val20)+(val14*val24)+(val15*val28));
    acc4 = (acc4+(val1*val21)+(val0*val17)+(val2*val25)+(val3*val29));
    acc5 = (acc5+(val4*val17)+(val5*val21)+(val6*val25)+(val7*val29));
    acc6 = (acc6+(val8*val17)+(val9*val21)+(val10*val25)+(val11*val29));
    acc7 = (acc7+(val12*val17)+(val13*val21)+(val14*val25)+(val15*val29));
    acc8 = (acc8+(val1*val22)+(val0*val18)+(val2*val26)+(val3*val30));
    acc9 = (acc9+(val4*val18)+(val5*val22)+(val6*val26)+(val7*val30));
    acc10 = (acc10+(val8*val18)+(val9*val22)+(val10*val26)+(val11*val30));
    acc11 = (acc11+(val12*val18)+(val13*val22)+(val14*val26)+(val15*val30));
    acc12 = (acc12+(val1*val23)+(val0*val19)+(val2*val27)+(val3*val31));
    acc13 = (acc13+(val4*val19)+(val5*val23)+(val6*val27)+(val7*val31));
    acc14 = (acc14+(val8*val19)+(val9*val23)+(val10*val27)+(val11*val31));
    acc15 = (acc15+(val12*val19)+(val13*val23)+(val14*val27)+(val15*val31));
  }
  var alu20 = ((gidx0<<11)+(gidx1<<16)+(lidx0<<8)+alu0);
  data0[alu20] = acc0;
  data0[(alu20+1)] = acc4;
  data0[(alu20+2)] = acc8;
  data0[(alu20+3)] = acc12;
  data0[(alu20+64)] = acc1;
  data0[(alu20+65)] = acc5;
  data0[(alu20+66)] = acc9;
  data0[(alu20+67)] = acc13;
  data0[(alu20+128)] = acc2;
  data0[(alu20+129)] = acc6;
  data0[(alu20+130)] = acc10;
  data0[(alu20+131)] = acc14;
  data0[(alu20+192)] = acc3;
  data0[(alu20+193)] = acc7;
  data0[(alu20+194)] = acc11;
  data0[(alu20+195)] = acc15;
}`;

const r_32_32_8_16_512_4_4_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu0 = ((gidx0<<17)+(lidx1<<13)+(ridx0<<2));
    var val0 = data3[alu0];
    var val1 = data3[(alu0+1)];
    var val2 = data3[(alu0+2)];
    var val3 = data3[(alu0+3)];
    var val4 = data3[(alu0+2048)];
    var val5 = data3[(alu0+2049)];
    var val6 = data3[(alu0+2050)];
    var val7 = data3[(alu0+2051)];
    var val8 = data3[(alu0+4096)];
    var val9 = data3[(alu0+4097)];
    var val10 = data3[(alu0+4098)];
    var val11 = data3[(alu0+4099)];
    var val12 = data3[(alu0+6144)];
    var val13 = data3[(alu0+6145)];
    var val14 = data3[(alu0+6146)];
    var val15 = data3[(alu0+6147)];
    var alu1 = ((gidx1<<11)+(lidx0<<8)+((ridx0>>4)<<16)+((ridx0&15)<<2));
    var val16 = data2[alu1];
    var val17 = data2[(alu1+1)];
    var val18 = data2[(alu1+2)];
    var val19 = data2[(alu1+3)];
    var val20 = data2[(alu1+64)];
    var val21 = data2[(alu1+65)];
    var val22 = data2[(alu1+66)];
    var val23 = data2[(alu1+67)];
    var val24 = data2[(alu1+128)];
    var val25 = data2[(alu1+129)];
    var val26 = data2[(alu1+130)];
    var val27 = data2[(alu1+131)];
    var val28 = data2[(alu1+192)];
    var val29 = data2[(alu1+193)];
    var val30 = data2[(alu1+194)];
    var val31 = data2[(alu1+195)];
    acc0 = (acc0+(val17*val1)+(val16*val0)+(val18*val2)+(val19*val3));
    acc1 = (acc1+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3));
    acc2 = (acc2+(val24*val0)+(val25*val1)+(val26*val2)+(val27*val3));
    acc3 = (acc3+(val28*val0)+(val29*val1)+(val30*val2)+(val31*val3));
    acc4 = (acc4+(val17*val5)+(val16*val4)+(val18*val6)+(val19*val7));
    acc5 = (acc5+(val20*val4)+(val21*val5)+(val22*val6)+(val23*val7));
    acc6 = (acc6+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7));
    acc7 = (acc7+(val28*val4)+(val29*val5)+(val30*val6)+(val31*val7));
    acc8 = (acc8+(val17*val9)+(val16*val8)+(val18*val10)+(val19*val11));
    acc9 = (acc9+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc10 = (acc10+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc11 = (acc11+(val28*val8)+(val29*val9)+(val30*val10)+(val31*val11));
    acc12 = (acc12+(val17*val13)+(val16*val12)+(val18*val14)+(val19*val15));
    acc13 = (acc13+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc14 = (acc14+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc15 = (acc15+(val28*val12)+(val29*val13)+(val30*val14)+(val31*val15));
  }
  var alu19 = ((gidx0<<6)+(gidx1<<16)+(lidx0<<13)+(lidx1<<2));
  var val32 = data1[alu19];
  var alu20 = (alu19+1);
  var val33 = data1[alu20];
  var alu21 = (alu19+2);
  var val34 = data1[alu21];
  var alu22 = (alu19+3);
  var val35 = data1[alu22];
  var alu23 = (alu19+2048);
  var val36 = data1[alu23];
  var alu24 = (alu19+2049);
  var val37 = data1[alu24];
  var alu25 = (alu19+2050);
  var val38 = data1[alu25];
  var alu26 = (alu19+2051);
  var val39 = data1[alu26];
  var alu27 = (alu19+4096);
  var val40 = data1[alu27];
  var alu28 = (alu19+4097);
  var val41 = data1[alu28];
  var alu29 = (alu19+4098);
  var val42 = data1[alu29];
  var alu30 = (alu19+4099);
  var val43 = data1[alu30];
  var alu31 = (alu19+6144);
  var val44 = data1[alu31];
  var alu32 = (alu19+6145);
  var val45 = data1[alu32];
  var alu33 = (alu19+6146);
  var val46 = data1[alu33];
  var alu34 = (alu19+6147);
  var val47 = data1[alu34];
  data0[alu20] = (val33+acc4);
  data0[alu21] = (val34+acc8);
  data0[alu22] = (val35+acc12);
  data0[alu23] = (val36+acc1);
  data0[alu24] = (val37+acc5);
  data0[alu25] = (val38+acc9);
  data0[alu26] = (val39+acc13);
  data0[alu27] = (val40+acc2);
  data0[alu28] = (val41+acc6);
  data0[alu29] = (val42+acc10);
  data0[alu30] = (val43+acc14);
  data0[alu31] = (val44+acc3);
  data0[alu32] = (val45+acc7);
  data0[alu33] = (val46+acc11);
  data0[alu34] = (val47+acc15);
  data0[alu19] = (val32+acc0);
}`;

const r_32_128_8_16_512_4_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0<<17)+(lidx1<<13)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+2048)];
    var val5 = data2[(alu1+2049)];
    var val6 = data2[(alu1+2050)];
    var val7 = data2[(alu1+2051)];
    var val8 = data2[(alu1+4096)];
    var val9 = data2[(alu1+4097)];
    var val10 = data2[(alu1+4098)];
    var val11 = data2[(alu1+4099)];
    var val12 = data2[(alu1+6144)];
    var val13 = data2[(alu1+6145)];
    var val14 = data2[(alu1+6146)];
    var val15 = data2[(alu1+6147)];
    var alu2 = ((gidx1<<16)+(lidx0<<13)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+2048)];
    var val21 = data1[(alu2+2049)];
    var val22 = data1[(alu2+2050)];
    var val23 = data1[(alu2+2051)];
    var val24 = data1[(alu2+4096)];
    var val25 = data1[(alu2+4097)];
    var val26 = data1[(alu2+4098)];
    var val27 = data1[(alu2+4099)];
    var val28 = data1[(alu2+6144)];
    var val29 = data1[(alu2+6145)];
    var val30 = data1[(alu2+6146)];
    var val31 = data1[(alu2+6147)];
    acc0 = (acc0+(val17*val1)+(val16*val0)+(val18*val2)+(val19*val3));
    acc1 = (acc1+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3));
    acc2 = (acc2+(val24*val0)+(val25*val1)+(val26*val2)+(val27*val3));
    acc3 = (acc3+(val28*val0)+(val29*val1)+(val30*val2)+(val31*val3));
    acc4 = (acc4+(val17*val5)+(val16*val4)+(val18*val6)+(val19*val7));
    acc5 = (acc5+(val20*val4)+(val21*val5)+(val22*val6)+(val23*val7));
    acc6 = (acc6+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7));
    acc7 = (acc7+(val28*val4)+(val29*val5)+(val30*val6)+(val31*val7));
    acc8 = (acc8+(val17*val9)+(val16*val8)+(val18*val10)+(val19*val11));
    acc9 = (acc9+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc10 = (acc10+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc11 = (acc11+(val28*val8)+(val29*val9)+(val30*val10)+(val31*val11));
    acc12 = (acc12+(val17*val13)+(val16*val12)+(val18*val14)+(val19*val15));
    acc13 = (acc13+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc14 = (acc14+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc15 = (acc15+(val28*val12)+(val29*val13)+(val30*val14)+(val31*val15));
  }
  var alu20 = ((gidx0<<6)+(gidx1<<18)+(lidx0<<15)+(lidx1<<2));
  data0[alu20] = acc0;
  data0[(alu20+1)] = acc4;
  data0[(alu20+2)] = acc8;
  data0[(alu20+3)] = acc12;
  data0[(alu20+8192)] = acc1;
  data0[(alu20+8193)] = acc5;
  data0[(alu20+8194)] = acc9;
  data0[(alu20+8195)] = acc13;
  data0[(alu20+16384)] = acc2;
  data0[(alu20+16385)] = acc6;
  data0[(alu20+16386)] = acc10;
  data0[(alu20+16387)] = acc14;
  data0[(alu20+24576)] = acc3;
  data0[(alu20+24577)] = acc7;
  data0[(alu20+24578)] = acc11;
  data0[(alu20+24579)] = acc15;
}`;

const r_32_128_8_16_512_4_4_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0<<17)+(lidx1<<13)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+2048)];
    var val5 = data2[(alu1+2049)];
    var val6 = data2[(alu1+2050)];
    var val7 = data2[(alu1+2051)];
    var val8 = data2[(alu1+4096)];
    var val9 = data2[(alu1+4097)];
    var val10 = data2[(alu1+4098)];
    var val11 = data2[(alu1+4099)];
    var val12 = data2[(alu1+6144)];
    var val13 = data2[(alu1+6145)];
    var val14 = data2[(alu1+6146)];
    var val15 = data2[(alu1+6147)];
    var alu2 = ((gidx1<<16)+(lidx0<<13)+alu0);
    var val16 = data1[alu2];
    var val17 = data1[(alu2+1)];
    var val18 = data1[(alu2+2)];
    var val19 = data1[(alu2+3)];
    var val20 = data1[(alu2+2048)];
    var val21 = data1[(alu2+2049)];
    var val22 = data1[(alu2+2050)];
    var val23 = data1[(alu2+2051)];
    var val24 = data1[(alu2+4096)];
    var val25 = data1[(alu2+4097)];
    var val26 = data1[(alu2+4098)];
    var val27 = data1[(alu2+4099)];
    var val28 = data1[(alu2+6144)];
    var val29 = data1[(alu2+6145)];
    var val30 = data1[(alu2+6146)];
    var val31 = data1[(alu2+6147)];
    acc0 = (acc0+(val17*val1)+(val16*val0)+(val18*val2)+(val19*val3));
    acc1 = (acc1+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3));
    acc2 = (acc2+(val24*val0)+(val25*val1)+(val26*val2)+(val27*val3));
    acc3 = (acc3+(val28*val0)+(val29*val1)+(val30*val2)+(val31*val3));
    acc4 = (acc4+(val17*val5)+(val16*val4)+(val18*val6)+(val19*val7));
    acc5 = (acc5+(val20*val4)+(val21*val5)+(val22*val6)+(val23*val7));
    acc6 = (acc6+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7));
    acc7 = (acc7+(val28*val4)+(val29*val5)+(val30*val6)+(val31*val7));
    acc8 = (acc8+(val17*val9)+(val16*val8)+(val18*val10)+(val19*val11));
    acc9 = (acc9+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc10 = (acc10+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc11 = (acc11+(val28*val8)+(val29*val9)+(val30*val10)+(val31*val11));
    acc12 = (acc12+(val17*val13)+(val16*val12)+(val18*val14)+(val19*val15));
    acc13 = (acc13+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc14 = (acc14+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc15 = (acc15+(val28*val12)+(val29*val13)+(val30*val14)+(val31*val15));
  }
  var alu20 = ((gidx0<<6)+(gidx1<<18)+(lidx0<<15)+(lidx1<<2));
  var val32 = data3[alu20];
  var alu21 = (alu20+1);
  var val33 = data3[alu21];
  var alu22 = (alu20+2);
  var val34 = data3[alu22];
  var alu23 = (alu20+3);
  var val35 = data3[alu23];
  var alu24 = (alu20+8192);
  var val36 = data3[alu24];
  var alu25 = (alu20+8193);
  var val37 = data3[alu25];
  var alu26 = (alu20+8194);
  var val38 = data3[alu26];
  var alu27 = (alu20+8195);
  var val39 = data3[alu27];
  var alu28 = (alu20+16384);
  var val40 = data3[alu28];
  var alu29 = (alu20+16385);
  var val41 = data3[alu29];
  var alu30 = (alu20+16386);
  var val42 = data3[alu30];
  var alu31 = (alu20+16387);
  var val43 = data3[alu31];
  var alu32 = (alu20+24576);
  var val44 = data3[alu32];
  var alu33 = (alu20+24577);
  var val45 = data3[alu33];
  var alu34 = (alu20+24578);
  var val46 = data3[alu34];
  var alu35 = (alu20+24579);
  var val47 = data3[alu35];
  data0[alu20] = (val32*(1/(exp2((acc0*-1.4426950408889634f))+1.0f))*acc0);
  data0[alu24] = (val36*(1/(exp2((acc1*-1.4426950408889634f))+1.0f))*acc1);
  data0[alu28] = (val40*(1/(exp2((acc2*-1.4426950408889634f))+1.0f))*acc2);
  data0[alu32] = (val44*(1/(exp2((acc3*-1.4426950408889634f))+1.0f))*acc3);
  data0[alu21] = (val33*(1/(exp2((acc4*-1.4426950408889634f))+1.0f))*acc4);
  data0[alu25] = (val37*(1/(exp2((acc5*-1.4426950408889634f))+1.0f))*acc5);
  data0[alu29] = (val41*(1/(exp2((acc6*-1.4426950408889634f))+1.0f))*acc6);
  data0[alu33] = (val45*(1/(exp2((acc7*-1.4426950408889634f))+1.0f))*acc7);
  data0[alu22] = (val34*(1/(exp2((acc8*-1.4426950408889634f))+1.0f))*acc8);
  data0[alu26] = (val38*(1/(exp2((acc9*-1.4426950408889634f))+1.0f))*acc9);
  data0[alu30] = (val42*(1/(exp2((acc10*-1.4426950408889634f))+1.0f))*acc10);
  data0[alu34] = (val46*(1/(exp2((acc11*-1.4426950408889634f))+1.0f))*acc11);
  data0[alu23] = (val35*(1/(exp2((acc12*-1.4426950408889634f))+1.0f))*acc12);
  data0[alu27] = (val39*(1/(exp2((acc13*-1.4426950408889634f))+1.0f))*acc13);
  data0[alu31] = (val43*(1/(exp2((acc14*-1.4426950408889634f))+1.0f))*acc14);
  data0[alu35] = (val47*(1/(exp2((acc15*-1.4426950408889634f))+1.0f))*acc15);
}`;

const r_32_32_8_16_2048_4_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 2048; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0<<19)+(lidx1<<15)+alu0);
    var val0 = data3[alu1];
    var val1 = data3[(alu1+1)];
    var val2 = data3[(alu1+2)];
    var val3 = data3[(alu1+3)];
    var val4 = data3[(alu1+8192)];
    var val5 = data3[(alu1+8193)];
    var val6 = data3[(alu1+8194)];
    var val7 = data3[(alu1+8195)];
    var val8 = data3[(alu1+16384)];
    var val9 = data3[(alu1+16385)];
    var val10 = data3[(alu1+16386)];
    var val11 = data3[(alu1+16387)];
    var val12 = data3[(alu1+24576)];
    var val13 = data3[(alu1+24577)];
    var val14 = data3[(alu1+24578)];
    var val15 = data3[(alu1+24579)];
    var alu2 = ((gidx1<<18)+(lidx0<<15)+alu0);
    var val16 = data2[alu2];
    var val17 = data2[(alu2+1)];
    var val18 = data2[(alu2+2)];
    var val19 = data2[(alu2+3)];
    var val20 = data2[(alu2+8192)];
    var val21 = data2[(alu2+8193)];
    var val22 = data2[(alu2+8194)];
    var val23 = data2[(alu2+8195)];
    var val24 = data2[(alu2+16384)];
    var val25 = data2[(alu2+16385)];
    var val26 = data2[(alu2+16386)];
    var val27 = data2[(alu2+16387)];
    var val28 = data2[(alu2+24576)];
    var val29 = data2[(alu2+24577)];
    var val30 = data2[(alu2+24578)];
    var val31 = data2[(alu2+24579)];
    acc0 = (acc0+(val17*val1)+(val16*val0)+(val18*val2)+(val19*val3));
    acc1 = (acc1+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3));
    acc2 = (acc2+(val24*val0)+(val25*val1)+(val26*val2)+(val27*val3));
    acc3 = (acc3+(val28*val0)+(val29*val1)+(val30*val2)+(val31*val3));
    acc4 = (acc4+(val17*val5)+(val16*val4)+(val18*val6)+(val19*val7));
    acc5 = (acc5+(val20*val4)+(val21*val5)+(val22*val6)+(val23*val7));
    acc6 = (acc6+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7));
    acc7 = (acc7+(val28*val4)+(val29*val5)+(val30*val6)+(val31*val7));
    acc8 = (acc8+(val17*val9)+(val16*val8)+(val18*val10)+(val19*val11));
    acc9 = (acc9+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc10 = (acc10+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc11 = (acc11+(val28*val8)+(val29*val9)+(val30*val10)+(val31*val11));
    acc12 = (acc12+(val17*val13)+(val16*val12)+(val18*val14)+(val19*val15));
    acc13 = (acc13+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc14 = (acc14+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc15 = (acc15+(val28*val12)+(val29*val13)+(val30*val14)+(val31*val15));
  }
  var alu20 = ((gidx0<<6)+(gidx1<<16)+(lidx0<<13)+(lidx1<<2));
  var val32 = data1[alu20];
  var alu21 = (alu20+1);
  var val33 = data1[alu21];
  var alu22 = (alu20+2);
  var val34 = data1[alu22];
  var alu23 = (alu20+3);
  var val35 = data1[alu23];
  var alu24 = (alu20+2048);
  var val36 = data1[alu24];
  var alu25 = (alu20+2049);
  var val37 = data1[alu25];
  var alu26 = (alu20+2050);
  var val38 = data1[alu26];
  var alu27 = (alu20+2051);
  var val39 = data1[alu27];
  var alu28 = (alu20+4096);
  var val40 = data1[alu28];
  var alu29 = (alu20+4097);
  var val41 = data1[alu29];
  var alu30 = (alu20+4098);
  var val42 = data1[alu30];
  var alu31 = (alu20+4099);
  var val43 = data1[alu31];
  var alu32 = (alu20+6144);
  var val44 = data1[alu32];
  var alu33 = (alu20+6145);
  var val45 = data1[alu33];
  var alu34 = (alu20+6146);
  var val46 = data1[alu34];
  var alu35 = (alu20+6147);
  var val47 = data1[alu35];
  data0[alu21] = (val33+acc4);
  data0[alu22] = (val34+acc8);
  data0[alu23] = (val35+acc12);
  data0[alu24] = (val36+acc1);
  data0[alu25] = (val37+acc5);
  data0[alu26] = (val38+acc9);
  data0[alu27] = (val39+acc13);
  data0[alu28] = (val40+acc2);
  data0[alu29] = (val41+acc6);
  data0[alu30] = (val42+acc10);
  data0[alu31] = (val43+acc14);
  data0[alu32] = (val44+acc3);
  data0[alu33] = (val45+acc7);
  data0[alu34] = (val46+acc11);
  data0[alu35] = (val47+acc15);
  data0[alu20] = (val32+acc0);
}`;

const r_1024_16_128 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<atomic<u32>>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = atomicLoad(&data1[(gidx0>>2)]);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 128; ridx0++) {
    var val1 = data2[((gidx0<<11)+(lidx0<<7)+ridx0)];
    acc0 = (acc0+select((val1*val1),0.0f,(bool(((val0>>(((u32(gidx0))&3u)<<3u))&255u)))));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val2 = temp0[ridx1];
      acc1 = (acc1+val2);
    }
    data0[gidx0] = sqrt((1/((acc1*0.00048828125f)+1e-05f)));
  }
}`;

const E_32_32_8_16_4_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<atomic<u32>>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@group(0) @binding(4)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<6);
  var alu1 = ((gidx1<<5)+(lidx0<<2));
  var val0 = data3[alu1];
  var alu2 = (alu1+1);
  var val1 = data3[alu2];
  var alu3 = (alu1+2);
  var val2 = data3[alu3];
  var alu4 = (alu1+3);
  var val3 = data3[alu4];
  var alu5 = (lidx1<<2);
  var alu6 = (alu0+(gidx1<<16)+(lidx0<<13)+alu5);
  var val4 = data2[alu6];
  var alu7 = (alu6+1);
  var val5 = data2[alu7];
  var alu8 = (alu6+2);
  var val6 = data2[alu8];
  var alu9 = (alu6+3);
  var val7 = data2[alu9];
  var alu10 = (alu6+2048);
  var val8 = data2[alu10];
  var alu11 = (alu6+2049);
  var val9 = data2[alu11];
  var alu12 = (alu6+2050);
  var val10 = data2[alu12];
  var alu13 = (alu6+2051);
  var val11 = data2[alu13];
  var alu14 = (alu6+4096);
  var val12 = data2[alu14];
  var alu15 = (alu6+4097);
  var val13 = data2[alu15];
  var alu16 = (alu6+4098);
  var val14 = data2[alu16];
  var alu17 = (alu6+4099);
  var val15 = data2[alu17];
  var alu18 = (alu6+6144);
  var val16 = data2[alu18];
  var alu19 = (alu6+6145);
  var val17 = data2[alu19];
  var alu20 = (alu6+6146);
  var val18 = data2[alu20];
  var alu21 = (alu6+6147);
  var val19 = data2[alu21];
  var alu22 = (alu0+alu5);
  var val20 = data4[alu22];
  var val21 = data4[(alu22+1)];
  var val22 = data4[(alu22+2)];
  var val23 = data4[(alu22+3)];
  var val24 = atomicLoad(&data1[(alu2>>2)]);
  var val25 = atomicLoad(&data1[(alu3>>2)]);
  var val26 = atomicLoad(&data1[(alu4>>2)]);
  var val27 = atomicLoad(&data1[(alu1>>2)]);
  var cast0 = (bool(((val24>>(((u32(alu2))&3u)<<3u))&255u)));
  var cast1 = (bool(((val25>>(((u32(alu3))&3u)<<3u))&255u)));
  var cast2 = (bool(((val26>>(((u32(alu4))&3u)<<3u))&255u)));
  var cast3 = (bool(((val27>>(((u32(alu1))&3u)<<3u))&255u)));
  data0[alu10] = (val20*val1*select(val8,0.0f,cast0));
  data0[alu11] = (val21*val1*select(val9,0.0f,cast0));
  data0[alu12] = (val22*val1*select(val10,0.0f,cast0));
  data0[alu13] = (val23*val1*select(val11,0.0f,cast0));
  data0[alu14] = (val20*val2*select(val12,0.0f,cast1));
  data0[alu15] = (val21*val2*select(val13,0.0f,cast1));
  data0[alu16] = (val22*val2*select(val14,0.0f,cast1));
  data0[alu17] = (val23*val2*select(val15,0.0f,cast1));
  data0[alu18] = (val20*val3*select(val16,0.0f,cast2));
  data0[alu19] = (val21*val3*select(val17,0.0f,cast2));
  data0[alu20] = (val22*val3*select(val18,0.0f,cast2));
  data0[alu21] = (val23*val3*select(val19,0.0f,cast2));
  data0[alu7] = (val21*val0*select(val5,0.0f,cast3));
  data0[alu8] = (val22*val0*select(val6,0.0f,cast3));
  data0[alu9] = (val23*val0*select(val7,0.0f,cast3));
  data0[alu6] = (val20*val0*select(val4,0.0f,cast3));
}`;

const r_32_32_8_16_512_4_4_4n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<atomic<u32>>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@group(0) @binding(4)var<storage,read_write>data4:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  var acc12 = 0.0f;
  var acc13 = 0.0f;
  var acc14 = 0.0f;
  var acc15 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu0 = ((gidx0<<17)+(lidx1<<13)+(ridx0<<2));
    var val0 = data4[alu0];
    var val1 = data4[(alu0+1)];
    var val2 = data4[(alu0+2)];
    var val3 = data4[(alu0+3)];
    var val4 = data4[(alu0+2048)];
    var val5 = data4[(alu0+2049)];
    var val6 = data4[(alu0+2050)];
    var val7 = data4[(alu0+2051)];
    var val8 = data4[(alu0+4096)];
    var val9 = data4[(alu0+4097)];
    var val10 = data4[(alu0+4098)];
    var val11 = data4[(alu0+4099)];
    var val12 = data4[(alu0+6144)];
    var val13 = data4[(alu0+6145)];
    var val14 = data4[(alu0+6146)];
    var val15 = data4[(alu0+6147)];
    var alu1 = ((gidx1<<11)+(lidx0<<8)+((ridx0>>4)<<16)+((ridx0&15)<<2));
    var val16 = data3[alu1];
    var val17 = data3[(alu1+1)];
    var val18 = data3[(alu1+2)];
    var val19 = data3[(alu1+3)];
    var val20 = data3[(alu1+64)];
    var val21 = data3[(alu1+65)];
    var val22 = data3[(alu1+66)];
    var val23 = data3[(alu1+67)];
    var val24 = data3[(alu1+128)];
    var val25 = data3[(alu1+129)];
    var val26 = data3[(alu1+130)];
    var val27 = data3[(alu1+131)];
    var val28 = data3[(alu1+192)];
    var val29 = data3[(alu1+193)];
    var val30 = data3[(alu1+194)];
    var val31 = data3[(alu1+195)];
    acc0 = (acc0+(val17*val1)+(val16*val0)+(val18*val2)+(val19*val3));
    acc1 = (acc1+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3));
    acc2 = (acc2+(val24*val0)+(val25*val1)+(val26*val2)+(val27*val3));
    acc3 = (acc3+(val28*val0)+(val29*val1)+(val30*val2)+(val31*val3));
    acc4 = (acc4+(val17*val5)+(val16*val4)+(val18*val6)+(val19*val7));
    acc5 = (acc5+(val20*val4)+(val21*val5)+(val22*val6)+(val23*val7));
    acc6 = (acc6+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7));
    acc7 = (acc7+(val28*val4)+(val29*val5)+(val30*val6)+(val31*val7));
    acc8 = (acc8+(val17*val9)+(val16*val8)+(val18*val10)+(val19*val11));
    acc9 = (acc9+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc10 = (acc10+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc11 = (acc11+(val28*val8)+(val29*val9)+(val30*val10)+(val31*val11));
    acc12 = (acc12+(val17*val13)+(val16*val12)+(val18*val14)+(val19*val15));
    acc13 = (acc13+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc14 = (acc14+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc15 = (acc15+(val28*val12)+(val29*val13)+(val30*val14)+(val31*val15));
  }
  var alu19 = ((gidx1<<5)+(lidx0<<2));
  var alu20 = (alu19+1);
  var alu21 = (alu19+2);
  var alu22 = (alu19+3);
  var alu23 = ((gidx0<<6)+(gidx1<<16)+(lidx0<<13)+(lidx1<<2));
  var val32 = data2[alu23];
  var alu24 = (alu23+1);
  var val33 = data2[alu24];
  var alu25 = (alu23+2);
  var val34 = data2[alu25];
  var alu26 = (alu23+3);
  var val35 = data2[alu26];
  var alu27 = (alu23+2048);
  var val36 = data2[alu27];
  var alu28 = (alu23+2049);
  var val37 = data2[alu28];
  var alu29 = (alu23+2050);
  var val38 = data2[alu29];
  var alu30 = (alu23+2051);
  var val39 = data2[alu30];
  var alu31 = (alu23+4096);
  var val40 = data2[alu31];
  var alu32 = (alu23+4097);
  var val41 = data2[alu32];
  var alu33 = (alu23+4098);
  var val42 = data2[alu33];
  var alu34 = (alu23+4099);
  var val43 = data2[alu34];
  var alu35 = (alu23+6144);
  var val44 = data2[alu35];
  var alu36 = (alu23+6145);
  var val45 = data2[alu36];
  var alu37 = (alu23+6146);
  var val46 = data2[alu37];
  var alu38 = (alu23+6147);
  var val47 = data2[alu38];
  var val48 = atomicLoad(&data1[(alu20>>2)]);
  var val49 = atomicLoad(&data1[(alu21>>2)]);
  var val50 = atomicLoad(&data1[(alu22>>2)]);
  var val51 = atomicLoad(&data1[(alu19>>2)]);
  var cast0 = (bool(((val48>>(((u32(alu20))&3u)<<3u))&255u)));
  var cast1 = (bool(((val49>>(((u32(alu21))&3u)<<3u))&255u)));
  var cast2 = (bool(((val50>>(((u32(alu22))&3u)<<3u))&255u)));
  var cast3 = (bool(((val51>>(((u32(alu19))&3u)<<3u))&255u)));
  data0[alu27] = (select(val36,0.0f,cast0)+acc1);
  data0[alu28] = (select(val37,0.0f,cast0)+acc5);
  data0[alu29] = (select(val38,0.0f,cast0)+acc9);
  data0[alu30] = (select(val39,0.0f,cast0)+acc13);
  data0[alu31] = (select(val40,0.0f,cast1)+acc2);
  data0[alu32] = (select(val41,0.0f,cast1)+acc6);
  data0[alu33] = (select(val42,0.0f,cast1)+acc10);
  data0[alu34] = (select(val43,0.0f,cast1)+acc14);
  data0[alu35] = (select(val44,0.0f,cast2)+acc3);
  data0[alu36] = (select(val45,0.0f,cast2)+acc7);
  data0[alu37] = (select(val46,0.0f,cast2)+acc11);
  data0[alu38] = (select(val47,0.0f,cast2)+acc15);
  data0[alu24] = (select(val33,0.0f,cast3)+acc4);
  data0[alu25] = (select(val34,0.0f,cast3)+acc8);
  data0[alu26] = (select(val35,0.0f,cast3)+acc12);
  data0[alu23] = (select(val32,0.0f,cast3)+acc0);
}`;

const r_32_2672_8_16_512_3_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<i32>;
@group(0) @binding(4)var<storage,read_write>data4:array<i32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2672 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  var acc4 = 0.0f;
  var acc5 = 0.0f;
  var acc6 = 0.0f;
  var acc7 = 0.0f;
  var acc8 = 0.0f;
  var acc9 = 0.0f;
  var acc10 = 0.0f;
  var acc11 = 0.0f;
  for (var ridx0 = 0; ridx0 < 512; ridx0++) {
    var alu0 = (ridx0<<2);
    var alu1 = ((gidx0*98304)+(lidx1*6144)+alu0);
    var val0 = data2[alu1];
    var val1 = data2[(alu1+1)];
    var val2 = data2[(alu1+2)];
    var val3 = data2[(alu1+3)];
    var val4 = data2[(alu1+2048)];
    var val5 = data2[(alu1+2049)];
    var val6 = data2[(alu1+2050)];
    var val7 = data2[(alu1+2051)];
    var val8 = data2[(alu1+4096)];
    var val9 = data2[(alu1+4097)];
    var val10 = data2[(alu1+4098)];
    var val11 = data2[(alu1+4099)];
    var alu2 = ((gidx1<<16)+(lidx0<<13)+alu0);
    var val12 = data1[alu2];
    var val13 = data1[(alu2+1)];
    var val14 = data1[(alu2+2)];
    var val15 = data1[(alu2+3)];
    var val16 = data1[(alu2+2048)];
    var val17 = data1[(alu2+2049)];
    var val18 = data1[(alu2+2050)];
    var val19 = data1[(alu2+2051)];
    var val20 = data1[(alu2+4096)];
    var val21 = data1[(alu2+4097)];
    var val22 = data1[(alu2+4098)];
    var val23 = data1[(alu2+4099)];
    var val24 = data1[(alu2+6144)];
    var val25 = data1[(alu2+6145)];
    var val26 = data1[(alu2+6146)];
    var val27 = data1[(alu2+6147)];
    acc0 = (acc0+(val13*val1)+(val12*val0)+(val14*val2)+(val15*val3));
    acc1 = (acc1+(val16*val0)+(val17*val1)+(val18*val2)+(val19*val3));
    acc2 = (acc2+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3));
    acc3 = (acc3+(val24*val0)+(val25*val1)+(val26*val2)+(val27*val3));
    acc4 = (acc4+(val13*val5)+(val12*val4)+(val14*val6)+(val15*val7));
    acc5 = (acc5+(val16*val4)+(val17*val5)+(val18*val6)+(val19*val7));
    acc6 = (acc6+(val20*val4)+(val21*val5)+(val22*val6)+(val23*val7));
    acc7 = (acc7+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7));
    acc8 = (acc8+(val13*val9)+(val12*val8)+(val14*val10)+(val15*val11));
    acc9 = (acc9+(val16*val8)+(val17*val9)+(val18*val10)+(val19*val11));
    acc10 = (acc10+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc11 = (acc11+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
  }
  var val28 = data3[0];
  var alu16 = ((gidx1<<5)+(lidx0<<2));
  var val29 = data4[alu16];
  var val30 = data4[(alu16+1)];
  var val31 = data4[(alu16+2)];
  var val32 = data4[(alu16+3)];
  var alu17 = ((gidx0*48)+(gidx1*4104192)+(lidx0*513024)+(lidx1*3));
  var cast0 = (f32(((val28!=val30)!=true)));
  data0[(alu17+128256)] = (cast0*acc1);
  data0[(alu17+128257)] = (cast0*acc5);
  data0[(alu17+128258)] = (cast0*acc9);
  var cast1 = (f32(((val28!=val31)!=true)));
  data0[(alu17+256512)] = (cast1*acc2);
  data0[(alu17+256513)] = (cast1*acc6);
  data0[(alu17+256514)] = (cast1*acc10);
  var cast2 = (f32(((val28!=val32)!=true)));
  data0[(alu17+384768)] = (cast2*acc3);
  data0[(alu17+384769)] = (cast2*acc7);
  data0[(alu17+384770)] = (cast2*acc11);
  var cast3 = (f32(((val28!=val29)!=true)));
  data0[alu17] = (cast3*acc0);
  data0[(alu17+1)] = (cast3*acc4);
  data0[(alu17+2)] = (cast3*acc8);
}`;

const r_4008_32_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4008 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+(gidx0<<5));
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu1 = (alu0+(ridx0*513024));
    var val0 = data1[alu1];
    var val1 = data1[(alu1+128256)];
    var val2 = data1[(alu1+256512)];
    var val3 = data1[(alu1+384768)];
    acc0 = (acc0+val3+val2+val0+val1);
  }
  data0[alu0] = acc0;
}`;

const r_8_32_501 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 32 */
  var acc0 = (f32(-inf(1.0)));
  for (var ridx0 = 0; ridx0 < 501; ridx0++) {
    var val0 = data1[((gidx0*16032)+(lidx0*501)+ridx0)];
    var alu0 = (select(val0,(f32(-inf(1.0))),is_nan(val0))*1.0526316165924072f);
    acc0 = select(acc0,alu0,(acc0<alu0));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[lidx0];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = (f32(-inf(1.0)));
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = select(acc0,val1,(acc0<val1));
    }
    data0[0] = acc0;
  }
}`;

const r_8_32_501n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 32 */
  var val0 = data2[0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 501; ridx0++) {
    var val1 = data1[((gidx0*16032)+(lidx0*501)+ridx0)];
    acc0 = (acc0+exp2((((select(val1,(f32(-inf(1.0))),is_nan(val1))*1.0526316165924072f)-val0)*1.4426950408889634f)));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_256n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[lidx0];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0.0f;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[0] = acc0;
  }
}`;

const E_1336_32_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1336 */
  var lidx0 = i32(lindex.x); /* 32 */
  var val0 = data2[0];
  var val1 = data3[0];
  var alu0 = ((gidx0*96)+(lidx0*3));
  var val2 = data1[alu0];
  var alu1 = (alu0+1);
  var val3 = data1[alu1];
  var alu2 = (alu0+2);
  var val4 = data1[alu2];
  var alu3 = (1/val1);
  data0[alu1] = (exp2((((select(val3,(f32(-inf(1.0))),is_nan(val3))*1.0526316165924072f)-val0)*1.4426950408889634f))*alu3);
  data0[alu2] = (exp2((((select(val4,(f32(-inf(1.0))),is_nan(val4))*1.0526316165924072f)-val0)*1.4426950408889634f))*alu3);
  data0[alu0] = (exp2((((select(val2,(f32(-inf(1.0))),is_nan(val2))*1.0526316165924072f)-val0)*1.4426950408889634f))*alu3);
}`;

const r_167_16_3_16_64_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(3,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 167 */
  var lidx0 = i32(lindex.x); /* 3 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<4);
  var alu1 = (gidx1*768);
  var alu2 = (lidx0<<8);
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 64; ridx0++) {
    var alu3 = (lidx1+alu0+(ridx0<<2));
    var alu4 = (alu3+alu1+alu2);
    var val0 = select(0.0f, data1[(alu4+-252)], ((alu3<252)!=true));
    var val1 = select(0.0f, data1[(alu4+-253)], ((alu3<253)!=true));
    var val2 = select(0.0f, data1[(alu4+-254)], ((alu3<254)!=true));
    var val3 = select(0.0f, data1[(alu4+-255)], ((alu3<255)!=true));
    acc0 = (acc0+val0+val1+val3+val2);
  }
  data0[(lidx1+alu0+alu1+alu2)] = acc0;
}`;

const r_167_3_501 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 167 */
  var lidx0 = i32(lindex.x); /* 3 */
  var alu0 = (lidx0+(gidx0*3));
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 501; ridx0++) {
    var val0 = select(0.0f, data1[((gidx0*768)+(lidx0<<8)+(ridx0<<8)+-128001)], (((alu0+ridx0)<501)!=true));
    acc0 = (acc0+val0);
  }
  data0[alu0] = acc0;
}`;

const r_8_32_501n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<i32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 32 */
  var val0 = data1[0];
  var val1 = data2[128255];
  var val2 = data3[500];
  var acc0 = 0;
  for (var ridx0 = 0; ridx0 < 501; ridx0++) {
    var alu0 = ((gidx0*16032)+(lidx0*501)+ridx0);
    var val3 = data2[alu0];
    var val4 = data3[(alu0>>8)];
    acc0 = (acc0+(i32(((val0<((1/(val1+val2))*(val3+val4)))!=true))));
  }
  data0[(lidx0+(gidx0<<5))] = acc0;
}`;

const r_256n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<i32, 256>;
@group(0) @binding(0)var<storage,read_write>data0:array<i32>;
@group(0) @binding(1)var<storage,read_write>data1:array<i32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var lidx0 = i32(lindex.x); /* 256 */
  var val0 = data1[lidx0];
  temp0[lidx0] = val0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc0 = 0;
    for (var ridx0 = 0; ridx0 < 256; ridx0++) {
      var val1 = temp0[ridx0];
      acc0 = (acc0+val1);
    }
    data0[0] = acc0;
  }
}`;

    return {
      "setup": async (device, safetensor) => {
        const metadata = safetensor ? getTensorMetadata(safetensor[0]) : null;

        const buf_0 = createEmptyBuf(device, 4096);;
    const input0 = createEmptyBuf(device, 4096);;
    const buf_1 = createEmptyBuf(device, 4194304);;
    const buf_2 = createEmptyBuf(device, 4);;
    const buf_3 = createEmptyBuf(device, 1024);;
    const buf_4 = createEmptyBuf(device, 4096);;
    const buf_5 = createEmptyBuf(device, 4);;
    const buf_6 = createEmptyBuf(device, 8);;
    const buf_7 = createEmptyBuf(device, 4096);;
    const buf_8 = createEmptyBuf(device, 4096);;
    const buf_9 = createEmptyBuf(device, 4);;
    const buf_10 = createEmptyBuf(device, 8388608);;
    const buf_11 = createWeightBuf(device, 513024, getTensorBuffer(safetensor, metadata['tok_embeddings.arange'], 'tok_embeddings.arange'));
    const buf_12 = createWeightBuf(device, 1050673152, getTensorBuffer(safetensor, metadata['tok_embeddings.weight'], 'tok_embeddings.weight'));
    const buf_13 = createEmptyBuf(device, 16);;
    const buf_14 = createEmptyBuf(device, 4096);;
    const buf_15 = createEmptyBuf(device, 4);;
    const buf_16 = createEmptyBuf(device, 8388608);;
    const buf_17 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.0.attention_norm.weight'], 'layers.0.attention_norm.weight'));
    const buf_18 = createEmptyBuf(device, 2097152);;
    const buf_19 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.0.attention.wv.weight'], 'layers.0.attention.wv.weight'));
    const buf_20 = createEmptyBuf(device, 8388608);;
    const buf_21 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.0.attention.wq.weight'], 'layers.0.attention.wq.weight'));
    const buf_22 = createEmptyBuf(device, 2097152);;
    const buf_23 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.0.attention.wk.weight'], 'layers.0.attention.wk.weight'));
    const buf_24 = createWeightBuf(device, 524288, getTensorBuffer(safetensor, metadata['freqs_cis'], 'freqs_cis'));
    const buf_25 = createEmptyBuf(device, 2097152);;
    const buf_26 = createEmptyBuf(device, 134217728);;
    const buf_27 = createEmptyBuf(device, 131072);;
    const buf_28 = createEmptyBuf(device, 131072);;
    const buf_29 = createEmptyBuf(device, 134217728);;
    const buf_30 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.0.attention.wo.weight'], 'layers.0.attention.wo.weight'));
    const buf_31 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.0.ffn_norm.weight'], 'layers.0.ffn_norm.weight'));
    const buf_32 = createEmptyBuf(device, 33554432);;
    const buf_33 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.0.feed_forward.w3.weight'], 'layers.0.feed_forward.w3.weight'));
    const buf_34 = createEmptyBuf(device, 33554432);;
    const buf_35 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.0.feed_forward.w1.weight'], 'layers.0.feed_forward.w1.weight'));
    const buf_36 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.0.feed_forward.w2.weight'], 'layers.0.feed_forward.w2.weight'));
    const buf_37 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.1.attention_norm.weight'], 'layers.1.attention_norm.weight'));
    const buf_38 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.1.attention.wv.weight'], 'layers.1.attention.wv.weight'));
    const buf_39 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.1.attention.wq.weight'], 'layers.1.attention.wq.weight'));
    const buf_40 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.1.attention.wk.weight'], 'layers.1.attention.wk.weight'));
    const buf_41 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.1.attention.wo.weight'], 'layers.1.attention.wo.weight'));
    const buf_42 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.1.ffn_norm.weight'], 'layers.1.ffn_norm.weight'));
    const buf_43 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.1.feed_forward.w3.weight'], 'layers.1.feed_forward.w3.weight'));
    const buf_44 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.1.feed_forward.w1.weight'], 'layers.1.feed_forward.w1.weight'));
    const buf_45 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.1.feed_forward.w2.weight'], 'layers.1.feed_forward.w2.weight'));
    const buf_46 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.2.attention_norm.weight'], 'layers.2.attention_norm.weight'));
    const buf_47 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.2.attention.wv.weight'], 'layers.2.attention.wv.weight'));
    const buf_48 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.2.attention.wq.weight'], 'layers.2.attention.wq.weight'));
    const buf_49 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.2.attention.wk.weight'], 'layers.2.attention.wk.weight'));
    const buf_50 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.2.attention.wo.weight'], 'layers.2.attention.wo.weight'));
    const buf_51 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.2.ffn_norm.weight'], 'layers.2.ffn_norm.weight'));
    const buf_52 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.2.feed_forward.w3.weight'], 'layers.2.feed_forward.w3.weight'));
    const buf_53 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.2.feed_forward.w1.weight'], 'layers.2.feed_forward.w1.weight'));
    const buf_54 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.2.feed_forward.w2.weight'], 'layers.2.feed_forward.w2.weight'));
    const buf_55 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.3.attention_norm.weight'], 'layers.3.attention_norm.weight'));
    const buf_56 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.3.attention.wv.weight'], 'layers.3.attention.wv.weight'));
    const buf_57 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.3.attention.wq.weight'], 'layers.3.attention.wq.weight'));
    const buf_58 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.3.attention.wk.weight'], 'layers.3.attention.wk.weight'));
    const buf_59 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.3.attention.wo.weight'], 'layers.3.attention.wo.weight'));
    const buf_60 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.3.ffn_norm.weight'], 'layers.3.ffn_norm.weight'));
    const buf_61 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.3.feed_forward.w3.weight'], 'layers.3.feed_forward.w3.weight'));
    const buf_62 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.3.feed_forward.w1.weight'], 'layers.3.feed_forward.w1.weight'));
    const buf_63 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.3.feed_forward.w2.weight'], 'layers.3.feed_forward.w2.weight'));
    const buf_64 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.4.attention_norm.weight'], 'layers.4.attention_norm.weight'));
    const buf_65 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.4.attention.wv.weight'], 'layers.4.attention.wv.weight'));
    const buf_66 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.4.attention.wq.weight'], 'layers.4.attention.wq.weight'));
    const buf_67 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.4.attention.wk.weight'], 'layers.4.attention.wk.weight'));
    const buf_68 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.4.attention.wo.weight'], 'layers.4.attention.wo.weight'));
    const buf_69 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.4.ffn_norm.weight'], 'layers.4.ffn_norm.weight'));
    const buf_70 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.4.feed_forward.w3.weight'], 'layers.4.feed_forward.w3.weight'));
    const buf_71 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.4.feed_forward.w1.weight'], 'layers.4.feed_forward.w1.weight'));
    const buf_72 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.4.feed_forward.w2.weight'], 'layers.4.feed_forward.w2.weight'));
    const buf_73 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.5.attention_norm.weight'], 'layers.5.attention_norm.weight'));
    const buf_74 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.5.attention.wv.weight'], 'layers.5.attention.wv.weight'));
    const buf_75 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.5.attention.wq.weight'], 'layers.5.attention.wq.weight'));
    const buf_76 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.5.attention.wk.weight'], 'layers.5.attention.wk.weight'));
    const buf_77 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.5.attention.wo.weight'], 'layers.5.attention.wo.weight'));
    const buf_78 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.5.ffn_norm.weight'], 'layers.5.ffn_norm.weight'));
    const buf_79 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.5.feed_forward.w3.weight'], 'layers.5.feed_forward.w3.weight'));
    const buf_80 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.5.feed_forward.w1.weight'], 'layers.5.feed_forward.w1.weight'));
    const buf_81 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.5.feed_forward.w2.weight'], 'layers.5.feed_forward.w2.weight'));
    const buf_82 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.6.attention_norm.weight'], 'layers.6.attention_norm.weight'));
    const buf_83 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.6.attention.wv.weight'], 'layers.6.attention.wv.weight'));
    const buf_84 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.6.attention.wq.weight'], 'layers.6.attention.wq.weight'));
    const buf_85 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.6.attention.wk.weight'], 'layers.6.attention.wk.weight'));
    const buf_86 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.6.attention.wo.weight'], 'layers.6.attention.wo.weight'));
    const buf_87 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.6.ffn_norm.weight'], 'layers.6.ffn_norm.weight'));
    const buf_88 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.6.feed_forward.w3.weight'], 'layers.6.feed_forward.w3.weight'));
    const buf_89 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.6.feed_forward.w1.weight'], 'layers.6.feed_forward.w1.weight'));
    const buf_90 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.6.feed_forward.w2.weight'], 'layers.6.feed_forward.w2.weight'));
    const buf_91 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.7.attention_norm.weight'], 'layers.7.attention_norm.weight'));
    const buf_92 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.7.attention.wv.weight'], 'layers.7.attention.wv.weight'));
    const buf_93 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.7.attention.wq.weight'], 'layers.7.attention.wq.weight'));
    const buf_94 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.7.attention.wk.weight'], 'layers.7.attention.wk.weight'));
    const buf_95 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.7.attention.wo.weight'], 'layers.7.attention.wo.weight'));
    const buf_96 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.7.ffn_norm.weight'], 'layers.7.ffn_norm.weight'));
    const buf_97 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.7.feed_forward.w3.weight'], 'layers.7.feed_forward.w3.weight'));
    const buf_98 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.7.feed_forward.w1.weight'], 'layers.7.feed_forward.w1.weight'));
    const buf_99 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.7.feed_forward.w2.weight'], 'layers.7.feed_forward.w2.weight'));
    const buf_100 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.8.attention_norm.weight'], 'layers.8.attention_norm.weight'));
    const buf_101 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.8.attention.wv.weight'], 'layers.8.attention.wv.weight'));
    const buf_102 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.8.attention.wq.weight'], 'layers.8.attention.wq.weight'));
    const buf_103 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.8.attention.wk.weight'], 'layers.8.attention.wk.weight'));
    const buf_104 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.8.attention.wo.weight'], 'layers.8.attention.wo.weight'));
    const buf_105 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.8.ffn_norm.weight'], 'layers.8.ffn_norm.weight'));
    const buf_106 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.8.feed_forward.w3.weight'], 'layers.8.feed_forward.w3.weight'));
    const buf_107 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.8.feed_forward.w1.weight'], 'layers.8.feed_forward.w1.weight'));
    const buf_108 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.8.feed_forward.w2.weight'], 'layers.8.feed_forward.w2.weight'));
    const buf_109 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.9.attention_norm.weight'], 'layers.9.attention_norm.weight'));
    const buf_110 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.9.attention.wv.weight'], 'layers.9.attention.wv.weight'));
    const buf_111 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.9.attention.wq.weight'], 'layers.9.attention.wq.weight'));
    const buf_112 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.9.attention.wk.weight'], 'layers.9.attention.wk.weight'));
    const buf_113 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.9.attention.wo.weight'], 'layers.9.attention.wo.weight'));
    const buf_114 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.9.ffn_norm.weight'], 'layers.9.ffn_norm.weight'));
    const buf_115 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.9.feed_forward.w3.weight'], 'layers.9.feed_forward.w3.weight'));
    const buf_116 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.9.feed_forward.w1.weight'], 'layers.9.feed_forward.w1.weight'));
    const buf_117 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.9.feed_forward.w2.weight'], 'layers.9.feed_forward.w2.weight'));
    const buf_118 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.10.attention_norm.weight'], 'layers.10.attention_norm.weight'));
    const buf_119 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.10.attention.wv.weight'], 'layers.10.attention.wv.weight'));
    const buf_120 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.10.attention.wq.weight'], 'layers.10.attention.wq.weight'));
    const buf_121 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.10.attention.wk.weight'], 'layers.10.attention.wk.weight'));
    const buf_122 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.10.attention.wo.weight'], 'layers.10.attention.wo.weight'));
    const buf_123 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.10.ffn_norm.weight'], 'layers.10.ffn_norm.weight'));
    const buf_124 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.10.feed_forward.w3.weight'], 'layers.10.feed_forward.w3.weight'));
    const buf_125 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.10.feed_forward.w1.weight'], 'layers.10.feed_forward.w1.weight'));
    const buf_126 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.10.feed_forward.w2.weight'], 'layers.10.feed_forward.w2.weight'));
    const buf_127 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.11.attention_norm.weight'], 'layers.11.attention_norm.weight'));
    const buf_128 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.11.attention.wv.weight'], 'layers.11.attention.wv.weight'));
    const buf_129 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.11.attention.wq.weight'], 'layers.11.attention.wq.weight'));
    const buf_130 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.11.attention.wk.weight'], 'layers.11.attention.wk.weight'));
    const buf_131 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.11.attention.wo.weight'], 'layers.11.attention.wo.weight'));
    const buf_132 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.11.ffn_norm.weight'], 'layers.11.ffn_norm.weight'));
    const buf_133 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.11.feed_forward.w3.weight'], 'layers.11.feed_forward.w3.weight'));
    const buf_134 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.11.feed_forward.w1.weight'], 'layers.11.feed_forward.w1.weight'));
    const buf_135 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.11.feed_forward.w2.weight'], 'layers.11.feed_forward.w2.weight'));
    const buf_136 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.12.attention_norm.weight'], 'layers.12.attention_norm.weight'));
    const buf_137 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.12.attention.wv.weight'], 'layers.12.attention.wv.weight'));
    const buf_138 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.12.attention.wq.weight'], 'layers.12.attention.wq.weight'));
    const buf_139 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.12.attention.wk.weight'], 'layers.12.attention.wk.weight'));
    const buf_140 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.12.attention.wo.weight'], 'layers.12.attention.wo.weight'));
    const buf_141 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.12.ffn_norm.weight'], 'layers.12.ffn_norm.weight'));
    const buf_142 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.12.feed_forward.w3.weight'], 'layers.12.feed_forward.w3.weight'));
    const buf_143 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.12.feed_forward.w1.weight'], 'layers.12.feed_forward.w1.weight'));
    const buf_144 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.12.feed_forward.w2.weight'], 'layers.12.feed_forward.w2.weight'));
    const buf_145 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.13.attention_norm.weight'], 'layers.13.attention_norm.weight'));
    const buf_146 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.13.attention.wv.weight'], 'layers.13.attention.wv.weight'));
    const buf_147 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.13.attention.wq.weight'], 'layers.13.attention.wq.weight'));
    const buf_148 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.13.attention.wk.weight'], 'layers.13.attention.wk.weight'));
    const buf_149 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.13.attention.wo.weight'], 'layers.13.attention.wo.weight'));
    const buf_150 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.13.ffn_norm.weight'], 'layers.13.ffn_norm.weight'));
    const buf_151 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.13.feed_forward.w3.weight'], 'layers.13.feed_forward.w3.weight'));
    const buf_152 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.13.feed_forward.w1.weight'], 'layers.13.feed_forward.w1.weight'));
    const buf_153 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.13.feed_forward.w2.weight'], 'layers.13.feed_forward.w2.weight'));
    const buf_154 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.14.attention_norm.weight'], 'layers.14.attention_norm.weight'));
    const buf_155 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.14.attention.wv.weight'], 'layers.14.attention.wv.weight'));
    const buf_156 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.14.attention.wq.weight'], 'layers.14.attention.wq.weight'));
    const buf_157 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.14.attention.wk.weight'], 'layers.14.attention.wk.weight'));
    const buf_158 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.14.attention.wo.weight'], 'layers.14.attention.wo.weight'));
    const buf_159 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.14.ffn_norm.weight'], 'layers.14.ffn_norm.weight'));
    const buf_160 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.14.feed_forward.w3.weight'], 'layers.14.feed_forward.w3.weight'));
    const buf_161 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.14.feed_forward.w1.weight'], 'layers.14.feed_forward.w1.weight'));
    const buf_162 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.14.feed_forward.w2.weight'], 'layers.14.feed_forward.w2.weight'));
    const buf_163 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.15.attention_norm.weight'], 'layers.15.attention_norm.weight'));
    const buf_164 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.15.attention.wv.weight'], 'layers.15.attention.wv.weight'));
    const buf_165 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.15.attention.wq.weight'], 'layers.15.attention.wq.weight'));
    const buf_166 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.15.attention.wk.weight'], 'layers.15.attention.wk.weight'));
    const buf_167 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.15.attention.wo.weight'], 'layers.15.attention.wo.weight'));
    const buf_168 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.15.ffn_norm.weight'], 'layers.15.ffn_norm.weight'));
    const buf_169 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.15.feed_forward.w3.weight'], 'layers.15.feed_forward.w3.weight'));
    const buf_170 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.15.feed_forward.w1.weight'], 'layers.15.feed_forward.w1.weight'));
    const buf_171 = createEmptyBuf(device, 8388608);;
    const buf_172 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.15.feed_forward.w2.weight'], 'layers.15.feed_forward.w2.weight'));
    const buf_173 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['norm.weight'], 'norm.weight'));
    const buf_174 = createEmptyBuf(device, 525336576);;
    const buf_175 = createWeightBuf(device, 1050673152, getTensorBuffer(safetensor, metadata['output.weight'], 'output.weight'));
    const buf_176 = createEmptyBuf(device, 513024);;
    const buf_177 = createEmptyBuf(device, 1024);;
    const buf_178 = createEmptyBuf(device, 4);;
    const buf_179 = createEmptyBuf(device, 4);;
    const buf_180 = createEmptyBuf(device, 513024);;
    const buf_181 = createEmptyBuf(device, 513024);;
    const buf_182 = createEmptyBuf(device, 2004);;
    const buf_183 = createEmptyBuf(device, 1024);;
    const output0 = createEmptyBuf(device, 4);;

        const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
        const gpuReadBuffer = device.createBuffer({ size: output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        const kernels = [E_8_32_4, E_128_16_8_16_4, E_n7, E_8_32_4n1, r_1024_16_64, E_n5, E_8_32_4n2, r_4_256_16_16, E_n6, r_32_32_8_16_32064_4_4_4, r_4_4, r_64_4_8_256_4, r_256_4, E_32_32_8_16_4_4, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n1, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_8_8_16_512_4_4_4, r_32_32_8_16_512_4_4_4, r_32_8_8_16_512_4_4_4, E_1024_2_8_16_2_4, E_256_2_4_2_16_2_4, r_8_32_16_8_16_64_4_4_4, r_1024_32_256_4, r_256_32_256_4_4, E_4096_16_8_16_4, r_32_32_8_16_256_4_4_4, r_32_32_8_16_512_4_4_4n2, r_64_4_8_256_4, E_32_32_8_16_4_4, r_32_128_8_16_512_4_4_4, r_32_128_8_16_512_4_4_4n1, r_32_32_8_16_2048_4_4_4, r_1024_16_128, E_32_32_8_16_4_4n1, r_32_2672_8_16_512_3_4_4, r_4008_32_256_4, r_8_32_501, r_256, r_8_32_501n1, r_256n1, E_1336_32_3, r_167_16_3_16_64_4, r_167_3_501, r_8_32_501n2, r_256n2];
        const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({layout: "auto", compute: { module: device.createShaderModule({ code: name }), entryPoint: "main" }})));

        return async (data0) => {
            const commandEncoder = device.createCommandEncoder();

            await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
    new Int32Array(gpuWriteBuffer0.getMappedRange()).set(data0);
    gpuWriteBuffer0.unmap();
commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);

            addComputePass(device, commandEncoder, piplines[0], [buf_0, input0], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[1], [buf_1, buf_0], [16, 128, 1]);
        addComputePass(device, commandEncoder, piplines[2], [buf_2], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[3], [buf_3, input0], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[4], [buf_4], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[5], [buf_5, buf_2, buf_6], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[6], [buf_7, buf_3, input0], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[7], [buf_8, buf_3], [256, 4, 1]);
        addComputePass(device, commandEncoder, piplines[8], [buf_9, buf_2, buf_6], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[9], [buf_10, buf_11, buf_7, buf_12], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[10], [buf_13, buf_8], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[11], [buf_14, buf_10], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[12], [buf_15, buf_8, buf_13], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[13], [buf_16, buf_10, buf_14, buf_17], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[14], [buf_18, buf_16, buf_19], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[15], [buf_20, buf_16, buf_21], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[16], [buf_22, buf_16, buf_23], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[17], [buf_16, buf_20, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[18], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[19], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[20], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[21], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[22], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[23], [buf_20, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[24], [buf_16, buf_10, buf_20, buf_30], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[25], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[26], [buf_10, buf_16, buf_14, buf_31], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[27], [buf_32, buf_10, buf_33], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[28], [buf_34, buf_10, buf_35, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[29], [buf_20, buf_16, buf_34, buf_36], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[30], [buf_14, buf_3, buf_20], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[31], [buf_16, buf_3, buf_20, buf_14, buf_37], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[32], [buf_18, buf_16, buf_38], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[33], [buf_10, buf_16, buf_39], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[34], [buf_22, buf_16, buf_40], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[35], [buf_16, buf_10, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[36], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[37], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[38], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[39], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[40], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[41], [buf_10, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[42], [buf_16, buf_3, buf_20, buf_10, buf_41], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[43], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[44], [buf_20, buf_16, buf_14, buf_42], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[45], [buf_32, buf_20, buf_43], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[46], [buf_34, buf_20, buf_44, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[47], [buf_10, buf_16, buf_34, buf_45], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[48], [buf_14, buf_3, buf_10], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[49], [buf_16, buf_3, buf_10, buf_14, buf_46], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[50], [buf_18, buf_16, buf_47], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[51], [buf_20, buf_16, buf_48], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[52], [buf_22, buf_16, buf_49], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[53], [buf_16, buf_20, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[54], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[55], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[56], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[57], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[58], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[59], [buf_20, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[60], [buf_16, buf_3, buf_10, buf_20, buf_50], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[61], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[62], [buf_10, buf_16, buf_14, buf_51], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[63], [buf_32, buf_10, buf_52], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[64], [buf_34, buf_10, buf_53, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[65], [buf_20, buf_16, buf_34, buf_54], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[66], [buf_14, buf_3, buf_20], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[67], [buf_16, buf_3, buf_20, buf_14, buf_55], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[68], [buf_18, buf_16, buf_56], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[69], [buf_10, buf_16, buf_57], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[70], [buf_22, buf_16, buf_58], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[71], [buf_16, buf_10, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[72], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[73], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[74], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[75], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[76], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[77], [buf_10, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[78], [buf_16, buf_3, buf_20, buf_10, buf_59], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[79], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[80], [buf_20, buf_16, buf_14, buf_60], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[81], [buf_32, buf_20, buf_61], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[82], [buf_34, buf_20, buf_62, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[83], [buf_10, buf_16, buf_34, buf_63], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[84], [buf_14, buf_3, buf_10], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[85], [buf_16, buf_3, buf_10, buf_14, buf_64], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[86], [buf_18, buf_16, buf_65], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[87], [buf_20, buf_16, buf_66], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[88], [buf_22, buf_16, buf_67], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[89], [buf_16, buf_20, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[90], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[91], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[92], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[93], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[94], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[95], [buf_20, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[96], [buf_16, buf_3, buf_10, buf_20, buf_68], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[97], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[98], [buf_10, buf_16, buf_14, buf_69], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[99], [buf_32, buf_10, buf_70], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[100], [buf_34, buf_10, buf_71, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[101], [buf_20, buf_16, buf_34, buf_72], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[102], [buf_14, buf_3, buf_20], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[103], [buf_16, buf_3, buf_20, buf_14, buf_73], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[104], [buf_18, buf_16, buf_74], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[105], [buf_10, buf_16, buf_75], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[106], [buf_22, buf_16, buf_76], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[107], [buf_16, buf_10, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[108], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[109], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[110], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[111], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[112], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[113], [buf_10, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[114], [buf_16, buf_3, buf_20, buf_10, buf_77], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[115], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[116], [buf_20, buf_16, buf_14, buf_78], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[117], [buf_32, buf_20, buf_79], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[118], [buf_34, buf_20, buf_80, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[119], [buf_10, buf_16, buf_34, buf_81], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[120], [buf_14, buf_3, buf_10], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[121], [buf_16, buf_3, buf_10, buf_14, buf_82], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[122], [buf_18, buf_16, buf_83], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[123], [buf_20, buf_16, buf_84], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[124], [buf_22, buf_16, buf_85], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[125], [buf_16, buf_20, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[126], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[127], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[128], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[129], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[130], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[131], [buf_20, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[132], [buf_16, buf_3, buf_10, buf_20, buf_86], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[133], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[134], [buf_10, buf_16, buf_14, buf_87], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[135], [buf_32, buf_10, buf_88], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[136], [buf_34, buf_10, buf_89, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[137], [buf_20, buf_16, buf_34, buf_90], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[138], [buf_14, buf_3, buf_20], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[139], [buf_16, buf_3, buf_20, buf_14, buf_91], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[140], [buf_18, buf_16, buf_92], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[141], [buf_10, buf_16, buf_93], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[142], [buf_22, buf_16, buf_94], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[143], [buf_16, buf_10, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[144], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[145], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[146], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[147], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[148], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[149], [buf_10, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[150], [buf_16, buf_3, buf_20, buf_10, buf_95], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[151], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[152], [buf_20, buf_16, buf_14, buf_96], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[153], [buf_32, buf_20, buf_97], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[154], [buf_34, buf_20, buf_98, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[155], [buf_10, buf_16, buf_34, buf_99], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[156], [buf_14, buf_3, buf_10], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[157], [buf_16, buf_3, buf_10, buf_14, buf_100], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[158], [buf_18, buf_16, buf_101], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[159], [buf_20, buf_16, buf_102], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[160], [buf_22, buf_16, buf_103], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[161], [buf_16, buf_20, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[162], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[163], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[164], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[165], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[166], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[167], [buf_20, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[168], [buf_16, buf_3, buf_10, buf_20, buf_104], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[169], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[170], [buf_10, buf_16, buf_14, buf_105], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[171], [buf_32, buf_10, buf_106], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[172], [buf_34, buf_10, buf_107, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[173], [buf_20, buf_16, buf_34, buf_108], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[174], [buf_14, buf_3, buf_20], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[175], [buf_16, buf_3, buf_20, buf_14, buf_109], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[176], [buf_18, buf_16, buf_110], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[177], [buf_10, buf_16, buf_111], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[178], [buf_22, buf_16, buf_112], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[179], [buf_16, buf_10, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[180], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[181], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[182], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[183], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[184], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[185], [buf_10, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[186], [buf_16, buf_3, buf_20, buf_10, buf_113], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[187], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[188], [buf_20, buf_16, buf_14, buf_114], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[189], [buf_32, buf_20, buf_115], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[190], [buf_34, buf_20, buf_116, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[191], [buf_10, buf_16, buf_34, buf_117], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[192], [buf_14, buf_3, buf_10], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[193], [buf_16, buf_3, buf_10, buf_14, buf_118], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[194], [buf_18, buf_16, buf_119], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[195], [buf_20, buf_16, buf_120], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[196], [buf_22, buf_16, buf_121], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[197], [buf_16, buf_20, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[198], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[199], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[200], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[201], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[202], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[203], [buf_20, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[204], [buf_16, buf_3, buf_10, buf_20, buf_122], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[205], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[206], [buf_10, buf_16, buf_14, buf_123], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[207], [buf_32, buf_10, buf_124], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[208], [buf_34, buf_10, buf_125, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[209], [buf_20, buf_16, buf_34, buf_126], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[210], [buf_14, buf_3, buf_20], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[211], [buf_16, buf_3, buf_20, buf_14, buf_127], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[212], [buf_18, buf_16, buf_128], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[213], [buf_10, buf_16, buf_129], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[214], [buf_22, buf_16, buf_130], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[215], [buf_16, buf_10, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[216], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[217], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[218], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[219], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[220], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[221], [buf_10, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[222], [buf_16, buf_3, buf_20, buf_10, buf_131], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[223], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[224], [buf_20, buf_16, buf_14, buf_132], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[225], [buf_32, buf_20, buf_133], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[226], [buf_34, buf_20, buf_134, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[227], [buf_10, buf_16, buf_34, buf_135], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[228], [buf_14, buf_3, buf_10], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[229], [buf_16, buf_3, buf_10, buf_14, buf_136], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[230], [buf_18, buf_16, buf_137], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[231], [buf_20, buf_16, buf_138], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[232], [buf_22, buf_16, buf_139], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[233], [buf_16, buf_20, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[234], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[235], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[236], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[237], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[238], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[239], [buf_20, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[240], [buf_16, buf_3, buf_10, buf_20, buf_140], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[241], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[242], [buf_10, buf_16, buf_14, buf_141], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[243], [buf_32, buf_10, buf_142], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[244], [buf_34, buf_10, buf_143, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[245], [buf_20, buf_16, buf_34, buf_144], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[246], [buf_14, buf_3, buf_20], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[247], [buf_16, buf_3, buf_20, buf_14, buf_145], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[248], [buf_18, buf_16, buf_146], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[249], [buf_10, buf_16, buf_147], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[250], [buf_22, buf_16, buf_148], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[251], [buf_16, buf_10, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[252], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[253], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[254], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[255], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[256], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[257], [buf_10, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[258], [buf_16, buf_3, buf_20, buf_10, buf_149], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[259], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[260], [buf_20, buf_16, buf_14, buf_150], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[261], [buf_32, buf_20, buf_151], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[262], [buf_34, buf_20, buf_152, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[263], [buf_10, buf_16, buf_34, buf_153], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[264], [buf_14, buf_3, buf_10], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[265], [buf_16, buf_3, buf_10, buf_14, buf_154], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[266], [buf_18, buf_16, buf_155], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[267], [buf_20, buf_16, buf_156], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[268], [buf_22, buf_16, buf_157], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[269], [buf_16, buf_20, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[270], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[271], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[272], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[273], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[274], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[275], [buf_20, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[276], [buf_16, buf_3, buf_10, buf_20, buf_158], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[277], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[278], [buf_10, buf_16, buf_14, buf_159], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[279], [buf_32, buf_10, buf_160], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[280], [buf_34, buf_10, buf_161, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[281], [buf_20, buf_16, buf_34, buf_162], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[282], [buf_14, buf_3, buf_20], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[283], [buf_16, buf_3, buf_20, buf_14, buf_163], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[284], [buf_18, buf_16, buf_164], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[285], [buf_10, buf_16, buf_165], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[286], [buf_22, buf_16, buf_166], [8, 32, 1]);
        addComputePass(device, commandEncoder, piplines[287], [buf_16, buf_10, buf_24], [2, 1024, 1]);
        addComputePass(device, commandEncoder, piplines[288], [buf_25, buf_22, buf_24], [2, 256, 1]);
        addComputePass(device, commandEncoder, piplines[289], [buf_26, buf_16, buf_25, buf_1], [16, 32, 8]);
        addComputePass(device, commandEncoder, piplines[290], [buf_27, buf_26], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[291], [buf_28, buf_26, buf_27], [256, 1, 1]);
        addComputePass(device, commandEncoder, piplines[292], [buf_29, buf_26, buf_27, buf_28], [16, 4096, 1]);
        addComputePass(device, commandEncoder, piplines[293], [buf_10, buf_29, buf_18], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[294], [buf_16, buf_3, buf_20, buf_10, buf_167], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[295], [buf_14, buf_16], [64, 1, 1]);
        addComputePass(device, commandEncoder, piplines[296], [buf_20, buf_16, buf_14, buf_168], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[297], [buf_32, buf_20, buf_169], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[298], [buf_34, buf_20, buf_170, buf_32], [128, 32, 1]);
        addComputePass(device, commandEncoder, piplines[299], [buf_171, buf_16, buf_34, buf_172], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[300], [buf_14, buf_3, buf_171], [1024, 1, 1]);
        addComputePass(device, commandEncoder, piplines[301], [buf_10, buf_3, buf_171, buf_14, buf_173], [32, 32, 1]);
        addComputePass(device, commandEncoder, piplines[302], [buf_174, buf_10, buf_175, buf_15, buf_4], [2672, 32, 1]);
        addComputePass(device, commandEncoder, piplines[303], [buf_176, buf_174], [4008, 1, 1]);
        addComputePass(device, commandEncoder, piplines[304], [buf_177, buf_176], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[305], [buf_178, buf_177], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[306], [buf_177, buf_176, buf_178], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[307], [buf_179, buf_177], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[308], [buf_180, buf_176, buf_178, buf_179], [1336, 1, 1]);
        addComputePass(device, commandEncoder, piplines[309], [buf_181, buf_180], [16, 167, 1]);
        addComputePass(device, commandEncoder, piplines[310], [buf_182, buf_181], [167, 1, 1]);
        addComputePass(device, commandEncoder, piplines[311], [buf_183, buf_9, buf_181, buf_182], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[312], [output0, buf_183], [1, 1, 1]);
            commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer, 0, output0.size);
            const gpuCommands = commandEncoder.finish();
            device.queue.submit([gpuCommands]);

            await gpuReadBuffer.mapAsync(GPUMapMode.READ);
            const resultBuffer = new Int32Array(gpuReadBuffer.size/4);
            resultBuffer.set(new Int32Array(gpuReadBuffer.getMappedRange()));
            gpuReadBuffer.unmap();
            return resultBuffer;
        }
      }
    }
  }
  