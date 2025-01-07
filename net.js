/*
const E_4_8_16_2n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
6

const E_2_8_16_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
8

const r_2_28start_pos2B129_4_8_8_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
9

const r_32_28start_pos2B129 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
10

const r_32_28start_pos2B129n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
11

const E_28start_pos2B129_8_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
12

const r_4_8_16_28start_pos2B129_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
13

*/

    window.MODEL_BASE_URL= ".";

  const getTensorBuffer = (safetensorParts, t) => {return safetensorParts[t.chunk].subarray(t.start_pos, t.start_pos + t.size)}

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

    const r_64_16_8_16_501_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<i32>;
@group(0) @binding(2)var<storage,read_write>data2:array<i32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val0 = data2[0];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 501; ridx0++) {
    var val1 = data1[((gidx0*8016)+(lidx1*501)+ridx0)];
    var alu0 = ((gidx0*16416768)+(gidx1<<5)+(lidx0<<2)+(lidx1*1026048)+(ridx0<<11));
    var val2 = data3[alu0];
    var val3 = data3[(alu0+1)];
    var val4 = data3[(alu0+2)];
    var val5 = data3[(alu0+3)];
    var cast0 = (f32(((val1!=val0)!=true)));
    acc0 = (acc0+(cast0*val2));
    acc1 = (acc1+(cast0*val3));
    acc2 = (acc2+(cast0*val4));
    acc3 = (acc3+(cast0*val5));
  }
  var alu6 = (lidx1+(gidx0<<4)+(gidx1<<13)+(lidx0<<10));
  data0[alu6] = acc0;
  data0[(alu6+256)] = acc1;
  data0[(alu6+512)] = acc2;
  data0[(alu6+768)] = acc3;
}`;

const r_2048_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 16>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 16; ridx0++) {
    var val0 = data1[((gidx0<<8)+(lidx0<<4)+ridx0)];
    acc0 = (acc0+val0);
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 16; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[gidx0] = acc1;
  }
}`;

const r_256_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 256>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var lidx0 = i32(lindex.x); /* 256 */
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < 8; ridx0++) {
    var val0 = data1[((lidx0<<3)+ridx0)];
    acc0 = (acc0+(val0*val0));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1 = 0; ridx1 < 256; ridx1++) {
      var val1 = temp0[ridx1];
      acc1 = (acc1+val1);
    }
    data0[0] = sqrt((1/((acc1*0.00048828125f)+1e-05f)));
  }
}`;

const E_16_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 32 */
  var val0 = data2[0];
  var alu0 = ((gidx0<<7)+(lidx0<<2));
  var val1 = data1[alu0];
  var val2 = data3[alu0];
  var alu1 = (alu0+1);
  var val3 = data1[alu1];
  var val4 = data3[alu1];
  var alu2 = (alu0+2);
  var val5 = data1[alu2];
  var val6 = data3[alu2];
  var alu3 = (alu0+3);
  var val7 = data1[alu3];
  var val8 = data3[alu3];
  data0[alu1] = (val4*val3*val0);
  data0[alu2] = (val6*val5*val0);
  data0[alu3] = (val8*val7*val0);
  data0[alu0] = (val2*val1*val0);
}`;

const r_32_4_8_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (lidx0<<5);
  var alu1 = (lidx0+(gidx0<<4));
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu2 = (ridx0<<3);
    var val0 = data1[(lidx1+alu2)];
    var alu3 = (lidx1+(gidx0<<15)+(lidx0<<11)+alu2);
    var val1 = data2[alu3];
    var val2 = data2[(alu3+8192)];
    var val3 = data2[(alu3+16384)];
    var val4 = data2[(alu3+24576)];
    acc0 = (acc0+(val0*val1));
    acc1 = (acc1+(val0*val2));
    acc2 = (acc2+(val0*val3));
    acc3 = (acc3+(val0*val4));
  }
  var alu9 = (alu0+(lidx1<<2));
  temp0[alu9] = acc0;
  temp0[(alu9+1)] = acc1;
  temp0[(alu9+2)] = acc2;
  temp0[(alu9+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu16 = (alu0+(ridx1<<2));
      var val5 = temp0[(alu16+1)];
      var val6 = temp0[(alu16+2)];
      var val7 = temp0[(alu16+3)];
      var val8 = temp0[alu16];
      acc4 = (acc4+val8);
      acc5 = (acc5+val5);
      acc6 = (acc6+val6);
      acc7 = (acc7+val7);
    }
    data0[alu1] = acc4;
    data0[(alu1+4)] = acc5;
    data0[(alu1+8)] = acc6;
    data0[(alu1+12)] = acc7;
  }
}`;

const E_4_8_16_2n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@group(0) @binding(4)var<uniform>start_pos:i32;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<4);
  var alu1 = (lidx0<<6);
  var alu2 = (alu0+alu1);
  var val0 = data3[(lidx1+alu2)];
  var alu3 = ((lidx1>>1)<<1);
  var alu4 = (alu2+alu3);
  var alu5 = (alu0+alu3+(start_pos<<6));
  var alu6 = ((lidx1&1)<1);
  var val1 = select(0.0f, data1[(alu4+1)], alu6);
  var val2 = select(0.0f, data1[alu4], alu6);
  var val3 = select(0.0f, data2[(alu5+1)], alu6);
  var val4 = select(0.0f, data2[alu5], alu6);
  var alu7 = (alu6!=true);
  var val5 = select(0.0f, data1[(alu4+1)], alu7);
  var val6 = select(0.0f, data1[alu4], alu7);
  var val7 = select(0.0f, data2[(alu5+1)], alu7);
  var val8 = select(0.0f, data2[alu5], alu7);
  var alu8 = (lidx1+(start_pos<<9)+alu0+alu1);
  data0[(alu8+524288)] = val0;
  data0[alu8] = ((val5*val8)+(val6*val7)+(val2*val4)+(val1*val3*select(0.0f,-1.0f,alu6)));
}`;

const r_128_4_8_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (lidx0<<5);
  var alu1 = (lidx0+(gidx0<<4));
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu2 = (ridx0<<3);
    var val0 = data1[(lidx1+alu2)];
    var alu3 = (lidx1+(gidx0<<15)+(lidx0<<11)+alu2);
    var val1 = data2[alu3];
    var val2 = data2[(alu3+8192)];
    var val3 = data2[(alu3+16384)];
    var val4 = data2[(alu3+24576)];
    acc0 = (acc0+(val0*val1));
    acc1 = (acc1+(val0*val2));
    acc2 = (acc2+(val0*val3));
    acc3 = (acc3+(val0*val4));
  }
  var alu9 = (alu0+(lidx1<<2));
  temp0[alu9] = acc0;
  temp0[(alu9+1)] = acc1;
  temp0[(alu9+2)] = acc2;
  temp0[(alu9+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu16 = (alu0+(ridx1<<2));
      var val5 = temp0[(alu16+1)];
      var val6 = temp0[(alu16+2)];
      var val7 = temp0[(alu16+3)];
      var val8 = temp0[alu16];
      acc4 = (acc4+val8);
      acc5 = (acc5+val5);
      acc6 = (acc6+val6);
      acc7 = (acc7+val7);
    }
    data0[alu1] = acc4;
    data0[(alu1+4)] = acc5;
    data0[(alu1+8)] = acc6;
    data0[(alu1+12)] = acc7;
  }
}`;

const E_2_8_16_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<uniform>start_pos:i32;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (gidx0<<5);
  var alu1 = (lidx1<<1);
  var alu2 = ((start_pos<<6)+alu0+alu1);
  var val0 = data2[alu2];
  var val1 = data2[(alu2+1)];
  var alu3 = (alu0+(lidx0<<8)+alu1);
  var val2 = data1[alu3];
  var alu4 = (alu3+1);
  var val3 = data1[alu4];
  var alu5 = (alu3+64);
  var val4 = data1[alu5];
  var alu6 = (alu3+65);
  var val5 = data1[alu6];
  var alu7 = (alu3+128);
  var val6 = data1[alu7];
  var alu8 = (alu3+129);
  var val7 = data1[alu8];
  var alu9 = (alu3+192);
  var val8 = data1[alu9];
  var alu10 = (alu3+193);
  var val9 = data1[alu10];
  data0[alu6] = ((val4*val1)+(val5*val0));
  data0[alu8] = ((val6*val1)+(val7*val0));
  data0[alu10] = ((val8*val1)+(val9*val0));
  data0[alu4] = ((val3*val0)+(val2*val1));
  data0[alu5] = ((val4*val0)-(val5*val1));
  data0[alu7] = ((val6*val0)-(val7*val1));
  data0[alu9] = ((val8*val0)-(val9*val1));
  data0[alu3] = ((val2*val0)-(val3*val1));
}`;

const r_2_28start_pos2B129_4_8_8_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<uniform>start_pos:i32;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.DEFINE_VAR, dtypes.int, arg=('start_pos', 0, 1024), src=()),
  UOp(Ops.CONST, dtypes.int, arg=1, src=()),)) */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (start_pos<<3);
  var alu1 = (lidx0<<5);
  var alu2 = (gidx0+(gidx1*((start_pos<<4)+16))+(lidx0*(start_pos+1)));
  var alu3 = (alu0+8);
  var alu4 = ((gidx0<<3)+(gidx1<<2));
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 8; ridx0++) {
    var alu5 = (ridx0<<3);
    var alu6 = (lidx1+alu5);
    var alu7 = (lidx1+(gidx1<<10)+(lidx0<<6)+alu5);
    var val0 = data1[alu7];
    var val1 = data2[(alu6+(((alu4+1)%alu3)<<6))];
    var val2 = data2[(alu6+(((alu4+2)%alu3)<<6))];
    var val3 = data2[(alu6+(((alu4+3)%alu3)<<6))];
    var val4 = data2[(alu6+((alu4%alu3)<<6))];
    var val5 = data1[(alu7+256)];
    var val6 = data1[(alu7+512)];
    var val7 = data1[(alu7+768)];
    acc0 = (acc0+(val0*val4));
    acc1 = (acc1+(val5*val1));
    acc2 = (acc2+(val6*val2));
    acc3 = (acc3+(val7*val3));
  }
  var alu13 = (alu1+(lidx1<<2));
  temp0[alu13] = acc0;
  temp0[(alu13+1)] = acc1;
  temp0[(alu13+2)] = acc2;
  temp0[(alu13+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu20 = (alu1+(ridx1<<2));
      var val8 = temp0[(alu20+1)];
      var val9 = temp0[(alu20+2)];
      var val10 = temp0[(alu20+3)];
      var val11 = temp0[alu20];
      acc4 = (acc4+val11);
      acc5 = (acc5+val8);
      acc6 = (acc6+val9);
      acc7 = (acc7+val10);
    }
    data0[alu2] = (acc4*0.125f);
    data0[(alu2+(start_pos<<2)+4)] = (acc5*0.125f);
    data0[(alu2+alu0+8)] = (acc6*0.125f);
    data0[(alu2+(start_pos*12)+12)] = (acc7*0.125f);
  }
}`;

const r_32_28start_pos2B129 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<uniform>start_pos:i32;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (start_pos+1);
  var acc0 = (f32(-inf(1.0)));
  for (var ridx0 = 0; ridx0 < alu0; ridx0++) {
    var val0 = data1[((lidx0*alu0)+ridx0)];
    acc0 = select(acc0,val0,(acc0<val0));
  }
  data0[lidx0] = acc0;
}`;

const r_32_28start_pos2B129n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<uniform>start_pos:i32;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (start_pos+1);
  var val0 = data2[lidx0];
  var acc0 = 0.0f;
  for (var ridx0 = 0; ridx0 < alu0; ridx0++) {
    var val1 = data1[((lidx0*alu0)+ridx0)];
    acc0 = (acc0+exp2(((val1-val0)*1.4426950408889634f)));
  }
  data0[lidx0] = acc0;
}`;

const E_28start_pos2B129_8_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@group(0) @binding(4)var<uniform>start_pos:i32;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.DEFINE_VAR, dtypes.int, arg=('start_pos', 0, 1024), src=()),
  UOp(Ops.CONST, dtypes.int, arg=1, src=()),)) */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (gidx0+(lidx0*((start_pos<<2)+4)));
  var val0 = data1[alu0];
  var alu1 = (start_pos+alu0+1);
  var val1 = data1[alu1];
  var alu2 = (alu0+(start_pos*3)+3);
  var val2 = data1[alu2];
  var alu3 = (alu0+(start_pos<<1)+2);
  var val3 = data1[alu3];
  var alu4 = (lidx0<<2);
  var val4 = data2[alu4];
  var val5 = data3[alu4];
  var alu5 = (alu4+1);
  var val6 = data2[alu5];
  var val7 = data3[alu5];
  var alu6 = (alu4+2);
  var val8 = data2[alu6];
  var val9 = data3[alu6];
  var alu7 = (alu4+3);
  var val10 = data2[alu7];
  var val11 = data3[alu7];
  data0[alu0] = (exp2(((val0-val4)*1.4426950408889634f))*(1/val5));
  data0[alu1] = (exp2(((val1-val6)*1.4426950408889634f))*(1/val7));
  data0[alu2] = (exp2(((val2-val10)*1.4426950408889634f))*(1/val11));
  data0[alu3] = (exp2(((val3-val8)*1.4426950408889634f))*(1/val9));
}`;

const r_4_8_16_28start_pos2B129_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<uniform>start_pos:i32;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (lidx1<<2);
  var alu1 = ((start_pos<<3)+8);
  var alu2 = (start_pos+1);
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < alu2; ridx0++) {
    var val0 = data1[((gidx0*alu1)+(lidx0*alu2)+ridx0)];
    var alu3 = (alu0+((((gidx0<<1)+(ridx0<<3)+(lidx0>>2))%alu1)<<6));
    var val1 = data2[(alu3+524288)];
    var val2 = data2[(alu3+524289)];
    var val3 = data2[(alu3+524290)];
    var val4 = data2[(alu3+524291)];
    acc0 = (acc0+(val0*val1));
    acc1 = (acc1+(val0*val2));
    acc2 = (acc2+(val0*val3));
    acc3 = (acc3+(val0*val4));
  }
  var alu9 = ((gidx0<<9)+(lidx0<<6)+alu0);
  data0[alu9] = acc0;
  data0[(alu9+1)] = acc1;
  data0[(alu9+2)] = acc2;
  data0[(alu9+3)] = acc3;
}`;

const r_128_4_8_256_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (lidx0<<5);
  var alu1 = (lidx0+(gidx0<<4));
  var val0 = data1[alu1];
  var alu2 = (alu1+4);
  var val1 = data1[alu2];
  var alu3 = (alu1+8);
  var val2 = data1[alu3];
  var alu4 = (alu1+12);
  var val3 = data1[alu4];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu5 = (ridx0<<3);
    var val4 = data2[(lidx1+alu5)];
    var alu6 = (lidx1+(gidx0<<15)+(lidx0<<11)+alu5);
    var val5 = data3[alu6];
    var val6 = data3[(alu6+8192)];
    var val7 = data3[(alu6+16384)];
    var val8 = data3[(alu6+24576)];
    acc0 = (acc0+(val4*val5));
    acc1 = (acc1+(val4*val6));
    acc2 = (acc2+(val4*val7));
    acc3 = (acc3+(val4*val8));
  }
  var alu12 = (alu0+(lidx1<<2));
  temp0[alu12] = acc0;
  temp0[(alu12+1)] = acc1;
  temp0[(alu12+2)] = acc2;
  temp0[(alu12+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu19 = (alu0+(ridx1<<2));
      var val9 = temp0[(alu19+1)];
      var val10 = temp0[(alu19+2)];
      var val11 = temp0[(alu19+3)];
      var val12 = temp0[alu19];
      acc4 = (acc4+val12);
      acc5 = (acc5+val9);
      acc6 = (acc6+val10);
      acc7 = (acc7+val11);
    }
    data0[alu1] = (val0+acc4);
    data0[alu2] = (val1+acc5);
    data0[alu3] = (val2+acc6);
    data0[alu4] = (val3+acc7);
  }
}`;

const r_512_4_8_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (lidx0<<5);
  var alu1 = (lidx0+(gidx0<<4));
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu2 = (ridx0<<3);
    var val0 = data1[(lidx1+alu2)];
    var alu3 = (lidx1+(gidx0<<15)+(lidx0<<11)+alu2);
    var val1 = data2[alu3];
    var val2 = data2[(alu3+8192)];
    var val3 = data2[(alu3+16384)];
    var val4 = data2[(alu3+24576)];
    acc0 = (acc0+(val0*val1));
    acc1 = (acc1+(val0*val2));
    acc2 = (acc2+(val0*val3));
    acc3 = (acc3+(val0*val4));
  }
  var alu9 = (alu0+(lidx1<<2));
  temp0[alu9] = acc0;
  temp0[(alu9+1)] = acc1;
  temp0[(alu9+2)] = acc2;
  temp0[(alu9+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu16 = (alu0+(ridx1<<2));
      var val5 = temp0[(alu16+1)];
      var val6 = temp0[(alu16+2)];
      var val7 = temp0[(alu16+3)];
      var val8 = temp0[alu16];
      acc4 = (acc4+val8);
      acc5 = (acc5+val5);
      acc6 = (acc6+val6);
      acc7 = (acc7+val7);
    }
    data0[alu1] = acc4;
    data0[(alu1+4)] = acc5;
    data0[(alu1+8)] = acc6;
    data0[(alu1+12)] = acc7;
  }
}`;

const r_512_4_8_256_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (lidx0<<5);
  var alu1 = (lidx0+(gidx0<<4));
  var val0 = data3[alu1];
  var alu2 = (alu1+4);
  var val1 = data3[alu2];
  var alu3 = (alu1+8);
  var val2 = data3[alu3];
  var alu4 = (alu1+12);
  var val3 = data3[alu4];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu5 = (ridx0<<3);
    var val4 = data1[(lidx1+alu5)];
    var alu6 = (lidx1+(gidx0<<15)+(lidx0<<11)+alu5);
    var val5 = data2[alu6];
    var val6 = data2[(alu6+8192)];
    var val7 = data2[(alu6+16384)];
    var val8 = data2[(alu6+24576)];
    acc0 = (acc0+(val4*val5));
    acc1 = (acc1+(val4*val6));
    acc2 = (acc2+(val4*val7));
    acc3 = (acc3+(val4*val8));
  }
  var alu12 = (alu0+(lidx1<<2));
  temp0[alu12] = acc0;
  temp0[(alu12+1)] = acc1;
  temp0[(alu12+2)] = acc2;
  temp0[(alu12+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu19 = (alu0+(ridx1<<2));
      var val9 = temp0[(alu19+1)];
      var val10 = temp0[(alu19+2)];
      var val11 = temp0[(alu19+3)];
      var val12 = temp0[alu19];
      acc4 = (acc4+val12);
      acc5 = (acc5+val9);
      acc6 = (acc6+val10);
      acc7 = (acc7+val11);
    }
    data0[alu1] = (val0*(1/(exp2((acc4*-1.4426950408889634f))+1.0f))*acc4);
    data0[alu2] = (val1*(1/(exp2((acc5*-1.4426950408889634f))+1.0f))*acc5);
    data0[alu3] = (val2*(1/(exp2((acc6*-1.4426950408889634f))+1.0f))*acc6);
    data0[alu4] = (val3*(1/(exp2((acc7*-1.4426950408889634f))+1.0f))*acc7);
  }
}`;

const r_128_4_8_1024_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@group(0) @binding(3)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (lidx0<<5);
  var alu1 = (lidx0+(gidx0<<4));
  var val0 = data1[alu1];
  var alu2 = (alu1+4);
  var val1 = data1[alu2];
  var alu3 = (alu1+8);
  var val2 = data1[alu3];
  var alu4 = (alu1+12);
  var val3 = data1[alu4];
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 1024; ridx0++) {
    var alu5 = (ridx0<<3);
    var val4 = data2[(lidx1+alu5)];
    var alu6 = (lidx1+(gidx0<<17)+(lidx0<<13)+alu5);
    var val5 = data3[alu6];
    var val6 = data3[(alu6+32768)];
    var val7 = data3[(alu6+65536)];
    var val8 = data3[(alu6+98304)];
    acc0 = (acc0+(val4*val5));
    acc1 = (acc1+(val4*val6));
    acc2 = (acc2+(val4*val7));
    acc3 = (acc3+(val4*val8));
  }
  var alu12 = (alu0+(lidx1<<2));
  temp0[alu12] = acc0;
  temp0[(alu12+1)] = acc1;
  temp0[(alu12+2)] = acc2;
  temp0[(alu12+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu19 = (alu0+(ridx1<<2));
      var val9 = temp0[(alu19+1)];
      var val10 = temp0[(alu19+2)];
      var val11 = temp0[(alu19+3)];
      var val12 = temp0[alu19];
      acc4 = (acc4+val12);
      acc5 = (acc5+val9);
      acc6 = (acc6+val10);
      acc7 = (acc7+val11);
    }
    data0[alu1] = (val0+acc4);
    data0[alu2] = (val1+acc5);
    data0[alu3] = (val2+acc6);
    data0[alu4] = (val3+acc7);
  }
}`;

const E_n8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
@group(0) @binding(0)var<storage,read_write>data0:array<u32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var val0 = data0[0];
  data0[0] = (val0+1u);
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

const r_8016_4_8_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }

fn inf(a: f32) -> f32 { return a/0.0; }
var<workgroup> temp0: array<f32, 128>;
@group(0) @binding(0)var<storage,read_write>data0:array<f32>;
@group(0) @binding(1)var<storage,read_write>data1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data2:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8016 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (lidx0<<5);
  var alu1 = (lidx0+(gidx0<<4));
  var acc0 = 0.0f;
  var acc1 = 0.0f;
  var acc2 = 0.0f;
  var acc3 = 0.0f;
  for (var ridx0 = 0; ridx0 < 256; ridx0++) {
    var alu2 = (ridx0<<3);
    var val0 = data1[(lidx1+alu2)];
    var alu3 = (lidx1+(gidx0<<15)+(lidx0<<11)+alu2);
    var val1 = data2[alu3];
    var val2 = data2[(alu3+8192)];
    var val3 = data2[(alu3+16384)];
    var val4 = data2[(alu3+24576)];
    acc0 = (acc0+(val0*val1));
    acc1 = (acc1+(val0*val2));
    acc2 = (acc2+(val0*val3));
    acc3 = (acc3+(val0*val4));
  }
  var alu9 = (alu0+(lidx1<<2));
  temp0[alu9] = acc0;
  temp0[(alu9+1)] = acc1;
  temp0[(alu9+2)] = acc2;
  temp0[(alu9+3)] = acc3;
  workgroupBarrier();
  if (((bool(lidx1))!=true)) {
    var acc4 = 0.0f;
    var acc5 = 0.0f;
    var acc6 = 0.0f;
    var acc7 = 0.0f;
    for (var ridx1 = 0; ridx1 < 8; ridx1++) {
      var alu16 = (alu0+(ridx1<<2));
      var val5 = temp0[(alu16+1)];
      var val6 = temp0[(alu16+2)];
      var val7 = temp0[(alu16+3)];
      var val8 = temp0[alu16];
      acc4 = (acc4+val8);
      acc5 = (acc5+val5);
      acc6 = (acc6+val6);
      acc7 = (acc7+val7);
    }
    data0[alu1] = acc4;
    data0[(alu1+4)] = acc5;
    data0[(alu1+8)] = acc6;
    data0[(alu1+12)] = acc7;
  }
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
      "setup": async (device, safetensor, metadata) => {

        const buf_0 = createEmptyBuf(device, 2097152);;
    const buf_1 = createWeightBuf(device, 513024, getTensorBuffer(safetensor, metadata['tok_embeddings.arange']));
    const input0 = createEmptyBuf(device, 4);;
    const start_pos = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });


    const buf_2 = createWeightBuf(device, 1050673152, getTensorBuffer(safetensor, metadata['tok_embeddings.weight']));
    const buf_3 = createEmptyBuf(device, 8192);;
    const buf_4 = createEmptyBuf(device, 4);;
    const buf_5 = createEmptyBuf(device, 8192);;
    const buf_6 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.0.attention_norm.weight']));
    const buf_7 = createEmptyBuf(device, 2048);;
    const buf_8 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.0.attention.wk.weight']));
    const buf_9 = createEmptyBuf(device, 2048);;
    const buf_10 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.0.attention.wv.weight']));
    const buf_11 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.0.attention.cache_kv']));
    const buf_12 = createWeightBuf(device, 524288, getTensorBuffer(safetensor, metadata['freqs_cis']));
    const buf_13 = createEmptyBuf(device, 8192);;
    const buf_14 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.0.attention.wq.weight']));
    const buf_15 = createEmptyBuf(device, 8192);;
    const buf_16 = createEmptyBuf(device, 131200);;
    const buf_17 = createEmptyBuf(device, 128);;
    const buf_18 = createEmptyBuf(device, 128);;
    const buf_19 = createEmptyBuf(device, 131200);;
    const buf_20 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.0.attention.wo.weight']));
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_21 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.0.ffn_norm.weight']));
    const buf_22 = createEmptyBuf(device, 32768);;
    const buf_23 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.0.feed_forward.w3.weight']));
    const buf_24 = createEmptyBuf(device, 32768);;
    const buf_25 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.0.feed_forward.w1.weight']));
    const buf_26 = createEmptyBuf(device, 8192);;
    const buf_27 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.0.feed_forward.w2.weight']));
    const buf_28 = createEmptyBuf(device, 8192);;
    const buf_29 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.1.attention_norm.weight']));
    const buf_30 = createEmptyBuf(device, 2048);;
    const buf_31 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.1.attention.wk.weight']));
    const buf_32 = createEmptyBuf(device, 2048);;
    const buf_33 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.1.attention.wv.weight']));
    const buf_34 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.1.attention.cache_kv']));
    const buf_35 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.1.attention.wq.weight']));
    const buf_36 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.1.attention.wo.weight']));
    const buf_37 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.1.ffn_norm.weight']));
    const buf_38 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.1.feed_forward.w3.weight']));
    const buf_39 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.1.feed_forward.w1.weight']));
    const buf_40 = createEmptyBuf(device, 8192);;
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_41 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.1.feed_forward.w2.weight']));
    const buf_42 = createEmptyBuf(device, 8192);;
    const buf_43 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.2.attention_norm.weight']));
    const buf_44 = createEmptyBuf(device, 2048);;
    const buf_45 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.2.attention.wk.weight']));
    const buf_46 = createEmptyBuf(device, 2048);;
    const buf_47 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.2.attention.wv.weight']));
    const buf_48 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.2.attention.cache_kv']));
    const buf_49 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.2.attention.wq.weight']));
    const buf_50 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.2.attention.wo.weight']));
    const buf_51 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.2.ffn_norm.weight']));
    const buf_52 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.2.feed_forward.w3.weight']));
    const buf_53 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.2.feed_forward.w1.weight']));
    const buf_54 = createEmptyBuf(device, 8192);;
    const buf_55 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.2.feed_forward.w2.weight']));
    const buf_56 = createEmptyBuf(device, 8192);;
    const buf_57 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.3.attention_norm.weight']));
    const buf_58 = createEmptyBuf(device, 2048);;
    const buf_59 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.3.attention.wk.weight']));
    const buf_60 = createEmptyBuf(device, 2048);;
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_61 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.3.attention.wv.weight']));
    const buf_62 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.3.attention.cache_kv']));
    const buf_63 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.3.attention.wq.weight']));
    const buf_64 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.3.attention.wo.weight']));
    const buf_65 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.3.ffn_norm.weight']));
    const buf_66 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.3.feed_forward.w3.weight']));
    const buf_67 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.3.feed_forward.w1.weight']));
    const buf_68 = createEmptyBuf(device, 8192);;
    const buf_69 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.3.feed_forward.w2.weight']));
    const buf_70 = createEmptyBuf(device, 8192);;
    const buf_71 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.4.attention_norm.weight']));
    const buf_72 = createEmptyBuf(device, 2048);;
    const buf_73 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.4.attention.wk.weight']));
    const buf_74 = createEmptyBuf(device, 2048);;
    const buf_75 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.4.attention.wv.weight']));
    const buf_76 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.4.attention.cache_kv']));
    const buf_77 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.4.attention.wq.weight']));
    const buf_78 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.4.attention.wo.weight']));
    const buf_79 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.4.ffn_norm.weight']));
    const buf_80 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.4.feed_forward.w3.weight']));
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_81 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.4.feed_forward.w1.weight']));
    const buf_82 = createEmptyBuf(device, 8192);;
    const buf_83 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.4.feed_forward.w2.weight']));
    const buf_84 = createEmptyBuf(device, 8192);;
    const buf_85 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.5.attention_norm.weight']));
    const buf_86 = createEmptyBuf(device, 2048);;
    const buf_87 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.5.attention.wk.weight']));
    const buf_88 = createEmptyBuf(device, 2048);;
    const buf_89 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.5.attention.wv.weight']));
    const buf_90 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.5.attention.cache_kv']));
    const buf_91 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.5.attention.wq.weight']));
    const buf_92 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.5.attention.wo.weight']));
    const buf_93 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.5.ffn_norm.weight']));
    const buf_94 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.5.feed_forward.w3.weight']));
    const buf_95 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.5.feed_forward.w1.weight']));
    const buf_96 = createEmptyBuf(device, 8192);;
    const buf_97 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.5.feed_forward.w2.weight']));
    const buf_98 = createEmptyBuf(device, 8192);;
    const buf_99 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.6.attention_norm.weight']));
    const buf_100 = createEmptyBuf(device, 2048);;
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_101 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.6.attention.wk.weight']));
    const buf_102 = createEmptyBuf(device, 2048);;
    const buf_103 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.6.attention.wv.weight']));
    const buf_104 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.6.attention.cache_kv']));
    const buf_105 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.6.attention.wq.weight']));
    const buf_106 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.6.attention.wo.weight']));
    const buf_107 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.6.ffn_norm.weight']));
    const buf_108 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.6.feed_forward.w3.weight']));
    const buf_109 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.6.feed_forward.w1.weight']));
    const buf_110 = createEmptyBuf(device, 8192);;
    const buf_111 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.6.feed_forward.w2.weight']));
    const buf_112 = createEmptyBuf(device, 8192);;
    const buf_113 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.7.attention_norm.weight']));
    const buf_114 = createEmptyBuf(device, 2048);;
    const buf_115 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.7.attention.wk.weight']));
    const buf_116 = createEmptyBuf(device, 2048);;
    const buf_117 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.7.attention.wv.weight']));
    const buf_118 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.7.attention.cache_kv']));
    const buf_119 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.7.attention.wq.weight']));
    const buf_120 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.7.attention.wo.weight']));
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_121 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.7.ffn_norm.weight']));
    const buf_122 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.7.feed_forward.w3.weight']));
    const buf_123 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.7.feed_forward.w1.weight']));
    const buf_124 = createEmptyBuf(device, 8192);;
    const buf_125 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.7.feed_forward.w2.weight']));
    const buf_126 = createEmptyBuf(device, 8192);;
    const buf_127 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.8.attention_norm.weight']));
    const buf_128 = createEmptyBuf(device, 2048);;
    const buf_129 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.8.attention.wk.weight']));
    const buf_130 = createEmptyBuf(device, 2048);;
    const buf_131 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.8.attention.wv.weight']));
    const buf_132 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.8.attention.cache_kv']));
    const buf_133 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.8.attention.wq.weight']));
    const buf_134 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.8.attention.wo.weight']));
    const buf_135 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.8.ffn_norm.weight']));
    const buf_136 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.8.feed_forward.w3.weight']));
    const buf_137 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.8.feed_forward.w1.weight']));
    const buf_138 = createEmptyBuf(device, 8192);;
    const buf_139 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.8.feed_forward.w2.weight']));
    const buf_140 = createEmptyBuf(device, 8192);;
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_141 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.9.attention_norm.weight']));
    const buf_142 = createEmptyBuf(device, 2048);;
    const buf_143 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.9.attention.wk.weight']));
    const buf_144 = createEmptyBuf(device, 2048);;
    const buf_145 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.9.attention.wv.weight']));
    const buf_146 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.9.attention.cache_kv']));
    const buf_147 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.9.attention.wq.weight']));
    const buf_148 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.9.attention.wo.weight']));
    const buf_149 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.9.ffn_norm.weight']));
    const buf_150 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.9.feed_forward.w3.weight']));
    const buf_151 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.9.feed_forward.w1.weight']));
    const buf_152 = createEmptyBuf(device, 8192);;
    const buf_153 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.9.feed_forward.w2.weight']));
    const buf_154 = createEmptyBuf(device, 8192);;
    const buf_155 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.10.attention_norm.weight']));
    const buf_156 = createEmptyBuf(device, 2048);;
    const buf_157 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.10.attention.wk.weight']));
    const buf_158 = createEmptyBuf(device, 2048);;
    const buf_159 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.10.attention.wv.weight']));
    const buf_160 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.10.attention.cache_kv']));
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_161 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.10.attention.wq.weight']));
    const buf_162 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.10.attention.wo.weight']));
    const buf_163 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.10.ffn_norm.weight']));
    const buf_164 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.10.feed_forward.w3.weight']));
    const buf_165 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.10.feed_forward.w1.weight']));
    const buf_166 = createEmptyBuf(device, 8192);;
    const buf_167 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.10.feed_forward.w2.weight']));
    const buf_168 = createEmptyBuf(device, 8192);;
    const buf_169 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.11.attention_norm.weight']));
    const buf_170 = createEmptyBuf(device, 2048);;
    const buf_171 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.11.attention.wk.weight']));
    const buf_172 = createEmptyBuf(device, 2048);;
    const buf_173 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.11.attention.wv.weight']));
    const buf_174 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.11.attention.cache_kv']));
    const buf_175 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.11.attention.wq.weight']));
    const buf_176 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.11.attention.wo.weight']));
    const buf_177 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.11.ffn_norm.weight']));
    const buf_178 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.11.feed_forward.w3.weight']));
    const buf_179 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.11.feed_forward.w1.weight']));
    const buf_180 = createEmptyBuf(device, 8192);;
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_181 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.11.feed_forward.w2.weight']));
    const buf_182 = createEmptyBuf(device, 8192);;
    const buf_183 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.12.attention_norm.weight']));
    const buf_184 = createEmptyBuf(device, 2048);;
    const buf_185 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.12.attention.wk.weight']));
    const buf_186 = createEmptyBuf(device, 2048);;
    const buf_187 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.12.attention.wv.weight']));
    const buf_188 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.12.attention.cache_kv']));
    const buf_189 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.12.attention.wq.weight']));
    const buf_190 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.12.attention.wo.weight']));
    const buf_191 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.12.ffn_norm.weight']));
    const buf_192 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.12.feed_forward.w3.weight']));
    const buf_193 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.12.feed_forward.w1.weight']));
    const buf_194 = createEmptyBuf(device, 8192);;
    const buf_195 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.12.feed_forward.w2.weight']));
    const buf_196 = createEmptyBuf(device, 8192);;
    const buf_197 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.13.attention_norm.weight']));
    const buf_198 = createEmptyBuf(device, 2048);;
    const buf_199 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.13.attention.wk.weight']));
    const buf_200 = createEmptyBuf(device, 2048);;
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_201 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.13.attention.wv.weight']));
    const buf_202 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.13.attention.cache_kv']));
    const buf_203 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.13.attention.wq.weight']));
    const buf_204 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.13.attention.wo.weight']));
    const buf_205 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.13.ffn_norm.weight']));
    const buf_206 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.13.feed_forward.w3.weight']));
    const buf_207 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.13.feed_forward.w1.weight']));
    const buf_208 = createEmptyBuf(device, 8192);;
    const buf_209 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.13.feed_forward.w2.weight']));
    const buf_210 = createEmptyBuf(device, 8192);;
    const buf_211 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.14.attention_norm.weight']));
    const buf_212 = createEmptyBuf(device, 2048);;
    const buf_213 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.14.attention.wk.weight']));
    const buf_214 = createEmptyBuf(device, 2048);;
    const buf_215 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.14.attention.wv.weight']));
    const buf_216 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.14.attention.cache_kv']));
    const buf_217 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.14.attention.wq.weight']));
    const buf_218 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.14.attention.wo.weight']));
    const buf_219 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.14.ffn_norm.weight']));
    const buf_220 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.14.feed_forward.w3.weight']));
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_221 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.14.feed_forward.w1.weight']));
    const buf_222 = createEmptyBuf(device, 8192);;
    const buf_223 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.14.feed_forward.w2.weight']));
    const buf_224 = createEmptyBuf(device, 8192);;
    const buf_225 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.15.attention_norm.weight']));
    const buf_226 = createEmptyBuf(device, 2048);;
    const buf_227 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.15.attention.wk.weight']));
    const buf_228 = createEmptyBuf(device, 2048);;
    const buf_229 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.15.attention.wv.weight']));
    const buf_230 = createWeightBuf(device, 4194304, getTensorBuffer(safetensor, metadata['layers.15.attention.cache_kv']));
    const buf_231 = createEmptyBuf(device, 4);;
    const buf_232 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.15.attention.wq.weight']));
    const buf_233 = createEmptyBuf(device, 4);;
    const buf_234 = createEmptyBuf(device, 8);;
    const buf_235 = createWeightBuf(device, 16777216, getTensorBuffer(safetensor, metadata['layers.15.attention.wo.weight']));
    const buf_236 = createEmptyBuf(device, 4);;
    const buf_237 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['layers.15.ffn_norm.weight']));
    const buf_238 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.15.feed_forward.w3.weight']));
    const buf_239 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.15.feed_forward.w1.weight']));
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    const buf_240 = createEmptyBuf(device, 8192);;
    const buf_241 = createWeightBuf(device, 67108864, getTensorBuffer(safetensor, metadata['layers.15.feed_forward.w2.weight']));
    const buf_242 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['norm.weight']));
    const buf_243 = createEmptyBuf(device, 513024);;
    const buf_244 = createWeightBuf(device, 1050673152, getTensorBuffer(safetensor, metadata['output.weight']));
    const buf_245 = createEmptyBuf(device, 1024);;
    const buf_246 = createEmptyBuf(device, 4);;
    const buf_247 = createEmptyBuf(device, 513024);;
    const buf_248 = createEmptyBuf(device, 513024);;
    const buf_249 = createEmptyBuf(device, 2004);;
    const buf_250 = createEmptyBuf(device, 1024);;
    const output0 = createEmptyBuf(device, 4);;
    await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag

        const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
        const gpuWriteBuffer1 = device.createBuffer({size:4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
        const gpuReadBuffer = device.createBuffer({ size: output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        const kernels = [r_64_16_8_16_501_4, r_2048_16_16, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, r_128_4_8_256_4, E_2_8_16_2_4, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_32_4_8_256_4, r_32_4_8_256_4, E_4_8_16_2n1, E_n8, r_128_4_8_256_4, E_n5, E_2_8_16_2_4, E_n6, r_2_28start_pos2B129_4_8_8_4, r_32_28start_pos2B129, r_32_28start_pos2B129n1, E_28start_pos2B129_8_4, r_4_8_16_28start_pos2B129_4, r_128_4_8_256_4n1, r_256_8, E_16_32_4, r_512_4_8_256_4, r_512_4_8_256_4n1, r_128_4_8_1024_4, r_256_8, E_16_32_4, r_8016_4_8_256_4, r_8_32_501, r_256, r_8_32_501n1, r_256n1, E_1336_32_3, r_167_16_3_16_64_4, r_167_3_501, r_8_32_501n2, r_256n2];
        const piplines = [];
        for (let i=0; i<kernels.length; i++) {
          const name = kernels[i];
          const pipeline = await device.createComputePipelineAsync({layout: "auto", compute: { module: device.createShaderModule({ code: name }), entryPoint: "main" }});
          piplines.push(pipeline);
          if (i % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
        }

        //const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({layout: "auto", compute: { module: device.createShaderModule({ code: name }), entryPoint: "main" }})));

        return async (data0, start_pos_input) => {
            const commandEncoder = device.createCommandEncoder();

            await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
    new Int32Array(gpuWriteBuffer0.getMappedRange()).set(data0);
    gpuWriteBuffer0.unmap();
commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);

            const start_pos_buf = new Int32Array([start_pos_input])
            await gpuWriteBuffer1.mapAsync(GPUMapMode.WRITE);
    new Int32Array(gpuWriteBuffer1.getMappedRange()).set(start_pos_buf);
    gpuWriteBuffer1.unmap();
commandEncoder.copyBufferToBuffer(gpuWriteBuffer1, 0, start_pos, 0, gpuWriteBuffer1.size);

            addComputePass(device, commandEncoder, piplines[0], [buf_0, buf_1, input0, buf_2], [16, 64, 1]);
        addComputePass(device, commandEncoder, piplines[1], [buf_3, buf_0], [2048, 1, 1]);
        addComputePass(device, commandEncoder, piplines[2], [buf_4, buf_3], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[3], [buf_5, buf_3, buf_4, buf_6], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[4], [buf_7, buf_5, buf_8], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[5], [buf_9, buf_5, buf_10], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[6], [buf_11, buf_7, buf_12, buf_9, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[7], [buf_13, buf_5, buf_14], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[8], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[9], [buf_16, buf_15, buf_11, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[10], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[11], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[12], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[13], [buf_13, buf_19, buf_11, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[14], [buf_15, buf_3, buf_13, buf_20], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[15], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[16], [buf_13, buf_15, buf_4, buf_21], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[17], [buf_22, buf_13, buf_23], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[18], [buf_24, buf_13, buf_25, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[19], [buf_26, buf_15, buf_24, buf_27], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[20], [buf_4, buf_26], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[21], [buf_28, buf_26, buf_4, buf_29], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[22], [buf_30, buf_28, buf_31], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[23], [buf_32, buf_28, buf_33], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[24], [buf_34, buf_30, buf_12, buf_32, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[25], [buf_13, buf_28, buf_35], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[26], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[27], [buf_16, buf_15, buf_34, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[28], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[29], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[30], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[31], [buf_13, buf_19, buf_34, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[32], [buf_15, buf_26, buf_13, buf_36], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[33], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[34], [buf_13, buf_15, buf_4, buf_37], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[35], [buf_22, buf_13, buf_38], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[36], [buf_24, buf_13, buf_39, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[37], [buf_40, buf_15, buf_24, buf_41], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[38], [buf_4, buf_40], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[39], [buf_42, buf_40, buf_4, buf_43], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[40], [buf_44, buf_42, buf_45], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[41], [buf_46, buf_42, buf_47], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[42], [buf_48, buf_44, buf_12, buf_46, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[43], [buf_13, buf_42, buf_49], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[44], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[45], [buf_16, buf_15, buf_48, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[46], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[47], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[48], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[49], [buf_13, buf_19, buf_48, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[50], [buf_15, buf_40, buf_13, buf_50], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[51], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[52], [buf_13, buf_15, buf_4, buf_51], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[53], [buf_22, buf_13, buf_52], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[54], [buf_24, buf_13, buf_53, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[55], [buf_54, buf_15, buf_24, buf_55], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[56], [buf_4, buf_54], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[57], [buf_56, buf_54, buf_4, buf_57], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[58], [buf_58, buf_56, buf_59], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[59], [buf_60, buf_56, buf_61], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[60], [buf_62, buf_58, buf_12, buf_60, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[61], [buf_13, buf_56, buf_63], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[62], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[63], [buf_16, buf_15, buf_62, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[64], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[65], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[66], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[67], [buf_13, buf_19, buf_62, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[68], [buf_15, buf_54, buf_13, buf_64], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[69], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[70], [buf_13, buf_15, buf_4, buf_65], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[71], [buf_22, buf_13, buf_66], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[72], [buf_24, buf_13, buf_67, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[73], [buf_68, buf_15, buf_24, buf_69], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[74], [buf_4, buf_68], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[75], [buf_70, buf_68, buf_4, buf_71], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[76], [buf_72, buf_70, buf_73], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[77], [buf_74, buf_70, buf_75], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[78], [buf_76, buf_72, buf_12, buf_74, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[79], [buf_13, buf_70, buf_77], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[80], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[81], [buf_16, buf_15, buf_76, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[82], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[83], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[84], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[85], [buf_13, buf_19, buf_76, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[86], [buf_15, buf_68, buf_13, buf_78], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[87], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[88], [buf_13, buf_15, buf_4, buf_79], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[89], [buf_22, buf_13, buf_80], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[90], [buf_24, buf_13, buf_81, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[91], [buf_82, buf_15, buf_24, buf_83], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[92], [buf_4, buf_82], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[93], [buf_84, buf_82, buf_4, buf_85], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[94], [buf_86, buf_84, buf_87], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[95], [buf_88, buf_84, buf_89], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[96], [buf_90, buf_86, buf_12, buf_88, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[97], [buf_13, buf_84, buf_91], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[98], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[99], [buf_16, buf_15, buf_90, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[100], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[101], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[102], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[103], [buf_13, buf_19, buf_90, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[104], [buf_15, buf_82, buf_13, buf_92], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[105], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[106], [buf_13, buf_15, buf_4, buf_93], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[107], [buf_22, buf_13, buf_94], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[108], [buf_24, buf_13, buf_95, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[109], [buf_96, buf_15, buf_24, buf_97], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[110], [buf_4, buf_96], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[111], [buf_98, buf_96, buf_4, buf_99], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[112], [buf_100, buf_98, buf_101], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[113], [buf_102, buf_98, buf_103], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[114], [buf_104, buf_100, buf_12, buf_102, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[115], [buf_13, buf_98, buf_105], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[116], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[117], [buf_16, buf_15, buf_104, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[118], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[119], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[120], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[121], [buf_13, buf_19, buf_104, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[122], [buf_15, buf_96, buf_13, buf_106], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[123], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[124], [buf_13, buf_15, buf_4, buf_107], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[125], [buf_22, buf_13, buf_108], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[126], [buf_24, buf_13, buf_109, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[127], [buf_110, buf_15, buf_24, buf_111], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[128], [buf_4, buf_110], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[129], [buf_112, buf_110, buf_4, buf_113], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[130], [buf_114, buf_112, buf_115], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[131], [buf_116, buf_112, buf_117], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[132], [buf_118, buf_114, buf_12, buf_116, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[133], [buf_13, buf_112, buf_119], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[134], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[135], [buf_16, buf_15, buf_118, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[136], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[137], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[138], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[139], [buf_13, buf_19, buf_118, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[140], [buf_15, buf_110, buf_13, buf_120], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[141], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[142], [buf_13, buf_15, buf_4, buf_121], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[143], [buf_22, buf_13, buf_122], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[144], [buf_24, buf_13, buf_123, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[145], [buf_124, buf_15, buf_24, buf_125], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[146], [buf_4, buf_124], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[147], [buf_126, buf_124, buf_4, buf_127], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[148], [buf_128, buf_126, buf_129], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[149], [buf_130, buf_126, buf_131], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[150], [buf_132, buf_128, buf_12, buf_130, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[151], [buf_13, buf_126, buf_133], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[152], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[153], [buf_16, buf_15, buf_132, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[154], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[155], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[156], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[157], [buf_13, buf_19, buf_132, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[158], [buf_15, buf_124, buf_13, buf_134], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[159], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[160], [buf_13, buf_15, buf_4, buf_135], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[161], [buf_22, buf_13, buf_136], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[162], [buf_24, buf_13, buf_137, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[163], [buf_138, buf_15, buf_24, buf_139], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[164], [buf_4, buf_138], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[165], [buf_140, buf_138, buf_4, buf_141], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[166], [buf_142, buf_140, buf_143], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[167], [buf_144, buf_140, buf_145], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[168], [buf_146, buf_142, buf_12, buf_144, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[169], [buf_13, buf_140, buf_147], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[170], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[171], [buf_16, buf_15, buf_146, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[172], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[173], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[174], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[175], [buf_13, buf_19, buf_146, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[176], [buf_15, buf_138, buf_13, buf_148], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[177], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[178], [buf_13, buf_15, buf_4, buf_149], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[179], [buf_22, buf_13, buf_150], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[180], [buf_24, buf_13, buf_151, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[181], [buf_152, buf_15, buf_24, buf_153], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[182], [buf_4, buf_152], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[183], [buf_154, buf_152, buf_4, buf_155], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[184], [buf_156, buf_154, buf_157], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[185], [buf_158, buf_154, buf_159], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[186], [buf_160, buf_156, buf_12, buf_158, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[187], [buf_13, buf_154, buf_161], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[188], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[189], [buf_16, buf_15, buf_160, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[190], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[191], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[192], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[193], [buf_13, buf_19, buf_160, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[194], [buf_15, buf_152, buf_13, buf_162], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[195], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[196], [buf_13, buf_15, buf_4, buf_163], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[197], [buf_22, buf_13, buf_164], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[198], [buf_24, buf_13, buf_165, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[199], [buf_166, buf_15, buf_24, buf_167], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[200], [buf_4, buf_166], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[201], [buf_168, buf_166, buf_4, buf_169], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[202], [buf_170, buf_168, buf_171], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[203], [buf_172, buf_168, buf_173], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[204], [buf_174, buf_170, buf_12, buf_172, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[205], [buf_13, buf_168, buf_175], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[206], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[207], [buf_16, buf_15, buf_174, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[208], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[209], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[210], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[211], [buf_13, buf_19, buf_174, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[212], [buf_15, buf_166, buf_13, buf_176], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[213], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[214], [buf_13, buf_15, buf_4, buf_177], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[215], [buf_22, buf_13, buf_178], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[216], [buf_24, buf_13, buf_179, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[217], [buf_180, buf_15, buf_24, buf_181], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[218], [buf_4, buf_180], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[219], [buf_182, buf_180, buf_4, buf_183], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[220], [buf_184, buf_182, buf_185], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[221], [buf_186, buf_182, buf_187], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[222], [buf_188, buf_184, buf_12, buf_186, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[223], [buf_13, buf_182, buf_189], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[224], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[225], [buf_16, buf_15, buf_188, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[226], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[227], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[228], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[229], [buf_13, buf_19, buf_188, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[230], [buf_15, buf_180, buf_13, buf_190], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[231], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[232], [buf_13, buf_15, buf_4, buf_191], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[233], [buf_22, buf_13, buf_192], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[234], [buf_24, buf_13, buf_193, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[235], [buf_194, buf_15, buf_24, buf_195], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[236], [buf_4, buf_194], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[237], [buf_196, buf_194, buf_4, buf_197], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[238], [buf_198, buf_196, buf_199], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[239], [buf_200, buf_196, buf_201], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[240], [buf_202, buf_198, buf_12, buf_200, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[241], [buf_13, buf_196, buf_203], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[242], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[243], [buf_16, buf_15, buf_202, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[244], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[245], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[246], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[247], [buf_13, buf_19, buf_202, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[248], [buf_15, buf_194, buf_13, buf_204], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[249], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[250], [buf_13, buf_15, buf_4, buf_205], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[251], [buf_22, buf_13, buf_206], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[252], [buf_24, buf_13, buf_207, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[253], [buf_208, buf_15, buf_24, buf_209], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[254], [buf_4, buf_208], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[255], [buf_210, buf_208, buf_4, buf_211], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[256], [buf_212, buf_210, buf_213], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[257], [buf_214, buf_210, buf_215], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[258], [buf_216, buf_212, buf_12, buf_214, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[259], [buf_13, buf_210, buf_217], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[260], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[261], [buf_16, buf_15, buf_216, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[262], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[263], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[264], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[265], [buf_13, buf_19, buf_216, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[266], [buf_15, buf_208, buf_13, buf_218], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[267], [buf_4, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[268], [buf_13, buf_15, buf_4, buf_219], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[269], [buf_22, buf_13, buf_220], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[270], [buf_24, buf_13, buf_221, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[271], [buf_222, buf_15, buf_24, buf_223], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[272], [buf_4, buf_222], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[273], [buf_224, buf_222, buf_4, buf_225], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[274], [buf_226, buf_224, buf_227], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[275], [buf_228, buf_224, buf_229], [32, 1, 1]);
        addComputePass(device, commandEncoder, piplines[276], [buf_230, buf_226, buf_12, buf_228, start_pos], [4, 1, 1]); // double check from here after
        addComputePass(device, commandEncoder, piplines[277], [buf_231], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[278], [buf_13, buf_224, buf_232], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[279], [buf_233, buf_231, buf_234], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[280], [buf_15, buf_13, buf_12, start_pos], [2, 1, 1]);
        addComputePass(device, commandEncoder, piplines[281], [buf_4, buf_231, buf_234], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[282], [buf_16, buf_15, buf_230, start_pos], [start_pos_input + 1, 2, 1]);
        addComputePass(device, commandEncoder, piplines[283], [buf_17, buf_16, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[284], [buf_18, buf_16, buf_17, start_pos], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[285], [buf_19, buf_16, buf_17, buf_18, start_pos], [start_pos_input + 1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[286], [buf_13, buf_19, buf_230, start_pos], [4, 1, 1]);
        addComputePass(device, commandEncoder, piplines[287], [buf_15, buf_222, buf_13, buf_235], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[288], [buf_236, buf_15], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[289], [buf_13, buf_15, buf_236, buf_237], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[290], [buf_22, buf_13, buf_238], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[291], [buf_24, buf_13, buf_239, buf_22], [512, 1, 1]);
        addComputePass(device, commandEncoder, piplines[292], [buf_240, buf_15, buf_24, buf_241], [128, 1, 1]);
        addComputePass(device, commandEncoder, piplines[293], [buf_236, buf_240], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[294], [buf_15, buf_240, buf_236, buf_242], [16, 1, 1]);
        addComputePass(device, commandEncoder, piplines[295], [buf_243, buf_15, buf_244], [8016, 1, 1]);
        addComputePass(device, commandEncoder, piplines[296], [buf_245, buf_243], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[297], [buf_236, buf_245], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[298], [buf_245, buf_243, buf_236], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[299], [buf_246, buf_245], [1, 1, 1]);
        addComputePass(device, commandEncoder, piplines[300], [buf_247, buf_243, buf_236, buf_246], [1336, 1, 1]);
        addComputePass(device, commandEncoder, piplines[301], [buf_248, buf_247], [16, 167, 1]);
        addComputePass(device, commandEncoder, piplines[302], [buf_249, buf_248], [167, 1, 1]);
        addComputePass(device, commandEncoder, piplines[303], [buf_250, buf_4, buf_248, buf_249], [8, 1, 1]);
        addComputePass(device, commandEncoder, piplines[304], [output0, buf_250], [1, 1, 1]);
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
  