#!/usr/bin/env python3
"""Convert google/gemma-4-12B-it-assistant safetensors -> llama.cpp GGUF (arch gemma4-assistant).
Pure stdlib. Mirrors llama.cpp src/models/gemma4-assistant.cpp tensor names.
Linear weights: HF [out,in] -> GGUF [in,out], byte order unchanged."""
import struct, json, sys, os

T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
TF32, TBF16 = 0, 30

class Rd:
    def __init__(s, f): s.f=f
    def u64(s): return struct.unpack('<Q', s.f.read(8))[0]
    def val(s, t):
        if t==T_U8: return struct.unpack('<B', s.f.read(1))[0]
        if t==T_I8: return struct.unpack('<b', s.f.read(1))[0]
        if t==T_U16: return struct.unpack('<H', s.f.read(2))[0]
        if t==T_I16: return struct.unpack('<h', s.f.read(2))[0]
        if t==T_U32: return struct.unpack('<I', s.f.read(4))[0]
        if t==T_I32: return struct.unpack('<i', s.f.read(4))[0]
        if t==T_F32: return struct.unpack('<f', s.f.read(4))[0]
        if t==T_BOOL: return bool(s.f.read(1)[0])
        if t==T_STR:
            n=s.u64(); return s.f.read(n).decode()
        if t==T_ARR:
            et=struct.unpack("<I", s.f.read(4))[0]; n=s.u64()
            return (et, [s.val(et) for _ in range(n)])
        if t==T_U64: return struct.unpack('<Q', s.f.read(8))[0]
        if t==T_I64: return struct.unpack('<q', s.f.read(8))[0]
        if t==T_F64: return struct.unpack('<d', s.f.read(8))[0]
        raise ValueError(t)

SCALAR_FMT = {T_U8:'<B',T_I8:'<b',T_U16:'<H',T_I16:'<h',T_U32:'<I',T_I32:'<i',T_F32:'<f',T_BOOL:'<B',T_U64:'<Q',T_I64:'<q',T_F64:'<d'}

def read_main_gguf(path):
    with open(path,'rb') as f:
        assert f.read(4)==b'GGUF'
        struct.unpack('<I', f.read(4))
        ntensor=struct.unpack('<Q', f.read(8))[0]; nkv=struct.unpack('<Q', f.read(8))[0]
        r=Rd(f); kvs=[]
        for _ in range(nkv):
            k=r.val(T_STR); t=struct.unpack("<I", f.read(4))[0]; v=r.val(t); kvs.append((k,t,v))
        tens={}
        for _ in range(ntensor):
            name=r.val(T_STR); nd=struct.unpack('<I', f.read(4))[0]
            dims=[r.u64() for _ in range(nd)]
            typ=struct.unpack('<I', f.read(4))[0]; off=r.u64()
            n=1
            for d in dims: n*=d
            tens[name]=(off, n*4 if typ==TF32 else n*2 if typ in (1,TBF16) else n)
        align = next((v for k, t, v in kvs if k == 'general.alignment'), 32)
        data_start = (f.tell() + align - 1) // align * align
        return kvs, {name: (data_start + off, size) for name, (off, size) in tens.items()}

def bf16_to_f32(buf):
    out = bytearray()
    for i in range(0, len(buf), 2):
        out += struct.pack('<I', struct.unpack('<H', buf[i:i+2])[0] << 16)
    return bytes(out)

class St:
    def __init__(s, path):
        s.f = open(path,'rb')
        n = struct.unpack('<Q', s.f.read(8))[0]
        s.hdr = json.loads(s.f.read(n)); s.data_off = 8+n
    def get(s, name):
        b0, b1 = s.hdr[name]['data_offsets']
        s.f.seek(s.data_off+b0); return s.f.read(b1-b0)
    def shape(s, name): return s.hdr[name]['shape']

def wstr(s): b=s.encode(); return struct.pack('<Q', len(b)) + b

def write_gguf(path, kvs, blobs, align=32):
    with open(path,'wb') as f:
        f.write(b'GGUF'); f.write(struct.pack('<I', 3))
        f.write(struct.pack('<QQ', len(blobs), len(kvs)))
        for k,t,v in kvs:
            f.write(wstr(k)); f.write(struct.pack("<I", t))
            if t==T_STR:
                f.write(wstr(v))
            elif t==T_ARR:
                et, xs = v
                f.write(struct.pack("<IQ", et, len(xs)))
                for e in xs:
                    if et==T_STR: f.write(wstr(e))
                    else: f.write(struct.pack(SCALAR_FMT[et], e))
            else:
                f.write(struct.pack(SCALAR_FMT[t], v))
        offset = 0
        for name, typ, dims, _ in blobs:
            f.write(wstr(name)); f.write(struct.pack('<I', len(dims)))
            for d in dims: f.write(struct.pack('<Q', d))
            f.write(struct.pack('<IQ', typ, offset))
            offset = (offset + len(dict((b[0], b[3]) for b in blobs)[name]) + align-1)//align*align
        while f.tell() % align: f.write(b'\0')
        for _,_,_,data in blobs:
            f.write(data)
            while f.tell() % align: f.write(b'\0')

def main():
    models = sys.argv[1] if len(sys.argv)>1 else 'models'
    main_gguf = os.path.join(models, 'gemma-4-12b-it-qat-q4_0.gguf')
    st = St(os.path.join(models, 'gemma-4-12b-it-assistant.safetensors'))
    out_path = os.path.join(models, 'gemma-4-12b-it-assistant-bf16.gguf')
    kvs, tens = read_main_gguf(main_gguf)
    with open(main_gguf,'rb') as f:
        f.seek(tens['rope_freqs.weight'][0]); rope_raw = f.read(tens['rope_freqs.weight'][1])

    # copy tokenizer + license KVs verbatim
    out_kvs = [(k,t,v) for k,t,v in kvs
               if k.startswith('tokenizer.') or k in ('general.license','general.license.link','general.tags')]
    swp = next(v for k,t,v in kvs if k=='gemma4.attention.sliding_window_pattern')
    N_EMBD, N_EMBD_OUT, N_FF = 1024, 3840, 8192
    out_kvs += [
        ('general.architecture', T_STR, 'gemma4-assistant'),
        ('general.name', T_STR, 'Gemma 4 12B It Assistant'),
        ('general.type', T_STR, 'model'),
        ('general.quantization_version', T_U32, 2),
        ('general.file_type', T_U32, 32),
        ('general.size_label', T_STR, '12B'),
        ('gemma4-assistant.block_count', T_U32, 4),
        ('gemma4-assistant.nextn_predict_layers', T_U32, 4),
        ('gemma4-assistant.context_length', T_U32, 262144),
        ('gemma4-assistant.embedding_length', T_U32, N_EMBD),
        ('gemma4-assistant.embedding_length_out', T_U32, N_EMBD_OUT),
        ('gemma4-assistant.feed_forward_length', T_U32, N_FF),
        ('gemma4-assistant.attention.head_count', T_U32, 16),
        ('gemma4-assistant.attention.head_count_kv', T_ARR, (T_U32, [8,8,8,1])),
        ('gemma4-assistant.attention.key_length', T_U32, 512),
        ('gemma4-assistant.attention.key_length_swa', T_U32, 256),
        ('gemma4-assistant.attention.value_length', T_U32, 512),
        ('gemma4-assistant.attention.value_length_swa', T_U32, 256),
        ('gemma4-assistant.attention.layer_norm_rms_epsilon', T_F32, 1e-6),
        ('gemma4-assistant.attention.sliding_window', T_U32, 1024),
        ('gemma4-assistant.attention.sliding_window_pattern', T_ARR, (swp[0], [1,1,1,0])),
        ('gemma4-assistant.attention.shared_kv_layers', T_U32, 4),
        ('gemma4-assistant.rope.freq_base', T_F32, 1000000.0),
        ('gemma4-assistant.rope.freq_base_swa', T_F32, 10000.0),
        ('gemma4-assistant.rope.dimension_count', T_U32, 512),
        ('gemma4-assistant.rope.dimension_count_swa', T_U32, 256),
    ]

    # plan entries: (gguf_name, ggml_type, dims_or_None_for_inferred, hf_name, conv)
    lin = lambda hf: (TBF16, list(reversed(st.shape(hf))), hf, None)   # [in,out] passthrough
    nrm = lambda hf: (TF32, [st.shape(hf)[0]], hf, 'f32')
    plan = [
        ('token_embd.weight',) + lin('model.embed_tokens.weight'),
        ('output_norm.weight',) + nrm('model.norm.weight'),
        ('nextn.pre_projection.weight',) + lin('pre_projection.weight'),
        ('nextn.post_projection.weight',) + lin('post_projection.weight'),
        ('rope_freqs.weight', TF32, [256], None, 'raw'),
    ]
    for i in range(4):
        g = f'blk.{i}.'
        plan += [
            (g+'attn_norm.weight',) + nrm(f'model.layers.{i}.input_layernorm.weight'),
            (g+'attn_q.weight',) + lin(f'model.layers.{i}.self_attn.q_proj.weight'),
            (g+'attn_q_norm.weight',) + nrm(f'model.layers.{i}.self_attn.q_norm.weight'),
            (g+'attn_output.weight',) + lin(f'model.layers.{i}.self_attn.o_proj.weight'),
            (g+'post_attention_norm.weight',) + nrm(f'model.layers.{i}.post_attention_layernorm.weight'),
            (g+'layer_output_scale.weight', TF32, [1], f'model.layers.{i}.layer_scalar', 'f32'),
            (g+'ffn_norm.weight',) + nrm(f'model.layers.{i}.pre_feedforward_layernorm.weight'),
            (g+'ffn_gate.weight',) + lin(f'model.layers.{i}.mlp.gate_proj.weight'),
            (g+'ffn_up.weight',) + lin(f'model.layers.{i}.mlp.up_proj.weight'),
            (g+'ffn_down.weight',) + lin(f'model.layers.{i}.mlp.down_proj.weight'),
            (g+'post_ffw_norm.weight',) + nrm(f'model.layers.{i}.post_feedforward_layernorm.weight'),
        ]

    blobs = []
    for name, typ, dims, hf, conv in plan:
        if conv == 'raw':
            data = rope_raw
        else:
            data = st.get(hf)
            if conv == 'f32': data = bf16_to_f32(data)
        blobs.append((name, typ, dims, data))

    write_gguf(out_path, out_kvs, blobs)
    print('wrote', out_path, f'{os.path.getsize(out_path)/1e9:.3f} GB,', len(blobs), 'tensors,', len(out_kvs), 'kvs')

if __name__ == '__main__':
    main()
