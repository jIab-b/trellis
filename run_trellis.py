#!/usr/bin/env python3
import argparse
import ctypes as C
import os
import sys


def load_lib(path_hint=None):
    if path_hint:
        return C.CDLL(path_hint)
    # Try common locations: ./build, ./csrc/build, LD_LIBRARY_PATH
    candidates = [
        os.path.join(os.getcwd(), 'build', 'libtrellis_infer.so'),
        os.path.join(os.getcwd(), 'csrc', 'build', 'libtrellis_infer.so'),
        'libtrellis_infer.so',
    ]
    last_err = None
    for p in candidates:
        try:
            return C.CDLL(p)
        except OSError as e:
            last_err = e
    raise last_err if last_err else OSError('libtrellis_infer.so not found')


def main():
    ap = argparse.ArgumentParser(description='Python caller for TRELLIS C API (trellis_run_v1)')
    ap.add_argument('--lib', help='Path to libtrellis_infer.so')
    ap.add_argument('--weights-root', required=True)
    ap.add_argument('--tokens', required=True, help='Path to tokens .npz')
    ap.add_argument('--uncond', required=True, help='Path to uncond .npz')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--target', default='gs', choices=['gs', 'rf', 'mesh', 'all'])
    ap.add_argument('--steps-ss', type=int, default=25)
    ap.add_argument('--steps-slat', type=int, default=25)
    ap.add_argument('--sigma-min', type=float, default=1e-5)
    ap.add_argument('--sigma-max', type=float, default=1.0)
    ap.add_argument('--cfg', type=float, default=5.0)
    ap.add_argument('--cfg-interval', default='0.5,1.0')
    args = ap.parse_args()

    lib = load_lib(args.lib)

    class RunSpec(C.Structure):
        _fields_ = [
            ('weights_root', C.c_char_p),
            ('tokens_npz',   C.c_char_p),
            ('uncond_npz',   C.c_char_p),
            ('out_dir',      C.c_char_p),
            ('target',       C.c_char_p),
            ('steps_ss',     C.c_int32),
            ('steps_slat',   C.c_int32),
            ('sigma_min',    C.c_float),
            ('sigma_max',    C.c_float),
            ('cfg_strength', C.c_float),
            ('cfg_interval_lo', C.c_float),
            ('cfg_interval_hi', C.c_float),
        ]

    class ErrBuf(C.Structure):
        _fields_ = [('message', C.c_char_p), ('capacity', C.c_uint64)]

    lib.trellis_run_v1.argtypes = [C.POINTER(RunSpec), C.POINTER(ErrBuf)]
    lib.trellis_run_v1.restype = C.c_int

    cfg_lo, cfg_hi = map(float, args.cfg_interval.split(','))

    spec = RunSpec(
        weights_root=args.weights_root.encode(),
        tokens_npz=args.tokens.encode(),
        uncond_npz=args.uncond.encode(),
        out_dir=args.out.encode(),
        target=args.target.encode(),
        steps_ss=args.steps_ss,
        steps_slat=args.steps_slat,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        cfg_strength=args.cfg,
        cfg_interval_lo=cfg_lo,
        cfg_interval_hi=cfg_hi,
    )

    buf = C.create_string_buffer(2048)
    err = ErrBuf(C.cast(buf, C.c_char_p), C.c_uint64(len(buf)))
    rc = lib.trellis_run_v1(C.byref(spec), C.byref(err))
    if rc != 0:
        msg = buf.value.decode(errors='ignore')
        print(f'Error (rc={rc}): {msg}', file=sys.stderr)
        sys.exit(rc)
    print('Success.')


if __name__ == '__main__':
    main()

