import numpy as np
rng = np.random.default_rng(0)
D_q15 = rng.integers(-32768, 32768, size=(128, 256), dtype=np.int16)
r_q15 = rng.integers(-32768, 32768, size=(128,),     dtype=np.int16)
def emit(path, vec):
    with open(path, "w") as f:
        for v in vec.astype(np.int16):
            f.write(f"{int(v) & 0xFFFF:04X}\n")
emit("atom0.mem",   D_q15[:, 0])
emit("atom157.mem", D_q15[:, 157])
emit("r.mem",       r_q15)