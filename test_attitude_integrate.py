import taichi as ti
import numpy as np
import time

ti.init(arch=ti.gpu, default_fp=ti.float32)

@ti.kernel
def integrate(q0: ti.math.vec4, w0: ti.math.vec3, tmax: float, h: float, itensor: ti.math.vec3):
    t = 0.0
    i = 0
    q[0] = q0
    w[0] = w0

    ti.loop_config(bit_vectorize=True)
    while t < tmax:
        i += 1
        q[i], w[i] = rk4_step(q[i-1], w[i-1], h, itensor)
        t += h

@ti.pyfunc
def rk4_step(q0, w0, h, itensor):
    k1q, k1w = deriv(q0, w0, itensor)
    k2q, k2w = deriv(q0 + h * k1q / 2, w0 + h * k1w / 2, itensor)
    k3q, k3w = deriv(q0 + h * k2q / 2, w0 + h * k2w / 2, itensor)
    k4q, k4w = deriv(q0 + h * k3q, w0 + h * k3w, itensor)
    # k3 = deriv(x0 + h * k2 / 2, itensor)
    # k4 = deriv(x0 + h * k3, itensor)
    return (
        q0 + h * (k1q + 2 * k2q + 2 * k3q + k4q) / 6, 
        w0 + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6)

@ti.pyfunc
def deriv(q, w, itensor):
    qmat = 0.5 * ti.math.mat4(
            q[3], -q[2], q[1], q[0],
            q[2], q[3], -q[0], q[1],
            -q[1], q[0], q[3], q[2],
            -q[0], -q[1], -q[2], q[3],
    )
    wvec = ti.math.vec4(w.xyz, 1.0)
    qdot = qmat @ wvec

    wdot = ti.math.vec3(0.0, 0.0, 0.0)
    wdot.x = -1 / itensor[0] * (itensor[2] - itensor[1]) * w.y * w.z
    wdot.y = -1 / itensor[1] * (itensor[0] - itensor[2]) * w.z * w.x
    wdot.z = -1 / itensor[2] * (itensor[1] - itensor[0]) * w.x * w.y

    return qdot, wdot

q0 = ti.Vector([1.0, 0.0, 0.0, 0.0])
w0 = ti.Vector([1.0, 1.0, 1.0])
itensor = ti.Vector([1.0, 2.0, 3.0])

n = int(1e5)
teval = ti.field(dtype=ti.f32, shape=n)
tmax = 10
tspace = np.linspace(0, tmax, n, dtype=np.float32)
h = tspace[1] - tspace[0]
teval.from_numpy(tspace)

q = ti.Vector.field(n=4, dtype=ti.f32, shape=n)
w = ti.Vector.field(n=3, dtype=ti.f32, shape=n)

t1 = time.time()
integrate(q0, w0, tmax, h, itensor)
qn = q.to_numpy()
wn = w.to_numpy()
print(time.time()-t1)
# exit(1)