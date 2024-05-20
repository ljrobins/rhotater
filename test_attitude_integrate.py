import taichi as ti
import numpy as np
import time
import trimesh

obj = trimesh.load('/Users/liamrobinson/Documents/mirage/mirage/resources/models/cube.obj')

ti.init(arch=ti.gpu, 
        default_fp=ti.float32,
        fast_math=True,
        advanced_optimization=True,
        num_compile_threads=32,
        opt_level=3)

@ti.func
def rand_b2(r: float) -> ti.math.vec3:
    return r * rand_s2() * ti.random() ** (1 / 3)

@ti.func
def rand_s2() -> ti.math.vec3:
    return ti.math.vec3(ti.randn(), ti.randn(), ti.randn()).normalized()


@ti.func
def rand_s3() -> ti.math.vec4:
    return ti.math.vec4(ti.randn(), ti.randn(), ti.randn(), ti.randn()).normalized()


@ti.func
def quat_to_dcm(q: ti.math.vec4) -> ti.math.mat3:
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    C = ti.math.mat3(
        1 - 2 * q1**2 - 2 * q2**2, 2 * (q0 * q1 + q2 * q3), 2 * (q0 * q2 - q1 * q3), 
        2 * (q0 * q1 - q2 * q3), 1 - 2 * q0**2 - 2 * q2**2, 2 * (q1 * q2 + q0 * q3), 
        2 * (q0 * q2 + q1 * q3), 2 * (q1 * q2 - q0 * q3), 1 - 2 * q0**2 - 2 * q1**2
    )
    return C.transpose()

@ti.func
def normalized_convex_light_curve(L: ti.math.vec3, O: ti.math.vec3) -> float:
    b = 0.0
    ti.loop_config(serialize=True)
    for i in range(fn.shape[0]):
        fr = brdf_phong(L, O, fn[i], fd[i], fs[i], fp[i])
        b += fr * rdot(fn[i], O) * rdot(fn[i], L)
    return b

@ti.func
def rdot(v1, v2) -> float:
    return ti.max(ti.math.dot(v1,v2), 0.0)

@ti.func
def brdf_phong(
    L: ti.math.vec3,
    O: ti.math.vec3,
    N: ti.math.vec3,
    cd: float,
    cs: float,
    n: float,
) -> float:
    NdL = ti.math.dot(N, L)
    if NdL <= 0.0:
        NdL = np.inf
    R = (2 * ti.math.dot(N, L) * N - L).normalized()
    fd = cd / np.pi
    fs = cs * (n + 2) / (2 * np.pi) * rdot(R, O) ** n / NdL
    return fd + fs


@ti.kernel
def integrate(itensor: ti.math.vec3) -> int:
    h = teval[1] - teval[0]
    for i in q:
        for j in range(teval.shape[0]):
            dcm = quat_to_dcm(q[i])
            lc[i,j] = normalized_convex_light_curve(dcm @ L, dcm @ O)
            
            q[i], w[i] = rk4_step(q[i], w[i], h, itensor)
            q[i] = q[i].normalized()

    return 0 # to prevent taichi from lying about timing

@ti.pyfunc
def rk4_step(q0, w0, h, itensor):
    k1q, k1w = deriv(q0, w0, itensor)
    k2q, k2w = deriv(q0 + h * k1q / 2, w0 + h * k1w / 2, itensor)
    k3q, k3w = deriv(q0 + h * k2q / 2, w0 + h * k2w / 2, itensor)
    k4q, k4w = deriv(q0 + h * k3q, w0 + h * k3w, itensor)
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
    qdot = qmat @ ti.math.vec4(w.xyz, 0.0)

    wdot = ti.math.vec3(0.0, 0.0, 0.0)
    wdot.x = -1 / itensor[0] * (itensor[2] - itensor[1]) * w.y * w.z
    wdot.y = -1 / itensor[1] * (itensor[0] - itensor[2]) * w.z * w.x
    wdot.z = -1 / itensor[2] * (itensor[1] - itensor[0]) * w.x * w.y

    return qdot, wdot



v1 = obj.vertices[obj.faces[:,0]]
v2 = obj.vertices[obj.faces[:,1]]
v3 = obj.vertices[obj.faces[:,2]]
fnn = np.cross(v2 - v1, v3 - v1)
fan = np.linalg.norm(fnn, axis=1, keepdims=True) / 2
fnn = fnn / fan / 2

fa = ti.field(dtype=ti.f32, shape=fan.shape)
fa.from_numpy(fan.astype(np.float32))

fn = ti.Vector.field(n=3, dtype=ti.f32, shape=fnn.shape[0])
fn.from_numpy(fnn.astype(np.float32))

fd = ti.field(dtype=ti.f32, shape=fnn.shape[0])
fd.fill(0.5)
fs = ti.field(dtype=ti.f32, shape=fnn.shape[0])
fs.fill(0.5)
fp = ti.field(dtype=ti.f32, shape=fnn.shape[0])
fp.fill(3)

L = ti.Vector([1.0, 0.0, 0.0]).normalized()
O = ti.Vector([1.0, 1.0, 0.0]).normalized()


@ti.kernel
def one_lc_with_transform() -> float:
    dcm = quat_to_dcm(q[0])
    lc = normalized_convex_light_curve(dcm @ L, dcm @ O)
    return lc

# print(one_lc())
# endd


q0 = ti.Vector([0.0, 0.0, 0.0, 1.0])
w0 = ti.Vector([1.0, 1.0, 1.0])
itensor = ti.Vector([1.0, 2.0, 3.0])

n = int(1e7)
m = 20
tmax = 10.0
tspace = np.linspace(0, tmax, m, dtype=np.float32)
h = tspace[1] - tspace[0]

teval = ti.field(dtype=ti.f32, shape=m)
teval.from_numpy(tspace)

q = ti.Vector.field(n=4, dtype=ti.f32, shape=n)
w = ti.Vector.field(n=3, dtype=ti.f32, shape=n)
lc = ti.field(dtype=ti.f32, shape=(n,teval.shape[0]))

# %%
# Initializing the quaternion and angular velocity guesses

@ti.kernel
def initialize_quats() -> int:
    for i in q:
        q[i] = rand_s3()
    return 0

initialize_quats()

@ti.kernel
def initialize_omegas() -> int:
    for i in w:
        w[i] = rand_b2(1.0)
    return 0

initialize_omegas()

for i in range(5):
    t1 = time.time()
    integrate(itensor)
    print(time.time()-t1)
    qn = q.to_numpy()
    wn = w.to_numpy()
    lcn = lc.to_numpy()

    # print(qn)