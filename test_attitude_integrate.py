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

@ti.dataclass
class PSOProblem:
    global_weight = 1.0
    neighbor_weight = 1.0

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
    for i in current_states:
        for j in range(teval.shape[0]):
            dcm = quat_to_dcm(current_states[i][:4])
            lc[i][j] = normalized_convex_light_curve(dcm @ L, dcm @ O)
            
            current_states[i][:4], current_states[i][4:] = rk4_step(current_states[i][:4], current_states[i][4:], h, itensor)
            current_states[i][:4] = current_states[i][:4].normalized()

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

q0 = ti.Vector([0.0, 0.0, 0.0, 1.0])
w0 = ti.Vector([1.0, 1.0, 1.0])
itensor = ti.Vector([1.0, 2.0, 3.0])

n = int(1e6)
m = 20
tmax = 10.0
tspace = np.linspace(0, tmax, m, dtype=np.float32)
h = tspace[1] - tspace[0]

teval = ti.field(dtype=ti.f32, shape=m)
teval.from_numpy(tspace)

initial_states = ti.Vector.field(n=7, dtype=ti.f32, shape=n)
states_v = ti.Vector.field(n=7, dtype=ti.f32, shape=n)
current_states = ti.Vector.field(n=7, dtype=ti.f32, shape=n)
lc = ti.Vector.field(n=teval.shape[0], dtype=ti.f32, shape=n)
local_best_states = ti.Vector.field(n=7, dtype=ti.f32, shape=n)

loss = ti.field(dtype=ti.f32, shape=(n,4)) # [i,j] is local current, local best, neighbor, and global loss for particle i
loss.fill(np.inf)
loss_inds = ti.field(dtype=ti.u32, shape=(n,4)) # [i,j] is global index of local, neighbor, and global loss for loss[i,j]

# %%
# Initializing the quaternion and angular velocity guesses

@ti.kernel
def initialize_states() -> int:
    for i in current_states:
        initial_states[i][:4] = rand_s3()
        initial_states[i][4:] = rand_b2(1.0)
        current_states[i] = initial_states[i]
    return 0

@ti.kernel
def reset_states() -> int:
    for i in current_states:
        current_states[i] = initial_states[i]
    return 0


@ti.kernel
def compute_loss() -> float:
    min_global_loss = np.inf
    min_global_loss_ind = 0

    # Computing local loss 
    for i in range(loss.shape[0]):
        lossi = (lc[i] - 0.4).norm_sqr()
        loss[i,0] = lossi
        # keeping track of the min global loss for this iteration
        old_min_global_loss = ti.atomic_min(min_global_loss, loss[i,0])
        if old_min_global_loss != min_global_loss:
            min_global_loss_ind = i

        # keeping track of the best loss seen by this particle
        if loss[i,0] < loss[i,1]: # then this is the best loss seen by this particle
            local_best_states[i] = initial_states[i]

        left = (i-1) % loss.shape[0]
        right = (i+1) % loss.shape[0]
        if loss[left,0] < ti.min(loss[i,0], loss[right,0]):
            loss_inds[i,2] = left
            loss[i,2] = loss[left,0]
        elif loss[right,0] < ti.min(loss[i,0], loss[left,0]):
            loss_inds[i,2] = right
            loss[i,2] = loss[right,0]
        else:
            loss_inds[i,2] = i
            loss[i,2] = loss[i,0]
    
    # Filling in global best loss
    for i in range(loss.shape[0]):
        loss[i,3] = min_global_loss
        loss_inds[i,3] = min_global_loss_ind
    return min_global_loss

@ti.kernel
def update_states() -> int:
    for i in range(loss.shape[0]):
        local_best_state = local_best_states[i]
        to_local_best = local_best_state - initial_states[i]
        neighbor_best_state = initial_states[loss_inds[i,2]]
        to_neighbor_best = neighbor_best_state - initial_states[i]
        global_best_state = initial_states[loss_inds[i,3]]
        to_global_best = global_best_state - initial_states[i]

        scale_best_local = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random(), ti.random(), ti.random(), ti.random()])
        scale_neighbor = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random(), ti.random(), ti.random(), ti.random()])
        scale_global = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random(), ti.random(), ti.random(), ti.random()])

        acc = scale_best_local * to_local_best + 0.5 * scale_neighbor * to_neighbor_best + 0.5 * scale_global * to_global_best
        states_v[i] += 0.004 * acc
        if states_v[i].norm_sqr() > 1.0:
            states_v[i] = states_v[i].normalized()
        initial_states[i] += states_v[i]
        
        initial_states[i][:4] = initial_states[i][:4].normalized() # make sure the quaternion keeps its norm
    return 0

for i in range(1000):
    t0 = time.time()
    if i == 0:
        initialize_states()

    t1 = time.time()
    integrate(itensor)
    # print(f'int time: {time.time()-t1:.1e}')

    t1 = time.time()
    best_global_loss = compute_loss()
    # print(f'loss time: {time.time()-t1:.1e}')

    t1 = time.time()
    update_states()
    # print(f'update time: {time.time()-t1:.1e}')

    reset_states()

    print(f'total time: {time.time()-t0:.1e}')

    print(best_global_loss)

    # print(state_before-state_after)

    # sn = initial_states.to_numpy()
    # print(sn[:,:4])

    # lcn = lc.to_numpy()

    # print(loss)
    # print(loss_inds)
