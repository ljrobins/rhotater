import tater
import taichi as ti
import numpy as np
import time

q1 = ti.Vector([0.0, 0.0, 1.0, 0.0])
q2 = ti.Vector([0.0, 1.0, 0.0, 0.0])

print(tater.quat_add(q1, q2))

hhat = ti.Vector([1.0, 0.0, 0.0])
phi = 0.1
theta = 0.2
psi = 0.3
is_sam = False
print(tater.quat_r_from_eas(hhat, phi, theta, psi, is_sam))

omega0 = ti.Vector([0.2, 0.3, 0.4])
itensor = ti.Vector([1.0, 2.0, 3.0])
teval = 5.0

omega, k2, tau0, taudot, is_sam = tater.torque_free_angular_velocity(omega0, itensor, teval)
print(omega, k2, tau0, taudot, is_sam)

print(tater.torque_free_attitude(q1, omega0, omega, itensor, teval, k2, tau0, taudot, is_sam))

n = int(1e7)
teval = ti.field(dtype=ti.f32, shape=n)
teval.from_numpy(np.linspace(0, 10, n, dtype=np.float32))
omegas = ti.Vector.field(3, dtype=ti.f32, shape=n)
quats = ti.Vector.field(4, dtype=ti.f32, shape=n)
quat0 = q1

print(n/1e6)

@ti.kernel
def prop_single(omega0: ti.math.vec3, quat0: ti.math.vec4, itensor: ti.math.vec3) -> float:
    for i in range(n):
        omega, k2, tau0, taudot, is_sam = tater.torque_free_angular_velocity(omega0, itensor, teval[i])
        omegas[i] = omega
        quats[i] = tater.torque_free_attitude(quat0, omega0, omega, itensor, teval[i], k2, tau0, taudot, is_sam)
    return 0.0

t1 = time.time()
prop_single(omega0, quat0, itensor)
# quats = quats.to_numpy()
print(f'{n/(time.time()-t1):.2e}')

# import matplotlib.pyplot as plt

# plt.plot(teval.to_numpy(), quats)
# plt.show()