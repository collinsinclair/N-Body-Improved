import torch
import triton
import triton.language as tl
import os

os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

def get_configs_sum():
    return [
        triton.Config({'BLOCK_SIZE': B}, num_warps=2, num_stages=1) \
        for B in [16, 32, 64, 128]
    ]

def get_configs_max_error():
    return [
        triton.Config({'BLOCK_SIZE': B}, num_warps=1, num_stages=2) \
        for B in [1, 2, 4, 8]
    ]

def get_configs_rk4():
    return [
        triton.Config({'BLOCK_SIZE': B}, num_warps=1, num_stages=1) \
        for B in [4, 8, 16, 32, 64]
    ]

def get_configs_mult():
    return [
        triton.Config({'BLOCK_SIZE': B}, num_warps=2, num_stages=2) \
        for B in [32, 64, 128, 256]
    ]

def get_configs_acc():
    return [
        triton.Config({'BLOCK_SIZE': B}, num_warps=2, num_stages=1) \
        for B in [4, 8, 16, 32]
    ]


@triton.autotune(
    configs=get_configs_sum(),
    key=["numel"],
)
@triton.jit
def sum_tensors_kernel(a_ptr, b_ptr, c_ptr, numel, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel
    a = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b = tl.load(b_ptr + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, a + b, mask=mask)

@triton.autotune(
    configs=get_configs_mult(),
    key=["numel"],
)
@triton.jit
def sum_scalar_mult_kernel(a_ptr, b_ptr, c_ptr, scalar, numel, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel
    a = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b = tl.load(b_ptr + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, a + b * scalar, mask=mask)

@triton.autotune(
    configs=get_configs_rk4(),
    key=["numel"],
)
@triton.jit
def rk4_sum_kernel(k1_ptr, k2_ptr, k3_ptr, k4_ptr, out_ptr, factor, numel, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel
    k1 = tl.load(k1_ptr + idx, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr + idx, mask=mask, other=0.0)
    k3 = tl.load(k3_ptr + idx, mask=mask, other=0.0)
    k4 = tl.load(k4_ptr + idx, mask=mask, other=0.0)
    out = (k1 + 2*k2 + 2*k3 + k4) * factor
    tl.store(out_ptr + idx, out, mask=mask)

@triton.autotune(
    configs=get_configs_max_error(),
    key=["numel"],
)
@triton.jit
def max_error_kernel(cur_ptr, prev_ptr, err_ptr, numel, BLOCK_SIZE: tl.constexpr):
    # Compute per-element squared diff ratio and reduce via atomic max
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel
    cur = tl.load(cur_ptr + idx, mask=mask, other=0.0)
    prev = tl.load(prev_ptr + idx, mask=mask, other=0.0)
    sq_diff = (cur - prev) * (cur - prev)
    sq_cur = cur * cur
    ratio = tl.where(sq_cur > 0, sq_diff / sq_cur, 0.0)
    max_val = tl.sqrt(tl.max(ratio, axis=0))
    tl.atomic_max(err_ptr, max_val)

@triton.autotune(
    configs=get_configs_acc(),
    key=["numel"],
)
@triton.jit
def calculate_acceleration_kernel(masses_ptr, positions_ptr, accelerations_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_i = offs_i < numel

    # Load positions for bodies i
    pos_i_x = tl.load(positions_ptr + offs_i * 3 + 0, mask=mask_i, other=0.0)
    pos_i_y = tl.load(positions_ptr + offs_i * 3 + 1, mask=mask_i, other=0.0)
    pos_i_z = tl.load(positions_ptr + offs_i * 3 + 2, mask=mask_i, other=0.0)

    acc_x = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    acc_y = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    acc_z = tl.zeros([BLOCK_SIZE], dtype=tl.float64)

    for j_start in range(0, numel, BLOCK_SIZE):
        offs_j = j_start + tl.arange(0, BLOCK_SIZE)
        mask_j = offs_j < numel

        pos_j_x = tl.load(positions_ptr + offs_j * 3 + 0, mask=mask_j, other=0.0)
        pos_j_y = tl.load(positions_ptr + offs_j * 3 + 1, mask=mask_j, other=0.0)
        pos_j_z = tl.load(positions_ptr + offs_j * 3 + 2, mask=mask_j, other=0.0)
        m_j     = tl.load(masses_ptr   + offs_j,           mask=mask_j, other=0.0)

        dx = pos_j_x[None, :] - pos_i_x[:, None]
        dy = pos_j_y[None, :] - pos_i_y[:, None]
        dz = pos_j_z[None, :] - pos_i_z[:, None]

        r2 = dx * dx + dy * dy + dz * dz + 1e-14
        inv_r3 = 1.0 / (r2 * tl.sqrt(r2))

        mask_ij = mask_i[:, None] & mask_j[None, :] & (offs_i[:, None] != offs_j[None, :])

        w = m_j[None, :] * inv_r3 * mask_ij
        acc_x += tl.sum(w * dx, axis=1)
        acc_y += tl.sum(w * dy, axis=1)
        acc_z += tl.sum(w * dz, axis=1)

    G_const = 6.67e-11
    acc_x *= G_const
    acc_y *= G_const
    acc_z *= G_const

    tl.store(accelerations_ptr + offs_i * 3 + 0, acc_x, mask=mask_i)
    tl.store(accelerations_ptr + offs_i * 3 + 1, acc_y, mask=mask_i)
    tl.store(accelerations_ptr + offs_i * 3 + 2, acc_z, mask=mask_i)

def grid(numel):
    return lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)

# a + b
def sum_tensors(a, b):
    numel = a.numel()
    c = torch.empty_like(a)
    sum_tensors_kernel[grid(numel)](a.view(-1), b.view(-1), c.view(-1), numel)
    return c.view_as(a)


# a + b * scalar
def sum_scalar_mult_tensor(a, b, scalar):
    numel = a.numel()
    c = torch.empty_like(a)
    sum_scalar_mult_kernel[grid(numel)](a.view(-1), b.view(-1), c.view(-1), scalar, numel)
    return c.view_as(a)

# (k1 + 2k2 + 2k3 + k4) * factor
def rk4_sum(k1, k2, k3, k4, factor):
    numel = k1.numel()
    out = torch.empty_like(k1)
    rk4_sum_kernel[grid(numel)](k1.view(-1), k2.view(-1), k3.view(-1), k4.view(-1), out.view(-1), factor, numel)
    return out.view_as(k1)

# acceleration due to gravity
def calc_acc(masses, positions):
    numel = masses.shape[0]
    accel = torch.zeros_like(positions)
    calculate_acceleration_kernel[grid(numel)](masses, positions.view(-1), accel.view(-1), numel)
    return accel

# Runge-Kutta step
def runge_kutta(masses, positions, velocities, time_step):
    k1_v = calc_acc(masses, positions)
    k1_p = velocities

    p2 = sum_scalar_mult_tensor(positions, k1_p, time_step * 0.5)
    k2_v = calc_acc(masses, p2)
    k2_p = sum_scalar_mult_tensor(velocities, k1_v, time_step * 0.5)

    p3 = sum_scalar_mult_tensor(positions, k2_p, time_step * 0.5)
    k3_v = calc_acc(masses, p3)
    k3_p = sum_scalar_mult_tensor(velocities, k2_v, time_step * 0.5)

    p4 = sum_scalar_mult_tensor(positions, k3_p, time_step)
    k4_v = calc_acc(masses, p4)
    k4_p = sum_scalar_mult_tensor(velocities, k3_v, time_step)

    factor = time_step / 6.0

    new_v = rk4_sum(k1_v, k2_v, k3_v, k4_v, factor)
    new_p = rk4_sum(k1_p, k2_p, k3_p, k4_p, factor)

    return new_p, new_v

# max error between flat change vectors
def max_error(cur, prev):
    numel = cur.numel()
    err = torch.zeros(1, device=cur.device)
    max_error_kernel[grid(numel)](cur.view(-1), prev.view(-1), err, numel)
    return err

# keep running RK4 steps until error is within bounds or depth limit is reached
def update_particles_recursive(masses, positions, velocities, time_step, prev_change, iter_lim, time):
    half = time_step / 2.0
    s1_p, s1_v = runge_kutta(masses, positions, velocities, half)
    p_mid = sum_tensors(positions, s1_p)
    v_mid = sum_tensors(velocities, s1_v)
    s2_p, s2_v = runge_kutta(masses, p_mid, v_mid, half)
    total_change = sum_tensors(s1_p, s2_p)
    if iter_lim > 0 and max_error(total_change, prev_change) > 1e-5:
        last_p, last_v = update_particles_recursive(masses, positions, velocities, half, s1_p, iter_lim-1, time)
        estimated_final_p = sum_tensors(positions, total_change)
        estimated_final_p_change = sum_scalar_mult_tensor(estimated_final_p, last_p, -1)
        return update_particles_recursive(masses, last_p, last_v, half, estimated_final_p_change, iter_lim-1, time+half)
    else:
        p_final = sum_tensors(positions, total_change)
        total_v_change = sum_tensors(s1_v, s2_v)
        v_final = sum_tensors(velocities, total_v_change)
        return p_final, v_final

# Evolve system forward by time_step
def update_particles(masses, positions, velocities, time_step):
    s_p, s_v = runge_kutta(masses, positions, velocities, time_step)
    return update_particles_recursive(masses, positions, velocities, time_step, s_p, 30, 0.0)

# Class for ease of use
class Simulator:
    def __init__(self, masses, positions, velocities, time_step):
        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.time_step = time_step
        self.time = 0.0

    def step_forward(self):
        self.time += self.time_step
        self.positions, self.velocities = update_particles(self.masses, self.positions, self.velocities, self.time_step)

    def get_positions(self): 
        return self.positions
    
    def get_velocities(self): 
        return self.velocities
