import torch
import triton
import triton.language as tl

# ---- Triton Kernels for basic tensor operations ----
@triton.jit
def sum_tensors_kernel(
    a_ptr, b_ptr, c_ptr, numel,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel
    a = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b = tl.load(b_ptr + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, a + b, mask=mask)

@triton.jit
def scalar_mult_kernel(
    a_ptr, c_ptr, scalar, numel,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel
    a = tl.load(a_ptr + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, a * scalar, mask=mask)

@triton.jit
def sum_scalar_mult_kernel(
    a_ptr, b_ptr, c_ptr, scalar, numel,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel
    a = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b = tl.load(b_ptr + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, a + b * scalar, mask=mask)

@triton.jit
def copy_tensor_kernel(
    src_ptr, dst_ptr, numel,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel
    val = tl.load(src_ptr + idx, mask=mask, other=0.0)
    tl.store(dst_ptr + idx, val, mask=mask)

@triton.jit
def max_error_kernel(
    cur_ptr, prev_ptr, err_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Compute per-element squared diff ratio and reduce via atomic max
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    cur = tl.load(cur_ptr + idx, mask=mask, other=0.0)
    prev = tl.load(prev_ptr + idx, mask=mask, other=0.0)
    sq_diff = (cur - prev) * (cur - prev)
    sq_cur = cur * cur
    ratio = tl.where(sq_cur > 0, sq_diff / sq_cur, 0.0)
    # For simplicity, write back local max into err_ptr
    # (one block only expected for error kernel)
    max_val = tl.max(ratio, axis=0)
    if tl.program_id(0) == 0:
        tl.store(err_ptr, tl.sqrt(max_val))

# ---- Reuse acceleration kernel from previous doc ----
@triton.jit
def calculate_acceleration_kernel(
    masses_ptr, positions_ptr, accelerations_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_i = offs_i < n_elements

    # Load positions for bodies i
    pos_i_x = tl.load(positions_ptr + offs_i * 3 + 0, mask=mask_i, other=0.0)
    pos_i_y = tl.load(positions_ptr + offs_i * 3 + 1, mask=mask_i, other=0.0)
    pos_i_z = tl.load(positions_ptr + offs_i * 3 + 2, mask=mask_i, other=0.0)

    acc_x = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    acc_y = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    acc_z = tl.zeros([BLOCK_SIZE], dtype=tl.float64)

    for j_start in range(0, n_elements, BLOCK_SIZE):
        offs_j = j_start + tl.arange(0, BLOCK_SIZE)
        mask_j = offs_j < n_elements

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

# ---- Python Wrappers ----
def _launch_kernel(kernel, args, numel, block_size):
    grid = ((numel + block_size - 1) // block_size,)
    kernel[grid](*(args + [numel, block_size]))

# Sum two [n,3] tensors
def sum_tensors(a, b, block_size=256):
    numel = a.numel()
    c = torch.empty_like(a)
    _launch_kernel(sum_tensors_kernel, [a.view(-1), b.view(-1), c.view(-1)], numel, block_size)
    return c.view_as(a)

# Scalar multiply [n,3]
def scalar_mult_tensor(a, scalar, block_size=256):
    numel = a.numel()
    c = torch.empty_like(a)
    _launch_kernel(scalar_mult_kernel, [a.view(-1), c.view(-1), scalar], numel, block_size)
    return c.view_as(a)

# a + b * scalar for [n,3]
def sum_scalar_mult_tensor(a, b, scalar, block_size=256):
    numel = a.numel()
    c = torch.empty_like(a)
    _launch_kernel(sum_scalar_mult_kernel, [a.view(-1), b.view(-1), c.view(-1), scalar], numel, block_size)
    return c.view_as(a)

# Deep copy tensor
copy_tensor = lambda x, bs=256: sum_tensors(x, torch.zeros_like(x), block_size=bs)

# Calculate acceleration
calc_acc = lambda masses, positions, bs=128: _acc_wrapper(masses, positions, bs)

def _acc_wrapper(masses, positions, block_size):
    n = masses.shape[0]
    accel = torch.zeros_like(positions)
    calculate_acceleration_kernel[( (n+block_size-1)//block_size, )](
        masses, positions.view(-1), accel.view(-1), n, BLOCK_SIZE=block_size
    )
    return accel

# Runge-Kutta step

def runge_kutta(masses, positions, velocities, time_step, bs=128):
    k1_v = calc_acc(masses, positions, bs)
    k1_p = velocities
    p2 = sum_scalar_mult_tensor(positions, k1_p, time_step * 0.5)
    k2_v = calc_acc(masses, p2, bs)
    k2_p = sum_scalar_mult_tensor(velocities, k1_v, time_step * 0.5)
    p3 = sum_scalar_mult_tensor(positions, k2_p, time_step * 0.5)
    k3_v = calc_acc(masses, p3, bs)
    k3_p = sum_scalar_mult_tensor(velocities, k2_v, time_step * 0.5)
    p4 = sum_scalar_mult_tensor(positions, k3_p, time_step)
    k4_v = calc_acc(masses, p4, bs)
    k4_p = sum_scalar_mult_tensor(velocities, k3_v, time_step)
    factor = time_step / 6.0
    new_v = (k1_v + 2*k2_v + 2*k3_v + k4_v) * factor
    new_p = (k1_p + 2*k2_p + 2*k3_p + k4_p) * factor
    return new_p, new_v

# Max error between flat change vectors
def max_error(cur, prev, bs=256):
    numel = cur.numel()
    err = torch.zeros(1, device=cur.device)
    max_error_kernel[( (numel+bs-1)//bs, )](
        cur.view(-1), prev.view(-1), err, numel, BLOCK_SIZE=bs
    )
    return err.item()

# Recursive update

def update_particles_recursive(masses, positions, velocities, time_step, prev_change, iter_lim, time, bs=128):
    half = time_step / 2.0
    s1_p, s1_v = runge_kutta(masses, positions, velocities, half, bs)
    s2_p, s2_v = runge_kutta(masses, s1_p + positions, velocities + s1_v, half, bs)
    total_change = sum_tensors(s1_p, s2_p)
    if iter_lim > 0 and max_error(total_change, prev_change) > 1e-5:
        r1 = update_particles_recursive(masses, positions, velocities, half, s1_p, iter_lim-1, time, bs)
        last_p = r1.positions_array[-1]
        last_v = r1.velocities_array[-1]
        r2 = update_particles_recursive(masses, last_p, last_v, half, last_p - last_p, iter_lim-1, time+half, bs)
        return r1 + r2
    else:
        p_mid = sum_tensors(positions, total_change)
        v_mid = sum_tensors(velocities, s1_v + s2_v)
        return SimStep([positions, p_mid], [velocities, v_mid], [time, time+time_step], len(positions)//3)

# Simplified SimStep for Python
class SimStep:
    def __init__(self, positions_array, velocities_array, times_array, n):
        self.positions_array = positions_array
        self.velocities_array = velocities_array
        self.times_array = times_array
        self.depth = len(times_array)
        self.n = n
    def __add__(self, other):
        return SimStep(self.positions_array + other.positions_array,
                       self.velocities_array + other.velocities_array,
                       self.times_array + other.times_array,
                       self.n)

# Entry point for stepping

def update_particles(masses, positions, velocities, time_step, bs=128):
    s_p, s_v = runge_kutta(masses, positions, velocities, time_step, bs)
    rec = update_particles_recursive(masses, positions, velocities, time_step, s_p, 30, 0.0, bs)
    return SimStep([positions, *rec.positions_array], [velocities, *rec.velocities_array], [0.0, *rec.times_array], len(positions)//3)

# Python Simulator class
class Simulator:
    def __init__(self, masses, positions, velocities, time_step):
        self.device = masses.device
        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.time_step = time_step
        self.time = 0.0
        self.data = None
    def step_forward(self):
        self.time += self.time_step
        self.data = update_particles(self.masses, self.positions, self.velocities, self.time_step)
        self.positions, self.velocities = self.data.positions_array[-1], self.data.velocities_array[-1]
    def get_positions(self): return self.positions
    def get_velocities(self): return self.velocities
