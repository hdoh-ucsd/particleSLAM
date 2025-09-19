# map_update.py
import os, math, numpy as np
import importlib.util

# robust local import of utils.py for Bresenham
THIS_DIR = os.path.dirname(__file__)
UTILS_PATH = os.path.join(THIS_DIR, "utils.py")
_spec = importlib.util.spec_from_file_location("utils_local", UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_utils)
bresenham2D = _utils.bresenham2D

# log-odds tuning
CONF      = 0.8
LOG_OCC   = math.log(CONF/(1-CONF))   # ~ +1.386
LOG_FREE  = -LOG_OCC                  # ~ -1.386
CLAMP_MIN = -10.0
CLAMP_MAX =  10.0

def init_map(res=0.05, xmin=-25, xmax=25, ymin=-25, ymax=25):
    MAP = dict(res=float(res), xmin=float(xmin), xmax=float(xmax),
               ymin=float(ymin), ymax=float(ymax))
    MAP['sizex'] = int(np.ceil((MAP['xmax']-MAP['xmin'])/MAP['res'])) + 1
    MAP['sizey'] = int(np.ceil((MAP['ymax']-MAP['ymin'])/MAP['res'])) + 1
    MAP['log']   = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32)
    return MAP

def meters_to_cells(x, y, MAP):
    ix = int(np.floor((x - MAP['xmin'])/MAP['res']))
    iy = int(np.floor((y - MAP['ymin'])/MAP['res']))
    ix = np.clip(ix, 0, MAP['sizex']-1)
    iy = np.clip(iy, 0, MAP['sizey']-1)
    return ix, iy

def update_with_scan_cpu(MAP, robot_pose, ranges, angles, lidar_cfg):
    """CPU OGM update using Bresenham; LiDAR extrinsics from lidar_cfg."""
    sx, sy, syaw = lidar_cfg.sensor_world_pose(robot_pose)
    rx, ry = meters_to_cells(sx, sy, MAP)

    rmin = max(lidar_cfg.rmin, 0.10)
    rmax = min(lidar_cfg.rmax, lidar_cfg.rmax_used)

    r = np.asarray(ranges, dtype=np.float32)
    a = np.asarray(angles, dtype=np.float32)
    valid = (r > rmin) & (r < rmax)
    if not np.any(valid):
        return
    r = r[valid]; a = a[valid]

    exm = sx + r*np.cos(a + syaw)
    eym = sy + r*np.sin(a + syaw)

    for xw, yw in zip(exm, eym):
        ex, ey = meters_to_cells(xw, yw, MAP)
        ray = bresenham2D(rx, ry, ex, ey)
        cx = np.asarray(ray[0], dtype=np.intp); cy = np.asarray(ray[1], dtype=np.intp)
        if cx.size == 0: continue
        inb = (cx>=0)&(cx<MAP['sizex'])&(cy>=0)&(cy<MAP['sizey'])
        cx, cy = cx[inb], cy[inb]
        if cx.size > 1:
            MAP['log'][cx[:-1], cy[:-1]] += LOG_FREE
        MAP['log'][cx[-1], cy[-1]]     += LOG_OCC

    np.clip(MAP['log'], CLAMP_MIN, CLAMP_MAX, out=MAP['log'])

def update_with_scan_torch(MAP, log_t, robot_pose, ranges, angles, lidar_cfg, device="cuda"):
    """GPU OGM update (PyTorch): batched integer DDA + scatter-add."""
    import torch
    sx, sy, syaw = lidar_cfg.sensor_world_pose(robot_pose)
    rx, ry = meters_to_cells(sx, sy, MAP)
    H, W = MAP['sizex'], MAP['sizey']

    rmin = max(lidar_cfg.rmin, 0.10)
    rmax = min(lidar_cfg.rmax, lidar_cfg.rmax_used)
    r = np.asarray(ranges, dtype=np.float32)
    a = np.asarray(angles, dtype=np.float32)
    mask = (r > rmin) & (r < rmax)
    if not np.any(mask): return
    r = torch.from_numpy(r[mask]).to(device)
    a = torch.from_numpy(a[mask]).to(device)

    sx_t = torch.tensor(sx, dtype=torch.float32, device=device)
    sy_t = torch.tensor(sy, dtype=torch.float32, device=device)
    syaw_t = torch.tensor(syaw, dtype=torch.float32, device=device)

    exm = sx_t + r*torch.cos(a + syaw_t)
    eym = sy_t + r*torch.sin(a + syaw_t)
    ex = torch.floor((exm - MAP['xmin'])/MAP['res']).to(torch.int64).clamp_(0, H-1)
    ey = torch.floor((eym - MAP['ymin'])/MAP['res']).to(torch.int64).clamp_(0, W-1)

    rx_t = torch.full_like(ex, rx); ry_t = torch.full_like(ey, ry)
    dx = (ex - rx_t).abs(); dy = (ey - ry_t).abs()
    n  = torch.maximum(dx, dy) + 1
    maxn = int(n.max().item())

    idx = torch.arange(maxn, device=device)[None, :].expand(n.size(0), -1)
    valid = idx < n[:, None]
    denom = (n[:, None]-1).clamp(min=1).to(torch.float32)
    t = idx.to(torch.float32)/denom

    x_line = torch.round(rx_t[:, None] + (ex - rx_t)[:, None]*t).to(torch.int64)
    y_line = torch.round(ry_t[:, None] + (ey - ry_t)[:, None]*t).to(torch.int64)

    free_valid = valid.clone()
    free_valid[torch.arange(n.size(0), device=device), n-1] = False

    cx_free = x_line[free_valid]; cy_free = y_line[free_valid]
    cx_occ  = ex;                 cy_occ  = ey

    log_flat = log_t.view(-1)
    idx_free = cx_free * W + cy_free
    idx_occ  = cx_occ  * W + cy_occ
    if idx_free.numel() > 0:
        log_flat.index_add_(0, idx_free, torch.full_like(idx_free, float(LOG_FREE), dtype=log_flat.dtype))
    if idx_occ.numel() > 0:
        log_flat.index_add_(0, idx_occ,  torch.full_like(idx_occ,  float(LOG_OCC),  dtype=log_flat.dtype))
    log_t.clamp_(CLAMP_MIN, CLAMP_MAX)

def logodds_to_prob(L):
    Lc = np.clip(L, CLAMP_MIN, CLAMP_MAX)
    return 1.0 - 1.0/(1.0 + np.exp(Lc))
