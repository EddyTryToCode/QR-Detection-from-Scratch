import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
import math
import argparse
import itertools
import numpy as np
import pandas as pd
import csv
import time
import concurrent.futures
import joblib
from tqdm import tqdm

# ==============================================================================
# GLOBAL MODEL
# ==============================================================================
RF_MODEL = None

FEATURE_NAMES = [
    'ratio', 'cos_val', 'proximity', 'size_variance', 'edge_density_raw',
    'black_ratio', 'edge_density', 'balance_score', 'center_mean', 'center_std',
    'poly_area_norm', 'corner_top2', 'corner_top3', 'timing_score', 'module_score',
    'avg_fp_width', 'intruder_count', 'dist_to_size', 'hyp_ratio', 'aspect_diag',
]

def _worker_init(model_path):
    global RF_MODEL
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        RF_MODEL = data.get('model', data) if isinstance(data, dict) else data
        if hasattr(RF_MODEL, 'n_jobs'):
            RF_MODEL.n_jobs = 1  # Ngăn chặn catastrophic thread contention

# ==============================================================================
# GEOMETRY UTILITIES
# ==============================================================================

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle_cos(vertex, p1, p2):
    v1 = (float(p1[0]) - float(vertex[0]), float(p1[1]) - float(vertex[1]))
    v2 = (float(p2[0]) - float(vertex[0]), float(p2[1]) - float(vertex[1]))
    denom = (math.hypot(*v1) * math.hypot(*v2)) + 1e-10
    return (v1[0] * v2[0] + v1[1] * v2[1]) / denom

def polygon_area(poly):
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), (float(point[0]), float(point[1])), False) >= 0

def convex_overlap_ratio(poly1, poly2):
    poly1, poly2 = poly1.astype(np.float32), poly2.astype(np.float32)
    area1, area2 = polygon_area(poly1), polygon_area(poly2)
    if area1 <= 1e-6 or area2 <= 1e-6: return 0.0
    inter_area, _ = cv2.intersectConvexConvex(poly1, poly2)
    return float(inter_area / min(area1, area2))

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start_idx = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start_idx, axis=0)
    return pts

def expand_polygon(base_poly_np, fp_widths, alpha=1.25):
    avg_fp_width = float(sum(fp_widths) / len(fp_widths))
    center = np.mean(base_poly_np.astype(np.float32), axis=0)
    expand_dist = avg_fp_width * alpha
    expanded = []
    for p in base_poly_np.astype(np.float32):
        vec = p - center
        norm = np.linalg.norm(vec) + 1e-6
        expanded.append(p + (vec / norm) * expand_dist)
    return np.asarray(expanded, dtype=np.float32)

def resize_for_inference(image, max_side=1200):
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale

# ==============================================================================
# FINDER PATTERN EXTRACTION
# ==============================================================================

def _is_valid_finder_pattern(contours, hierarchy, idx, scale=1.0):
    hier = hierarchy[0]
    scale_sq = scale * scale

    x, y, w, h = cv2.boundingRect(contours[idx])
    if w <= 0 or h <= 0: return False, None, None, None
    
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.4 or aspect_ratio > 2.5: return False, None, None, None

    area_outer = cv2.contourArea(contours[idx])
    if area_outer < 30 * scale_sq: return False, None, None, None

    child_idx = hier[idx][2]
    if child_idx == -1: return False, None, None, None

    area_middle = cv2.contourArea(contours[child_idx])
    if area_middle < 8 * scale_sq: return False, None, None, None

    grandchild_idx = hier[child_idx][2]
    if grandchild_idx == -1: return False, None, None, None

    area_inner = cv2.contourArea(contours[grandchild_idx])
    if area_inner < 2 * scale_sq: return False, None, None, None

    ratio_outer_middle = area_outer / (area_middle + 1e-5)
    ratio_outer_inner = area_outer / (area_inner + 1e-5)

    if not (1.2 < ratio_outer_middle < 4.5): return False, None, None, None
    if not (3.0 < ratio_outer_inner < 16.0): return False, None, None, None

    m_out   = cv2.moments(contours[idx])
    m_inner = cv2.moments(contours[grandchild_idx])
    if m_out['m00'] == 0 or m_inner['m00'] == 0: return False, None, None, None

    cx_out, cy_out = m_out['m10'] / m_out['m00'], m_out['m01'] / m_out['m00']
    cx_in,  cy_in  = m_inner['m10'] / m_inner['m00'], m_inner['m01'] / m_inner['m00']

    fp_width = (w + h) / 2.0
    if math.hypot(cx_out - cx_in, cy_out - cy_in) > fp_width * 0.2: return False, None, None, None

    return True, int(round(cx_out)), int(round(cy_out)), getattr(fp_width, "real", fp_width)

def detect_finder_patterns(image, scale=1.0, mode="normal"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mode == "otsu":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        block_size = int(41 * scale)
        if block_size % 2 == 0: block_size += 1
        block_size = max(11, min(block_size, 51))
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, 2
        )
        if scale > 1.0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    unique_patterns = []
    for i in range(len(contours)):
        valid, cx, cy, fp_width = _is_valid_finder_pattern(contours, hierarchy, i, scale)
        if valid:
            too_close = any(distance((cx, cy), up) < 8 for up in unique_patterns)
            if not too_close:
                unique_patterns.append((cx, cy, fp_width))

    return unique_patterns

# ==============================================================================
# ML FEATURE EXTRACTION
# ==============================================================================

def warp_poly_gray(image, poly, size=96):
    src = order_points(poly).astype(np.float32)
    dst = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (size, size))
        if len(warped.shape) == 3:
            return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        return warped
    except cv2.error:
        return None

def count_transitions(vec):
    b = (np.asarray(vec).astype(np.uint8) > 0).astype(np.uint8)
    if b.size <= 1: return 0
    return int(np.count_nonzero(b[1:] != b[:-1]))

def run_length_ratio_score(vec):
    b = (np.asarray(vec).astype(np.uint8) > 0).astype(np.uint8)
    if b.size < 9: return 0.0
    runs = []
    cur = int(b[0]); cnt = 1
    for v in b[1:]:
        v = int(v)
        if v == cur: cnt += 1
        else: runs.append(cnt); cur = v; cnt = 1
    runs.append(cnt)
    if len(runs) < 4: return 0.0
    targets = [
        (np.array([1, 1, 3, 1, 1], dtype=np.float32), 1.00),
        (np.array([1, 1, 3, 1], dtype=np.float32), 0.80),
        (np.array([1, 3, 1, 1], dtype=np.float32), 0.80),
    ]
    best = 0.0
    for target, weight in targets:
        k = len(target)
        if len(runs) < k: continue
        t = target / target.sum()
        for i in range(len(runs) - k + 1):
            seg = np.array(runs[i:i+k], dtype=np.float32)
            seg /= max(seg.sum(), 1e-6)
            err = float(np.mean(np.abs(seg - t)))
            score = weight * (1.0 - min(err / 0.22, 1.0))
            best = max(best, score)
    return float(np.clip(best, 0.0, 1.0))

def score_corner_patch(gray_patch):
    if gray_patch is None or min(gray_patch.shape[:2]) < 16: return 0.0
    blur = cv2.GaussianBlur(gray_patch, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape[:2]
    scores = [
        run_length_ratio_score(bw[int(h * 0.50), :]),
        run_length_ratio_score(bw[:, int(w * 0.50)]),
    ]
    return float(np.clip(max(scores) if scores else 0.0, 0.0, 1.0))

def alternating_line_score(vec):
    b = (np.asarray(vec).astype(np.uint8) > 0).astype(np.uint8)
    if b.size < 12: return 0.0
    trans_score = min(count_transitions(b) / 12.0, 1.0)
    runs = []
    cur = int(b[0]); cnt = 1
    for v in b[1:]:
        v = int(v)
        if v == cur: cnt += 1
        else: runs.append(cnt); cur = v; cnt = 1
    runs.append(cnt)
    if len(runs) < 6: return 0.45 * trans_score
    runs = np.asarray(runs, dtype=np.float32)
    cv_val = float(np.std(runs) / (np.mean(runs) + 1e-6))
    uniform_score = 1.0 - min(cv_val / 1.25, 1.0)
    return float(np.clip(0.6 * trans_score + 0.4 * uniform_score, 0.0, 1.0))

def extract_features(gray, bw, geometric_info):
    warp_size = 96
    black_ratio = float(np.mean(bw == 0))
    balance_score = 1.0 - min(abs(black_ratio - 0.5) / 0.5, 1.0)
    edges = cv2.Canny(bw, 100, 200)
    edge_density = float(np.count_nonzero(edges) / (edges.size + 1e-5))
    center_roi = gray[24:72, 24:72]
    center_mean = float(np.mean(center_roi) / 255.0)
    center_std = float(np.std(center_roi) / 255.0)

    s = max(18, min(int(round(warp_size * 0.28)), warp_size // 2 - 2))
    corner_rois = [
        gray[0:s, 0:s], gray[0:s, warp_size-s:warp_size],
        gray[warp_size-s:warp_size, warp_size-s:warp_size], gray[warp_size-s:warp_size, 0:s],
    ]
    corner_scores = sorted([score_corner_patch(roi) for roi in corner_rois], reverse=True)
    corner_top2 = float(np.mean(corner_scores[:2]))
    corner_top3 = float(np.mean(corner_scores[:3]))

    row_ids = [int(warp_size * f) for f in (0.22, 0.50, 0.78)]
    col_ids = [int(warp_size * f) for f in (0.22, 0.50, 0.78)]
    line_scores = []
    for r in row_ids: line_scores.append(alternating_line_score(bw[max(0, min(warp_size-1, r)), :]))
    for c in col_ids: line_scores.append(alternating_line_score(bw[:, max(0, min(warp_size-1, c))]))
    timing_score = float(np.mean(sorted(line_scores, reverse=True)[:4]))

    pooled = cv2.resize((bw == 0).astype(np.uint8) * 255, (24, 24), interpolation=cv2.INTER_AREA)
    pooled = (pooled > 127).astype(np.uint8)
    checker_h = float(np.mean(pooled[:, 1:] != pooled[:, :-1]))
    checker_v = float(np.mean(pooled[1:, :] != pooled[:-1, :]))
    module_score = 0.5 * (checker_h + checker_v)

    return [
        geometric_info['ratio'], geometric_info['cos_val'],
        geometric_info['proximity'], geometric_info['size_variance'],
        geometric_info['edge_density_raw'],
        black_ratio, edge_density, balance_score, center_mean, center_std,
        geometric_info['poly_area_norm'],
        corner_top2, corner_top3, timing_score, module_score,
        geometric_info['avg_fp_width'], geometric_info['intruder_count'],
        geometric_info['dist_to_size'], geometric_info['hyp_ratio'], geometric_info['aspect_diag'],
    ]

# ==============================================================================
# PATTERN GROUPING & CLASSIFICATION
# ==============================================================================

def group_patterns_to_candidates(image, patterns, scale, img_w, img_h, img_diag):
    if len(patterns) < 3: return []
    candidates = []
    img_area = img_h * img_w + 1e-5
    warp_size = 96

    pending_aabbs = []
    pending_features = []
    pending_scores = []

    for pA in patterns:
        others_sorted = sorted([p for p in patterns if p != pA], key=lambda p: distance(pA, p))
        closest_neighbors = others_sorted[:20] 

        for pB, pC in itertools.combinations(closest_neighbors, 2):
            wA, wB, wC = pA[2], pB[2], pC[2]
            size_var = max(wA, wB, wC) / (min(wA, wB, wC) + 1e-5)
            if size_var > 2.2: continue 

            d_AB, d_AC, d_BC = distance(pA, pB), distance(pA, pC), distance(pB, pC)
            avg_w = (wA + wB + wC) / 3.0
            max_d = max(d_AB, d_AC, d_BC)
            min_d = min(d_AB, d_AC, d_BC)
            if max_d / avg_w > 35.0 or min_d / avg_w < 1.2: continue 

            edges_sorted = sorted([(d_AB, pC, pA, pB), (d_AC, pB, pA, pC), (d_BC, pA, pB, pC)], key=lambda x: x[0], reverse=True)
            _hyp, vertex, pt1, pt2 = edges_sorted[0]

            d1, d2 = distance(vertex, pt1), distance(vertex, pt2)
            if d1 == 0 or d2 == 0: continue

            ratio = min(d1, d2) / max(d1, d2)
            if ratio < 0.35: continue 
                
            cos_val = abs(angle_cos(vertex, pt1, pt2))
            if cos_val > 0.75: continue 

            p4 = (int(pt1[0] + pt2[0] - vertex[0]), int(pt1[1] + pt2[1] - vertex[1]))
            base_poly_np = np.array([vertex[:2], pt1[:2], p4, pt2[:2]], dtype=np.float32)
            fp_widths = [pA[2], pB[2], pC[2]]
            
            detect_poly = expand_polygon(base_poly_np, fp_widths, alpha=0.68)
            
            intruder_count = sum(1 for p in patterns if p not in (pA, pB, pC) and point_in_polygon(p, detect_poly))
            if intruder_count > 0: continue
            
            gray_w = warp_poly_gray(image, detect_poly, size=warp_size)
            if gray_w is None: continue
            blur_w = cv2.GaussianBlur(gray_w, (3, 3), 0)
            _, bw_w = cv2.threshold(blur_w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            center_crop = bw_w[20:76, 20:76]
            ed_raw = float(np.count_nonzero(cv2.Canny(center_crop, 100, 200)) / (center_crop.size + 1e-5))
            if ed_raw < 0.12: continue

            sys_score = ((d_AB + d_AC + d_BC) / (3.0 * img_diag)) - ed_raw + abs(1.0 - ratio)

            output_poly = expand_polygon(base_poly_np, fp_widths, alpha=1.13)
            pts = np.asarray(output_poly, dtype=np.float32).reshape(-1, 2)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            aabb_poly = np.array([
                [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
            ], dtype=np.float32)

            if RF_MODEL is not None:
                proximity = (d_AB + d_AC + d_BC) / (3.0 * img_diag)
                hyp_ratio = _hyp / (d1 + d2 + 1e-6)
                pts_ordered = order_points(detect_poly)
                diag1 = distance(pts_ordered[0], pts_ordered[2])
                diag2 = distance(pts_ordered[1], pts_ordered[3])
                aspect_diag = min(diag1, diag2) / (max(diag1, diag2) + 1e-6)

                geometric_info = {
                    'ratio': ratio, 'cos_val': cos_val, 'proximity': proximity,
                    'size_variance': size_var, 'edge_density_raw': ed_raw,
                    'poly_area_norm': polygon_area(detect_poly) / (img_area * scale * scale + 1e-6),
                    'avg_fp_width': avg_w / (max(img_h, img_w) + 1e-6),
                    'intruder_count': 0.0, 'dist_to_size': max_d / (avg_w + 1e-6),
                    'hyp_ratio': hyp_ratio, 'aspect_diag': aspect_diag,
                }
                
                feat_vec = extract_features(gray_w, bw_w, geometric_info)
                if feat_vec is None: continue

                pending_aabbs.append(aabb_poly)
                pending_features.append(feat_vec)
                pending_scores.append(sys_score)
            else:
                candidates.append({'score': sys_score, 'poly': aabb_poly})

    if RF_MODEL is not None and pending_features:
        X = np.array(pending_features, dtype=np.float32)
        probs = RF_MODEL.predict_proba(X)[:, 1]
        for aabb, prob, sys_s in zip(pending_aabbs, probs, pending_scores):
            if prob >= 0.5:
                candidates.append({'score': (1.0 - prob) + sys_s * 0.001, 'poly': aabb})

    return candidates

# ==============================================================================
# PIPELINE CONFIGURATION & EXECUTION
# ==============================================================================

CONFIGS = [
    {"scale": 1.0, "mode": "normal"},
    {"scale": 0.5, "mode": "normal"},
    {"scale": 0.5, "mode": "otsu"},
    {"scale": 1.5, "mode": "normal"},
    {"scale": 2.5, "mode": "normal"},
]

def detect_qr_in_image(img):
    original_h, original_w = img.shape[:2]
    working_img, base_scale = resize_for_inference(img, max_side=1200)
    img_h, img_w = working_img.shape[:2]
    img_diag = math.hypot(img_w, img_h) + 1e-5

    cached_img = {}
    all_candidates = []

    for config in CONFIGS:
        scale, mode = config["scale"], config["mode"]
        
        if scale not in cached_img:
            if scale == 1.0:
                cached_img[scale] = working_img.copy()
            else:
                h, w = working_img.shape[:2]
                interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
                cached_img[scale] = cv2.resize(working_img, (int(w * scale), int(h * scale)), interpolation=interp)
        
        working = cached_img[scale]
        patterns = detect_finder_patterns(working, scale=scale, mode=mode)
        candidates = group_patterns_to_candidates(working, patterns, scale, img_w, img_h, img_diag)

        for cand in candidates:
            total_scale = scale * base_scale
            scaled_poly = (cand['poly'] / total_scale).astype(np.float32)
            all_candidates.append({'score': cand['score'], 'poly': scaled_poly})

    if not all_candidates: return []

    all_candidates.sort(key=lambda x: x['score'])
    final_boxes = [order_points(all_candidates[0]['poly'])]
    
    for cand in all_candidates[1:]:
        box = order_points(cand['poly'])
        if not any(convex_overlap_ratio(box, fb) > 0.20 for fb in final_boxes):
            final_boxes.append(box)

    for i in range(len(final_boxes)):
        final_boxes[i][:, 0] = np.clip(final_boxes[i][:, 0], 0, original_w - 1)
        final_boxes[i][:, 1] = np.clip(final_boxes[i][:, 1], 0, original_h - 1)

    return final_boxes

# ==============================================================================
# QR Decoder Engine
# ==============================================================================

_GF_EXP = [0] * 512
_GF_LOG = [0] * 256

def _init_gf256():
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x >= 256:
            x ^= 0x11D
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]

_init_gf256()

def _gf_mul(a, b):
    if a == 0 or b == 0: return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]

def _gf_div(a, b):
    if b == 0: raise ZeroDivisionError
    if a == 0: return 0
    return _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]

def _gf_pow(x, n):
    if x == 0: return 0
    return _GF_EXP[(_GF_LOG[x] * n) % 255]

def _gf_poly_eval_low(poly, x):
    """Evaluate poly (low-to-high order) at x."""
    result = 0
    xi = 1
    for coeff in poly:
        result ^= _gf_mul(coeff, xi)
        xi = _gf_mul(xi, x)
    return result

def _gf_poly_eval_high(poly, x):
    result = poly[0]
    for i in range(1, len(poly)):
        result = _gf_mul(result, x) ^ poly[i]
    return result

def _rs_calc_syndromes(msg, nsym):
    return [_gf_poly_eval_high(msg, _GF_EXP[i]) for i in range(nsym)]

def _rs_berlekamp_massey(synd, nsym):
    lam = [1]  # low-to-high: Λ(x) = 1
    B = [1]
    L = 0
    m = 1
    b = 1
    for n in range(nsym):
        delta = synd[n]
        for i in range(1, L + 1):
            if i < len(lam) and n - i >= 0:
                delta ^= _gf_mul(lam[i], synd[n - i])
        if delta == 0:
            m += 1
        elif 2 * L <= n:
            T = list(lam)
            coeff = _gf_div(delta, b)
            shifted_B = [0] * m + [_gf_mul(coeff, c) for c in B]
            while len(shifted_B) < len(T): shifted_B.append(0)
            while len(T) < len(shifted_B): T.append(0)
            new_lam = [a ^ bb for a, bb in zip(T, shifted_B)]
            B = list(lam)
            lam = new_lam
            b = delta
            L = n + 1 - L
            m = 1
        else:
            coeff = _gf_div(delta, b)
            shifted_B = [0] * m + [_gf_mul(coeff, c) for c in B]
            while len(shifted_B) < len(lam): shifted_B.append(0)
            while len(lam) < len(shifted_B): lam.append(0)
            lam = [a ^ bb for a, bb in zip(lam, shifted_B)]
            m += 1
    return lam

def _rs_chien_search(lam, n):
    errs = len(lam) - 1
    if errs == 0: return []
    err_pos = []
    for i in range(n):
        power = (i - n + 1) % 255
        if _gf_poly_eval_low(lam, _GF_EXP[power]) == 0:
            err_pos.append(i)
    return err_pos if len(err_pos) == errs else None

def _rs_forney(synd, lam, err_pos, n):
    nsym = len(synd)
    omega = [0] * nsym
    for i in range(len(lam)):
        for j in range(nsym):
            if i + j < nsym:
                omega[i + j] ^= _gf_mul(lam[i], synd[j])
    lam_prime = [lam[k + 1] if k % 2 == 0 and k + 1 < len(lam) else 0 for k in range(max(1, len(lam) - 1))]
    err_vals = []
    for pos in err_pos:
        power_inv = (pos - n + 1) % 255
        xi_inv = _GF_EXP[power_inv]
        omega_val = _gf_poly_eval_low(omega, xi_inv)
        lp_val = _gf_poly_eval_low(lam_prime, xi_inv)
        if lp_val == 0: return None
        xi = _GF_EXP[(n - 1 - pos) % 255]
        err_vals.append(_gf_mul(xi, _gf_div(omega_val, lp_val)))
    return err_vals

def _rs_correct(msg_in, nsym):
    msg = list(msg_in)
    synd = _rs_calc_syndromes(msg, nsym)
    if max(synd) == 0: return msg[:-nsym]
    lam = _rs_berlekamp_massey(synd, nsym)
    errs = len(lam) - 1
    if errs == 0 or errs * 2 > nsym: return None
    err_pos = _rs_chien_search(lam, len(msg))
    if err_pos is None: return None
    err_vals = _rs_forney(synd, lam, err_pos, len(msg))
    if err_vals is None: return None
    for i, pos in enumerate(err_pos):
        msg[pos] ^= err_vals[i]
    if max(_rs_calc_syndromes(msg, nsym)) != 0: return None
    return msg[:-nsym]

_QR_EC = {
    (1,'L'):(7,1,19,0,0),(1,'M'):(10,1,16,0,0),(1,'Q'):(13,1,13,0,0),(1,'H'):(17,1,9,0,0),
    (2,'L'):(10,1,34,0,0),(2,'M'):(16,1,28,0,0),(2,'Q'):(22,1,22,0,0),(2,'H'):(28,1,16,0,0),
    (3,'L'):(15,1,55,0,0),(3,'M'):(26,1,44,0,0),(3,'Q'):(18,2,17,0,0),(3,'H'):(22,2,13,0,0),
    (4,'L'):(20,1,80,0,0),(4,'M'):(18,2,32,0,0),(4,'Q'):(26,2,24,0,0),(4,'H'):(16,4,9,0,0),
    (5,'L'):(26,1,108,0,0),(5,'M'):(24,2,43,0,0),(5,'Q'):(18,2,15,2,16),(5,'H'):(22,2,11,2,12),
    (6,'L'):(18,2,68,0,0),(6,'M'):(16,4,27,0,0),(6,'Q'):(24,4,19,0,0),(6,'H'):(28,4,15,0,0),
    (7,'L'):(20,2,78,0,0),(7,'M'):(18,4,31,0,0),(7,'Q'):(18,2,14,4,15),(7,'H'):(26,4,13,1,14),
    (8,'L'):(24,2,97,0,0),(8,'M'):(22,2,38,2,39),(8,'Q'):(22,4,18,2,19),(8,'H'):(26,4,14,2,15),
    (9,'L'):(30,2,116,0,0),(9,'M'):(22,3,36,2,37),(9,'Q'):(20,4,16,4,17),(9,'H'):(24,4,12,4,13),
    (10,'L'):(18,2,68,2,69),(10,'M'):(26,4,43,1,44),(10,'Q'):(24,6,19,2,20),(10,'H'):(28,6,15,2,16),
    (11,'L'):(20,4,81,0,0),(11,'M'):(30,1,50,4,51),(11,'Q'):(28,4,22,4,23),(11,'H'):(24,3,12,8,13),
    (12,'L'):(24,2,92,2,93),(12,'M'):(22,6,36,2,37),(12,'Q'):(26,4,20,6,21),(12,'H'):(28,7,14,4,15),
    (13,'L'):(26,4,107,0,0),(13,'M'):(22,8,37,1,38),(13,'Q'):(24,8,20,4,21),(13,'H'):(22,12,11,4,12),
    (14,'L'):(30,3,115,1,116),(14,'M'):(24,4,40,5,41),(14,'Q'):(20,11,16,5,17),(14,'H'):(24,11,12,5,13),
    (15,'L'):(22,5,87,1,88),(15,'M'):(24,5,41,5,42),(15,'Q'):(30,5,24,7,25),(15,'H'):(24,11,12,7,13),
    (16,'L'):(24,5,98,1,99),(16,'M'):(28,7,45,3,46),(16,'Q'):(24,15,19,2,20),(16,'H'):(30,3,15,13,16),
    (17,'L'):(28,1,107,5,108),(17,'M'):(28,10,46,1,47),(17,'Q'):(28,1,22,15,23),(17,'H'):(28,2,14,17,15),
    (18,'L'):(30,5,120,1,121),(18,'M'):(26,9,43,4,44),(18,'Q'):(28,17,22,1,23),(18,'H'):(28,2,14,19,15),
    (19,'L'):(28,3,113,4,114),(19,'M'):(26,3,44,11,45),(19,'Q'):(26,17,21,4,22),(19,'H'):(26,9,13,16,14),
    (20,'L'):(28,3,107,5,108),(20,'M'):(26,3,41,13,42),(20,'Q'):(30,15,24,5,25),(20,'H'):(28,15,15,10,16),
}

_QR_ALIGN = {
    2:[6,18],3:[6,22],4:[6,26],5:[6,30],6:[6,34],
    7:[6,22,38],8:[6,24,42],9:[6,26,46],10:[6,28,50],
    11:[6,30,54],12:[6,32,58],13:[6,34,62],
    14:[6,26,46,66],15:[6,26,48,70],16:[6,26,50,74],
    17:[6,30,54,78],18:[6,30,56,82],19:[6,30,58,86],20:[6,34,62,90],
}

_FORMAT_INFO_TABLE = []
def _init_format_info():
    mask = 0x5412
    for i in range(32):
        d = i << 10
        rem = d
        for j in range(5):
            if rem & (1 << (14 - j)):
                rem ^= (0x537 << (4 - j))
        _FORMAT_INFO_TABLE.append((d | rem) ^ mask)
_init_format_info()

def _detect_version(binary, qr_left, qr_top, qr_size):
    best_v, best_s = None, -1.0
    h, w = binary.shape[:2]
    for v in range(1, 21):
        modules = 4 * v + 17
        ms = qr_size / modules
        if ms < 1.5: continue
        score, count = 0, 0
        y = int(qr_top + 6.5 * ms)
        if 0 <= y < h:
            for col in range(8, modules - 8):
                x = int(qr_left + (col + 0.5) * ms)
                if 0 <= x < w:
                    if ((col % 2 == 0) == (binary[y, x] == 0)):
                        score += 1
                    count += 1
        x = int(qr_left + 6.5 * ms)
        if 0 <= x < w:
            for row in range(8, modules - 8):
                y = int(qr_top + (row + 0.5) * ms)
                if 0 <= y < h:
                    if ((row % 2 == 0) == (binary[y, x] == 0)):
                        score += 1
                    count += 1
        if count > 0:
            ns = score / count
            if ns > best_s:
                best_s = ns
                best_v = v
    return best_v if best_s >= 0.55 else None

def _sample_grid(binary, qr_left, qr_top, ms, modules):
    h, w = binary.shape[:2]
    grid = [[False] * modules for _ in range(modules)]
    off = max(1, int(ms * 0.15))
    for r in range(modules):
        cy = qr_top + (r + 0.5) * ms
        for c in range(modules):
            cx = qr_left + (c + 0.5) * ms
            y0, x0 = int(round(cy)), int(round(cx))
            dark = 0
            for dy in (-off, 0, off):
                for dx in (-off, 0, off):
                    py = max(0, min(h - 1, y0 + dy))
                    px = max(0, min(w - 1, x0 + dx))
                    if binary[py, px] == 0:
                        dark += 1
            grid[r][c] = (dark >= 5)
    return grid

def _read_format_info(grid, modules):
    bits1 = 0
    cols1 = [0, 1, 2, 3, 4, 5, 7, 8]
    for i, c in enumerate(cols1):
        if grid[8][c]: bits1 |= (1 << (14 - i))
    rows1 = [7, 5, 4, 3, 2, 1, 0]
    for i, r in enumerate(rows1):
        if grid[r][8]: bits1 |= (1 << (14 - 8 - i))
    
    bits2 = 0
    for i in range(7):
        if grid[modules - 1 - i][8]: bits2 |= (1 << (14 - i))
    for i in range(8):
        if grid[8][modules - 8 + i]: bits2 |= (1 << (14 - 7 - i))
    for bits in [bits1, bits2]:
        ec, mask = _decode_format_bits(bits)
        if ec is not None: return ec, mask
    return None, None

def _decode_format_bits(bits_15):
    min_d, best = 16, 0
    for d in range(32):
        dist = bin(bits_15 ^ _FORMAT_INFO_TABLE[d]).count('1')
        if dist < min_d:
            min_d = dist
            best = d
    if min_d > 3: return None, None
    ec_map = {0b01: 'L', 0b00: 'M', 0b11: 'Q', 0b10: 'H'}
    return ec_map.get((best >> 3) & 3), best & 7

# --- 7.7: Mask Patterns ---

def _mask_fn(p, r, c):
    if p == 0: return (r + c) % 2 == 0
    if p == 1: return r % 2 == 0
    if p == 2: return c % 3 == 0
    if p == 3: return (r + c) % 3 == 0
    if p == 4: return (r // 2 + c // 3) % 2 == 0
    if p == 5: return (r * c) % 2 + (r * c) % 3 == 0
    if p == 6: return ((r * c) % 2 + (r * c) % 3) % 2 == 0
    if p == 7: return ((r + c) % 2 + (r * c) % 3) % 2 == 0
    return False

# --- 7.8: Function Pattern Mask ---

def _func_mask(modules, version):
    mask = [[False] * modules for _ in range(modules)]
    for r in range(9):
        for c in range(9): mask[r][c] = True
    for r in range(9):
        for c in range(modules - 8, modules): mask[r][c] = True
    for r in range(modules - 8, modules):
        for c in range(9): mask[r][c] = True
    for i in range(8, modules - 8):
        mask[6][i] = True
        mask[i][6] = True
    if 4 * version + 9 < modules:
        mask[4 * version + 9][8] = True
    if version in _QR_ALIGN:
        for rr in _QR_ALIGN[version]:
            for cc in _QR_ALIGN[version]:
                if rr <= 8 and cc <= 8: continue
                if rr <= 8 and cc >= modules - 8: continue
                if rr >= modules - 8 and cc <= 8: continue
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        r2, c2 = rr + dr, cc + dc
                        if 0 <= r2 < modules and 0 <= c2 < modules:
                            mask[r2][c2] = True
    if version >= 7:
        for r in range(6):
            for c in range(modules - 11, modules - 8): mask[r][c] = True
        for r in range(modules - 11, modules - 8):
            for c in range(6): mask[r][c] = True
    return mask

# --- 7.9: Zigzag Data Extraction ---

def _extract_data_bits(grid, modules, version, mask_pattern):
    fm = _func_mask(modules, version)
    bits = []
    col = modules - 1
    going_up = True
    while col > 0:
        if col == 6: col -= 1
        rows = range(modules - 1, -1, -1) if going_up else range(modules)
        for row in rows:
            for dc in (0, -1):
                c = col + dc
                if c < 0 or c >= modules: continue
                if fm[row][c]: continue
                val = grid[row][c]
                if _mask_fn(mask_pattern, row, c):
                    val = not val
                bits.append(1 if val else 0)
        col -= 2
        going_up = not going_up
    return bits

# --- 7.10: Deinterleave ---

def _deinterleave(codewords, version, ec_level):
    key = (version, ec_level)
    if key not in _QR_EC: return None
    ec_pb, g1n, g1d, g2n, g2d = _QR_EC[key]
    blk_d = [g1d] * g1n + [g2d] * g2n
    total_b = g1n + g2n
    max_d = max(blk_d) if blk_d else 0
    idx = 0
    bd = [[] for _ in range(total_b)]
    for i in range(max_d):
        for b in range(total_b):
            if i < blk_d[b] and idx < len(codewords):
                bd[b].append(codewords[idx]); idx += 1
    be = [[] for _ in range(total_b)]
    for i in range(ec_pb):
        for b in range(total_b):
            if idx < len(codewords):
                be[b].append(codewords[idx]); idx += 1
    return [(bd[b], be[b]) for b in range(total_b)]

# --- 7.11: Payload Parsing ---

_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

def _read_bits(bits, pos, count):
    if pos + count > len(bits): return None, pos
    val = 0
    for i in range(count):
        val = (val << 1) | bits[pos + i]
    return val, pos + count

def _parse_payload(data_cw, version):
    bits = []
    for byte in data_cw:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    pos = 0
    result = []
    while pos + 4 <= len(bits):
        mode, pos = _read_bits(bits, pos, 4)
        if mode is None or mode == 0: break
        if mode == 0b0001:  # Numeric
            ccl = 10 if version <= 9 else (12 if version <= 26 else 14)
            cnt, pos = _read_bits(bits, pos, ccl)
            if cnt is None: break
            while cnt >= 3:
                v, pos = _read_bits(bits, pos, 10)
                if v is None: break
                result.append(f"{v:03d}"); cnt -= 3
            if cnt == 2:
                v, pos = _read_bits(bits, pos, 7)
                if v is not None: result.append(f"{v:02d}")
            elif cnt == 1:
                v, pos = _read_bits(bits, pos, 4)
                if v is not None: result.append(str(v))
        elif mode == 0b0010:  # Alphanumeric
            ccl = 9 if version <= 9 else (11 if version <= 26 else 13)
            cnt, pos = _read_bits(bits, pos, ccl)
            if cnt is None: break
            while cnt >= 2:
                v, pos = _read_bits(bits, pos, 11)
                if v is None: break
                c1, c2 = v // 45, v % 45
                if c1 < len(_ALNUM): result.append(_ALNUM[c1])
                if c2 < len(_ALNUM): result.append(_ALNUM[c2])
                cnt -= 2
            if cnt == 1:
                v, pos = _read_bits(bits, pos, 6)
                if v is not None and v < len(_ALNUM): result.append(_ALNUM[v])
        elif mode == 0b0100:  # Byte
            ccl = 8 if version <= 9 else 16
            cnt, pos = _read_bits(bits, pos, ccl)
            if cnt is None: break
            for _ in range(cnt):
                v, pos = _read_bits(bits, pos, 8)
                if v is None: break
                try: result.append(chr(v))
                except: result.append('?')
        elif mode == 0b0111:  # ECI (skip)
            if pos < len(bits) and bits[pos] == 0: pos += 8
            elif pos + 1 < len(bits) and bits[pos:pos+2] == [1, 0]: pos += 16
            else: pos += 24
        else:
            break
    return "".join(result)

def _sample_grid_aligned(binary, origin, dir_r, dir_c, ms, modules):
    """Sample grid using direction vectors from finder patterns (handles rotation)."""
    h, w = binary.shape[:2]
    grid = [[False] * modules for _ in range(modules)]
    off = max(1, int(ms * 0.15))
    for r in range(modules):
        for c in range(modules):
            px = origin[0] + (c + 0.5) * ms * dir_c[0] + (r + 0.5) * ms * dir_r[0]
            py = origin[1] + (c + 0.5) * ms * dir_c[1] + (r + 0.5) * ms * dir_r[1]
            x0, y0 = int(round(px)), int(round(py))
            dark = 0
            for dy in (-off, 0, off):
                for dx in (-off, 0, off):
                    sx = max(0, min(w - 1, x0 + dx))
                    sy = max(0, min(h - 1, y0 + dy))
                    if binary[sy, sx] == 0:
                        dark += 1
            grid[r][c] = (dark >= 5)
    return grid

def _decode_from_grid(grid, modules, version):
    """Try full decode pipeline from a sampled grid."""
    ec_level, mask_p = _read_format_info(grid, modules)
    if ec_level is None: return None
    if (version, ec_level) not in _QR_EC: return None

    data_bits = _extract_data_bits(grid, modules, version, mask_p)
    codewords = []
    for i in range(0, len(data_bits) - 7, 8):
        byte = 0
        for j in range(8): byte = (byte << 1) | data_bits[i + j]
        codewords.append(byte)

    blocks = _deinterleave(codewords, version, ec_level)
    if blocks is None: return None

    corrected = []
    for bd_b, be_b in blocks:
        c = _rs_correct(bd_b + be_b, len(be_b))
        if c is None: return None
        corrected.extend(c)

    content = _parse_payload(corrected, version)
    return content if content else None

def _decode_finder_aligned(warped_bgr, binary):
    patterns = None
    for mode in ["normal", "otsu"]:
        pats = detect_finder_patterns(warped_bgr, scale=1.0, mode=mode)
        if len(pats) >= 3:
            patterns = pats
            break
    if patterns is None or len(patterns) < 3:
        return None

    h, w = binary.shape[:2]

    triplets = []
    if len(patterns) > 3:
        for combo in itertools.combinations(patterns[:12], 3):
            td = distance(combo[0], combo[1]) + distance(combo[0], combo[2]) + distance(combo[1], combo[2])
            triplets.append((td, combo))
        triplets.sort(key=lambda x: x[0])
        triplets = [t[1] for t in triplets[:3]]
    else:
        triplets = [tuple(patterns[:3])]

    for triplet in triplets:
        pA, pB, pC = triplet
        d_AB, d_AC, d_BC = distance(pA, pB), distance(pA, pC), distance(pB, pC)

        edges = sorted([
            (d_AB, pC, pA, pB), (d_AC, pB, pA, pC), (d_BC, pA, pB, pC)
        ], key=lambda x: x[0], reverse=True)
        _, vertex, pt1, pt2 = edges[0]

        d1, d2 = distance(vertex, pt1), distance(vertex, pt2)
        if d1 < 10 or d2 < 10: continue

        dir1 = np.array([(pt1[0] - vertex[0]) / d1, (pt1[1] - vertex[1]) / d1])
        dir2 = np.array([(pt2[0] - vertex[0]) / d2, (pt2[1] - vertex[1]) / d2])

        for swap in [False, True]:
            if swap:
                cur_dir1, cur_dir2 = dir2.copy(), dir1.copy()
                cur_d1, cur_d2 = d2, d1
            else:
                cur_dir1, cur_dir2 = dir1.copy(), dir2.copy()
                cur_d1, cur_d2 = d1, d2

            cross = cur_dir1[0] * cur_dir2[1] - cur_dir1[1] * cur_dir2[0]
            if cross < 0:
                cur_dir1, cur_dir2 = cur_dir2, cur_dir1
                cur_d1, cur_d2 = cur_d2, cur_d1

            dir_c, dir_r = cur_dir1, cur_dir2

            avg_side = (cur_d1 + cur_d2) / 2.0
            for v in range(1, 21):
                modules = 4 * v + 17
                expected_side = modules - 7
                ms = avg_side / expected_side

                if ms < 1.5 or ms > 40: continue

                origin = np.array([float(vertex[0]), float(vertex[1])]) - 3.5 * ms * dir_c - 3.5 * ms * dir_r

                grid = _sample_grid_aligned(binary, origin, dir_r, dir_c, ms, modules)
                content = _decode_from_grid(grid, modules, v)
                if content: return content

    return None

def _find_qr_bounds(binary):
    h, w = binary.shape[:2]
    thr = max(3, min(h, w) // 40)
    top, bottom, left, right = 0, h - 1, 0, w - 1
    for r in range(h):
        if np.count_nonzero(binary[r, :] == 0) > thr:
            top = r; break
    for r in range(h - 1, -1, -1):
        if np.count_nonzero(binary[r, :] == 0) > thr:
            bottom = r; break
    for c in range(w):
        if np.count_nonzero(binary[:, c] == 0) > thr:
            left = c; break
    for c in range(w - 1, -1, -1):
        if np.count_nonzero(binary[:, c] == 0) > thr:
            right = c; break
    return top, bottom, left, right

def _attempt_decode(binary, left, top, qr_size, version):
    """Try decoding with specific parameters. Returns content str or None."""
    modules = 4 * version + 17
    ms = qr_size / modules
    if ms < 1.5: return None
    grid = _sample_grid(binary, left, top, ms, modules)
    return _decode_from_grid(grid, modules, version)

def _try_decode_boundary(gray, rotation=0):
    try:
        if rotation == 1: gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 2: gray = cv2.rotate(gray, cv2.ROTATE_180)
        elif rotation == 3: gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

        binarizations = []
        bl = cv2.GaussianBlur(gray, (3, 3), 0)
        _, b1 = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binarizations.append(b1)
        bl2 = cv2.GaussianBlur(gray, (5, 5), 0)
        b2 = cv2.adaptiveThreshold(bl2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        binarizations.append(b2)
        _, b3 = cv2.threshold(gray, int(np.median(gray) * 0.85), 255, cv2.THRESH_BINARY)
        binarizations.append(b3)

        for binary in binarizations:
            top, bottom, left, right = _find_qr_bounds(binary)
            qr_w, qr_h = right - left, bottom - top
            if qr_w < 20 or qr_h < 20: continue
            base_qr_size = (qr_w + qr_h) / 2.0
            detected_v = _detect_version(binary, left, top, base_qr_size)
            candidate_vs = []
            if detected_v:
                for dv in [0, -1, 1]:
                    vv = detected_v + dv
                    if 1 <= vv <= 20: candidate_vs.append(vv)
            else:
                candidate_vs = list(range(1, 11))
            for dl, dt in [(0, 0), (-3, -3), (3, 3), (-3, 3), (3, -3)]:
                adj_left = max(0, left + dl)
                adj_top = max(0, top + dt)
                adj_right = min(binary.shape[1] - 1, right - dl)
                adj_bottom = min(binary.shape[0] - 1, bottom - dt)
                qr_size = (adj_right - adj_left + adj_bottom - adj_top) / 2.0
                if qr_size < 20: continue
                for v in candidate_vs:
                    content = _attempt_decode(binary, adj_left, adj_top, qr_size, v)
                    if content: return content
    except Exception:
        pass
    return ""

def decode_qr_content(image, poly):
    warp_size = 500
    src = order_points(poly).astype(np.float32)
    dst = np.array([[0, 0], [warp_size-1, 0], [warp_size-1, warp_size-1], [0, warp_size-1]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (warp_size, warp_size))
    except cv2.error:
        return ""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped

    binarizations = []
    bl = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b1 = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binarizations.append(b1)
    bl2 = cv2.GaussianBlur(gray, (5, 5), 0)
    b2 = cv2.adaptiveThreshold(bl2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    binarizations.append(b2)
    _, b3 = cv2.threshold(gray, int(np.median(gray) * 0.85), 255, cv2.THRESH_BINARY)
    binarizations.append(b3)

    for binary in binarizations:
        try:
            content = _decode_finder_aligned(warped, binary)
            if content: return content
        except Exception:
            pass

    for rot in range(4):
        content = _try_decode_boundary(gray.copy(), rot)
        if content: return content
    return ""

# ==============================================================================
# MULTIPROCESSING PIPELINE
# ==============================================================================

def process_single_row(args):
    row, base_dir, debug_dir, run_decode = args
    img_id, img_path = row.get("image_id", "").strip(), row.get("image_path", "").strip()
    
    empty_res = [[img_id, "", "", "", "", "", "", "", "", "", ""]]
    if not img_id or not img_path: return empty_res

    full_path = os.path.join(base_dir, img_path)
    img = cv2.imread(full_path)
    if img is None: return empty_res

    boxes = detect_qr_in_image(img)
    
    if debug_dir is not None:
        img_draw = img.copy()
        if boxes:
            for idx, box in enumerate(boxes):
                pts = np.array(box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_draw, [pts], True, (0, 255, 0), 3)
                top_left_pt = tuple(pts[0][0])
                cv2.circle(img_draw, top_left_pt, 8, (0, 0, 255), -1)
                cv2.putText(img_draw, f"QR {idx}", (top_left_pt[0] + 10, top_left_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        save_path = os.path.join(debug_dir, f"{img_id}.jpg")
        cv2.imwrite(save_path, img_draw)

    if not boxes: return empty_res
    results = []
    for idx, box in enumerate(boxes):
        content = ""
        if run_decode:
            try:
                content = decode_qr_content(img, box)
            except Exception:
                pass
        results.append([
            img_id, idx,
            float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1]),
            float(box[2][0]), float(box[2][1]), float(box[3][0]), float(box[3][1]), content
        ])
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Đường dẫn file CSV")
    parser.add_argument("--model", default="qr_rf_model.joblib", help="Đường dẫn mô hình Joblib")
    parser.add_argument("--workers", type=int, default=0, help="Số worker (0 = auto)")
    parser.add_argument("--chunksize", type=int, default=0, help="Chunksize (0 = auto)")
    parser.add_argument("--vis", action="store_true", help="Kích hoạt chế độ vẽ debug")
    parser.add_argument("--decode", type=str, choices=["yes", "no"], default="yes", help="Bật/tắt giải mã QR (yes/no)")
    args = parser.parse_args()

    input_csv = args.data
    base_dir = os.path.dirname(os.path.abspath(input_csv))
    model_path = os.path.abspath(args.model)
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.csv")

    debug_dir = None
    if args.vis:
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_visuals")
        os.makedirs(debug_dir, exist_ok=True)

    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Xử lý tự động CPU cores
    cpu_count = os.cpu_count() or 1
    if args.workers > 0:
        num_workers = max(1, min(args.workers, len(rows) or 1))
    else:
        auto_cap = 20 if os.name == "nt" else cpu_count
        num_workers = max(1, min(cpu_count, auto_cap, len(rows) or 1))

    if args.chunksize > 0:
        chunksize = args.chunksize
    else:
        chunksize = 4 if (os.name == "nt" and len(rows) >= 200) else max(1, len(rows) // (num_workers * 4))

    print(f"[INFO] Khởi động QR Detection Engine (RF-ML + Heuristic)...")
    print(f"[INFO] Cấu hình: {num_workers} workers | chunksize={chunksize} | Cache Resize: ON")
    if not os.path.exists(model_path):
        print(f"[WARNING] Không tìm thấy model '{model_path}'. Chuyển sang chế độ Heuristic Fallback.")

    t0 = time.time()
    run_decode = (args.decode.lower() == "yes")
    worker_args = [(row, base_dir, debug_dir, run_decode) for row in rows]
    all_results = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(model_path,)
    ) as executor:
        for res in tqdm(executor.map(process_single_row, worker_args, chunksize=chunksize), total=len(rows)):
            all_results.extend(res)

    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["image_id", "qr_index", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "content"])
        writer.writerows(all_results)

    decoded_count = sum(1 for res in all_results if len(res) > 10 and res[10])

    elapsed = time.time() - t0
    print(f"[DONE] Hoàn tất xử lý {len(rows)} ảnh trong {elapsed:.2f}s (~{elapsed/max(1, len(rows)):.3f}s/ảnh).")
    print(f"[STAT] Số QR giải mã thành công: {decoded_count}")

if __name__ == "__main__":
    main()