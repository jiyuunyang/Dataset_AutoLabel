# -*- coding: utf-8 -*-
"""
사진 폴더 → 사람 bbox + 자세(0=lie 눕기, 1=sit 앉기, 2=stand 서기) 자동 라벨

  python autolabel.py 사진폴더                 # 사진폴더_autolabel/ 에 결과
  python autolabel.py 사진폴더 --out 작업폴더

방법 (학습 없음, 기성 모델 + 규칙)
  1) 사람 검출 모델 (YOLO11 detect, COCO person) → 사람 bbox
  2) 포즈 관절 모델 (YOLO11 pose, COCO 17관절)   → 어깨·엉덩이·무릎·발목 좌표
  3) 두 결과를 IoU 로 짝지은 뒤, 몸통 기울기·다리 접힘·박스 비율 규칙으로 자세 판단 (classify)
  규칙 판단은 정답이 아니다. 점수(conf)가 낮거나 관절이 안 보이는 사람은 검수 화면에서 '확인 필요'로 뜬다.

결과 폴더
  images/*.jpg      원본 사진 복사 (원본 폴더는 건드리지 않음)
  labels/*.txt      YOLO 라벨: class cx cy w h (0~1)
  _auto/*.json      판단 근거 (관절 좌표, 각도, 점수) — 검수 화면이 사용
  data.yaml         names: ['lie', 'sit', 'stand']
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, 'models')
NAMES = ['lie', 'sit', 'stand']
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SKIP_DIRS = {'_removed', '_backup', '_auto'}

# COCO 17 관절 번호
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 0, 5, 6, 11, 12, 13, 14, 15, 16
KP_CONF = 0.4  # 이 값 이상인 관절만 '보임'으로 침


# ---------------------------------------------------------------- 규칙 (자세 판단)
def _mid(kp, a, b):
    """두 관절의 가운데. 한쪽만 보이면 그쪽"""
    pa, pb = kp[a], kp[b]
    va, vb = pa[2] >= KP_CONF, pb[2] >= KP_CONF
    if va and vb:
        return (pa[:2] + pb[:2]) / 2
    if va:
        return pa[:2].copy()
    if vb:
        return pb[:2].copy()
    return None


def _angle_from_vertical(v):
    """벡터가 세로축과 이루는 각도 0~90° (위/아래 방향 구분 없음)"""
    return math.degrees(math.atan2(abs(v[0]), abs(v[1]) + 1e-9))


def _knee_angle(kp, hip, knee, ank):
    """엉덩이-무릎-발목 사이 각도. 180°=다리 펴짐"""
    if min(kp[hip][2], kp[knee][2], kp[ank][2]) < KP_CONF:
        return None
    a, b, c = kp[hip][:2], kp[knee][:2], kp[ank][:2]
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def features(box, kp):
    """box: [x1,y1,x2,y2] 픽셀, kp: (17,3) [x,y,conf] 또는 None → 판단에 쓰는 값들"""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    f = {'aspect': w / h, 'n_kp': 0}
    if kp is None:
        return f
    kp = np.asarray(kp, dtype=float)
    vis = kp[:, 2] >= KP_CONF
    f['n_kp'] = int(vis.sum())
    sh, hip = _mid(kp, L_SH, R_SH), _mid(kp, L_HIP, R_HIP)
    if sh is not None and hip is not None:
        t = sh - hip
        f['torso_len'] = float(np.linalg.norm(t))
        f['torso_deg'] = _angle_from_vertical(t)  # 0=곧게 섬, 90=가로로 누움
        f['torso_up'] = bool(t[1] < 0)  # 어깨가 엉덩이보다 위
    knees = [a for a in (_knee_angle(kp, L_HIP, L_KNEE, L_ANK), _knee_angle(kp, R_HIP, R_KNEE, R_ANK)) if a is not None]
    if knees:
        f['knee_deg'] = float(max(knees))  # 더 펴진 쪽 (한쪽만 굽힌 서기 자세 보호)
    thighs = []
    for a, b in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        if kp[a][2] >= KP_CONF and kp[b][2] >= KP_CONF:
            thighs.append(_angle_from_vertical(kp[b][:2] - kp[a][:2]))
    if thighs:
        f['thigh_deg'] = float(min(thighs))  # 0=허벅지 세로(서기), 90=가로(앉기)
    ank = _mid(kp, L_ANK, R_ANK)
    if hip is not None and ank is not None and f.get('torso_len'):
        f['leg_ratio'] = float((ank[1] - hip[1]) / f['torso_len'])  # 엉덩이→발목 세로 길이 / 몸통 길이
    pts = kp[vis][:, :2]
    if len(pts) >= 5:  # 보이는 관절 전체의 주축 기울기 (누운 사람에 강함)
        c = pts - pts.mean(0)
        ev, evec = np.linalg.eigh(c.T @ c)
        f['body_deg'] = _angle_from_vertical(evec[:, 1])
        f['body_elong'] = float(math.sqrt(ev[1] / (ev[0] + 1e-9)))
    return f


# 기준값: evaluate.py 로 cctv2 정답 데이터(train)에 맞춰 정함 (HOW_TO_USE.md 참고)
# 각 단서는 '가운데 값' ± 폭 사이에서 0→1 로 바뀌는 투표를 한다. 가중치는 단서를 얼마나 믿을지.
RULE = {
    'lie_aspect': 1.1,       # 박스 가로/세로가 이보다 크면 눕기 쪽
    'lie_torso_deg': 60,     # 몸통(어깨-엉덩이)이 세로에서 이만큼 넘게 기울면 눕기 쪽
    'lie_body_deg': 60,      # 보이는 관절 전체의 주축 기울기
    'lie_leg_ratio': 0.6,    # 엉덩이→발목 높이가 몸통 길이의 이 배보다 짧으면 눕기/앉기 쪽
    'sit_aspect': 0.7,       # 서기보다 앉기가 박스가 덜 길쭉함
    'sit_knee_deg': 130,     # 무릎이 이보다 굽으면 앉기 쪽
    'sit_thigh_deg': 30,     # 허벅지가 세로에서 이만큼 넘게 누우면 앉기 쪽
    'sit_leg_ratio': 0.9,    # 엉덩이→발목 높이가 몸통의 이 배보다 짧으면 앉기 쪽
    'w_aspect': 1.0,         # 박스 비율 가중치 (관절보다 덜 틀림)
    'review_conf': 0.75,     # 이 점수 미만은 '확인 필요' (cctv2: 사람의 26% 표시, 틀린 것의 67% 포함)
}


def _ramp(x, mid, half):
    """x 가 mid-half 이하면 0, mid+half 이상이면 1, 사이는 직선"""
    return max(0.0, min(1.0, (x - mid + half) / (2 * half)))


def _vote(votes):
    """[(점수, 가중치)] → 가중 평균"""
    return sum(v * w for v, w in votes) / sum(w for _, w in votes)


def classify(f, r=RULE):
    """특징 → (class, conf 0~1, 근거 문자열). 1) 눕기인지  2) 아니면 앉기/서기"""
    why = [f"박스 가로/세로 {f['aspect']:.2f}"]
    lie = [(_ramp(f['aspect'], r['lie_aspect'], 0.3), r['w_aspect'])]
    if 'torso_deg' in f:
        lie.append((_ramp(f['torso_deg'], r['lie_torso_deg'], 15), 1.0))
        why.append(f"몸통 {f['torso_deg']:.0f}°")
        if not f['torso_up'] and f['torso_deg'] < 45:  # 어깨가 엉덩이보다 아래 = 머리가 아래로 쓰러짐
            lie.append((1.0, 1.0))
            why.append('머리가 아래')
    if 'body_deg' in f and f['body_elong'] > 1.5:
        lie.append((_ramp(f['body_deg'], r['lie_body_deg'], 15), 1.0))
        why.append(f"몸 축 {f['body_deg']:.0f}°")
    if 'leg_ratio' in f:
        lie.append((1 - _ramp(f['leg_ratio'], r['lie_leg_ratio'], 0.3), 0.5))
        why.append(f"다리/몸통 {f['leg_ratio']:.2f}")
    p_lie = _vote(lie)
    if p_lie >= 0.5:
        return 0, p_lie, ', '.join(why)
    sit = [(1 - _ramp(1 / f['aspect'], 1 / r['sit_aspect'], 0.4), r['w_aspect'])]
    if 'knee_deg' in f:
        sit.append((1 - _ramp(f['knee_deg'], r['sit_knee_deg'], 20), 1.0))
        why.append(f"무릎 {f['knee_deg']:.0f}°")
    if 'thigh_deg' in f:
        sit.append((_ramp(f['thigh_deg'], r['sit_thigh_deg'], 15), 1.0))
        why.append(f"허벅지 {f['thigh_deg']:.0f}°")
    if 'leg_ratio' in f:
        sit.append((1 - _ramp(f['leg_ratio'], r['sit_leg_ratio'], 0.2), 1.0))
    if len(sit) == 1:
        why.append('하체 관절 안 보임')
    p_sit = _vote(sit)
    # 확신도: 고른 쪽 점수와 '눕기가 아님' 점수 중 작은 값
    if p_sit >= 0.5:
        return 1, min(p_sit, 1 - p_lie), ', '.join(why)
    return 2, min(1 - p_sit, 1 - p_lie), ', '.join(why)


# ---------------------------------------------------------------- 모델
def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Analyzer:
    def __init__(self, det='yolo11m.pt', pose='yolo11m-pose.pt', conf=0.35, imgsz=960, device=None):
        from ultralytics import YOLO  # 무거우므로 실제로 쓸 때 불러옴
        here = os.getcwd()
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.chdir(MODEL_DIR)  # 가중치가 없으면 models/ 에 내려받게
        try:
            self.det = YOLO(det)
            self.pose = YOLO(pose)
        finally:
            os.chdir(here)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def _run(self, model, path, **kw):
        return model.predict(path, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False, **kw)[0]

    def analyze(self, path):
        """사진 한 장 → {'w','h','persons':[{box,det_conf,kp,src,cls,conf,why,feat}]}"""
        d = self._run(self.det, path, classes=[0])
        p = self._run(self.pose, path)
        h, w = d.orig_shape
        dets = [(b.tolist(), float(c)) for b, c in zip(d.boxes.xyxy.cpu().numpy(), d.boxes.conf.cpu().numpy())]
        poses = []
        if p.keypoints is not None and len(p.boxes):
            kxy = p.keypoints.data.cpu().numpy()  # (n,17,3)
            for b, c, k in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy(), kxy):
                poses.append((b.tolist(), float(c), k))
        persons, used = [], set()
        for box, c in dets:  # 검출 박스마다 IoU 가 가장 큰 포즈를 붙임
            best, bi = 0.0, -1
            for j, (pb, _, _) in enumerate(poses):
                if j not in used:
                    v = _iou(box, pb)
                    if v > best:
                        best, bi = v, j
            kp = None
            if best >= 0.45:
                used.add(bi)
                kp = poses[bi][2]
            persons.append({'box': box, 'det_conf': c, 'kp': kp, 'src': 'det+pose' if kp is not None else 'det'})
        for j, (pb, c, k) in enumerate(poses):  # 검출 모델이 놓친 사람 (주로 누운 사람)
            if j not in used and all(_iou(pb, q['box']) < 0.45 for q in persons):
                persons.append({'box': pb, 'det_conf': c, 'kp': k, 'src': 'pose'})
        for q in persons:
            q['box'] = [max(0.0, q['box'][0]), max(0.0, q['box'][1]), min(float(w), q['box'][2]), min(float(h), q['box'][3])]
            q['feat'] = features(q['box'], q['kp'])
            q['cls'], q['conf'], q['why'] = classify(q['feat'])
            q['conf'] = round(q['conf'] * min(1.0, q['det_conf'] / 0.6), 3)  # 사람 검출이 약하면 같이 낮춤
            q['kp'] = None if q['kp'] is None else np.round(q['kp'], 2).tolist()
        return {'w': w, 'h': h, 'persons': persons}


# ---------------------------------------------------------------- 저장
def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    return f'{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}'


def write_result(root, name, res):
    stem = os.path.splitext(name)[0]
    os.makedirs(os.path.join(root, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(root, '_auto'), exist_ok=True)
    with open(os.path.join(root, 'labels', stem + '.txt'), 'w', encoding='utf-8', newline='\n') as f:
        for q in res['persons']:
            f.write(yolo_line(q['cls'], q['box'], res['w'], res['h']) + '\n')
    res = dict(res, image=name, time=time.strftime('%Y-%m-%d %H:%M:%S'), rule=RULE)
    with open(os.path.join(root, '_auto', stem + '.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)


def write_yaml(root):
    p = os.path.join(root, 'data.yaml')
    if not os.path.isfile(p):
        with open(p, 'w', encoding='utf-8', newline='\n') as f:
            f.write('path: .\ntrain: images\nval: images\n\nnc: 3\nnames: ' + json.dumps(NAMES) + '\n')


def list_images(folder):
    out = []
    for dp, dn, fn in os.walk(folder):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        out += [os.path.join(dp, f) for f in sorted(fn) if os.path.splitext(f)[1].lower() in IMG_EXT]
    return out


def import_images(src, root):
    """원본 폴더의 사진을 root/images 로 복사. 하위 폴더 이름은 파일 이름 앞에 붙임 (겹침 방지)"""
    dst = os.path.join(root, 'images')
    os.makedirs(dst, exist_ok=True)
    copied = skipped = 0
    for p in list_images(src):
        rel = os.path.relpath(p, src).replace('\\', '/')
        name = rel.replace('/', '__')
        d = os.path.join(dst, name)
        if os.path.exists(d):
            skipped += 1
            continue
        shutil.copy2(p, d)
        copied += 1
    return copied, skipped


def run(root, analyzer, names=None, redo=False, progress=None):
    """root/images 의 사진을 분석. redo=False 면 이미 분석한(_auto/*.json 있는) 사진은 건너뜀"""
    img_dir = os.path.join(root, 'images')
    if names is None:
        names = [f for f in sorted(os.listdir(img_dir)) if os.path.splitext(f)[1].lower() in IMG_EXT]
    write_yaml(root)
    todo = [n for n in names if redo or not os.path.isfile(os.path.join(root, '_auto', os.path.splitext(n)[0] + '.json'))]
    for i, n in enumerate(todo):
        try:
            write_result(root, n, analyzer.analyze(os.path.join(img_dir, n)))
        except Exception as e:  # 깨진 사진 하나 때문에 전체가 멈추지 않게
            print('실패:', n, e, file=sys.stderr)
        if progress:
            progress(i + 1, len(todo), n)
    return len(todo)


def main():
    ap = argparse.ArgumentParser(description='사진 폴더 자동 라벨 (0=lie 1=sit 2=stand)')
    ap.add_argument('folder', help='사진 폴더')
    ap.add_argument('--out', help='결과 폴더 (기본: 사진폴더_autolabel)')
    ap.add_argument('--conf', type=float, default=0.35, help='사람 검출 최소 점수')
    ap.add_argument('--imgsz', type=int, default=960, help='모델 입력 크기 (작은 사람이 많으면 크게)')
    ap.add_argument('--redo', action='store_true', help='이미 분석한 사진도 다시')
    a = ap.parse_args()
    src = os.path.realpath(a.folder)
    root = os.path.realpath(a.out or src.rstrip('\\/') + '_autolabel')
    c, s = import_images(src, root)
    print(f'사진 복사 {c}장 (이미 있음 {s}장) → {root}')
    an = Analyzer(conf=a.conf, imgsz=a.imgsz)
    t = time.time()
    n = run(root, an, redo=a.redo, progress=lambda i, n, f: print(f'\r{i}/{n} {f[:50]:50s}', end='', flush=True))
    print(f'\n분석 {n}장, {time.time() - t:.1f}초. 검수: python app.py "{root}"')


if __name__ == '__main__':
    main()
