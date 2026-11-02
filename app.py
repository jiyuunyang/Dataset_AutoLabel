# -*- coding: utf-8 -*-
"""
자동 라벨 + 검수 화면 (브라우저에서 열림, 인터넷 필요 없음)

  python app.py            → http://127.0.0.1:8770

순서
  1) 사진 넣기   : 내 컴퓨터 폴더 경로를 고르거나, 폴더/사진을 끌어다 놓기 → work/작업이름/images 에 복사
  2) 자동 분석   : 사람 검출 + 포즈 관절 모델 → bbox 와 자세(0=lie 눕기, 1=sit 앉기, 2=stand 서기)
  3) 검수·수정   : 박스 이동·크기 조절·새로 그리기·삭제, 클래스 바꾸기, 단축키
  4) 내보내기    : 검수한 사진만 YOLO 데이터셋 (train/valid/test + data.yaml) + zip

작업 폴더(work/작업이름)는 images/ labels/ data.yaml 구조라서 dedup_tools 의 review_ui.py 로도 열린다.
라벨을 고치기 전에는 그날 처음 상태를 _backup/날짜/ 에 복사해 둔다. 사진을 지우는 기능은 없다 (_removed 로 옮김).
"""
import argparse
import io
import json
import os
import random
import re
import secrets
import shutil
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

import autolabel as A

WORK_DIR = os.path.join(A.HERE, 'work')
EXPORT_DIR = os.path.join(A.HERE, 'exports')
TRASH_DIR = os.path.join(A.HERE, 'trash')  # 삭제한 프로젝트
KEY = None
LOCK = threading.Lock()
THUMB_CACHE = {}
SIZE_CACHE = {}
JOB = {'busy': False, 'kind': '', 'work': '', 'done': 0, 'total': 0, 'msg': '', 'err': '', 'seq': 0}  # seq: 시작할 때마다 1 증가
ANALYZER = {'key': None, 'obj': None}


# ---------------------------------------------------------------- 작업 폴더
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def work_root(name):
    name = (name or '').strip()
    if not name or name in ('.', '..') or re.search(r'[\\/:*?"<>|]', name):
        raise ValueError('프로젝트 이름에 쓸 수 없는 글자가 있어요 (\\ / : * ? " < > |): ' + name)
    return os.path.join(WORK_DIR, name)


def safe_name(n):
    n = os.path.basename((n or '').replace('\\', '/'))
    if not n or n.startswith('.'):
        raise ValueError('잘못된 파일 이름: ' + n)
    return n


def images_of(root, sub='images'):
    d = os.path.join(root, sub)
    if not os.path.isdir(d):
        return []
    return sorted((f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in A.IMG_EXT), key=natural_key)


def stem(n):
    return os.path.splitext(n)[0]


def img_size(path):
    key = (path, os.path.getmtime(path))
    if key not in SIZE_CACHE:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # 휴대폰 사진 회전 정보 반영 (모델 입력과 같게)
            SIZE_CACHE[key] = im.size
    return SIZE_CACHE[key]


def read_label(root, n, w, h):
    p = os.path.join(root, 'labels', stem(n) + '.txt')
    if not os.path.isfile(p):
        return None
    out = []
    for line in open(p, encoding='utf-8', errors='replace'):
        t = line.split()
        if len(t) != 5:
            continue
        try:
            c, cx, cy, bw, bh = int(float(t[0])), *map(float, t[1:])
        except ValueError:
            continue
        out.append({'c': c, 'box': [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return out


def load_auto(root, n):
    p = os.path.join(root, '_auto', stem(n) + '.json')
    if os.path.isfile(p):
        try:
            return json.load(open(p, encoding='utf-8'))
        except ValueError:
            return None
    return None


def review_path(root):
    return os.path.join(root, '_auto', 'review.json')


def load_review(root):
    p = review_path(root)
    return json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {}


def save_review(root, rv):
    os.makedirs(os.path.join(root, '_auto'), exist_ok=True)
    tmp = review_path(root) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rv, f, ensure_ascii=False, indent=0)
    os.replace(tmp, review_path(root))


def backup_label(root, n):
    """그날 처음 고치기 전 라벨을 _backup/날짜/labels/ 에 한 번만 복사"""
    s = os.path.join(root, 'labels', stem(n) + '.txt')
    if os.path.isfile(s):
        d = os.path.join(root, '_backup', time.strftime('%Y%m%d'), 'labels', stem(n) + '.txt')
        if not os.path.exists(d):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def image_state(root, n, rv):
    auto = load_auto(root, n)
    r = rv.get(n, {})
    if auto:
        w, h = auto['w'], auto['h']
    else:
        try:
            w, h = img_size(os.path.join(root, 'images', n))
        except Exception:  # 깨진 사진 하나 때문에 목록 전체가 안 열리지 않게
            return {'n': n, 'w': 0, 'h': 0, 'anns': [], 'has_label': False, 'analyzed': False,
                    'reviewed': bool(r.get('reviewed')), 'need': False, 'broken': True}
    anns = read_label(root, n, w, h)
    persons = auto['persons'] if auto else []
    for a in anns or []:  # 자동 결과와 거의 같은 박스면 자동 점수·근거를 붙여 보여 줌
        best, bq = 0.9, None
        for q in persons:
            v = A._iou(a['box'], q['box'])
            if v >= best:
                best, bq = v, q
        if bq:
            a['auto'] = {'c': bq['cls'], 'conf': bq['conf'], 'why': bq['why'], 'src': bq['src']}
    low = [a for a in anns or [] if a.get('auto') and a['auto']['c'] == a['c'] and a['auto']['conf'] < A.RULE['review_conf']]
    return {'n': n, 'w': w, 'h': h, 'anns': anns or [], 'has_label': anns is not None,
            'analyzed': auto is not None, 'reviewed': bool(r.get('reviewed')),
            'need': bool(auto) and not r.get('reviewed') and (not anns or bool(low))}


# ---------------------------------------------------------------- API
def image_ok(data=None, path=None):
    """사진 파일로 열리는지 (올리기·가져오기 때 깨진 파일을 거름)"""
    try:
        with Image.open(io.BytesIO(data) if data is not None else path) as im:
            im.verify()
        return True
    except Exception:
        return False


def stems_of(root):
    """images/ 와 _removed/images/ 의 이름(확장자 뺀 것, 대소문자 무시) — a.jpg 와 a.png 가 labels/a.txt 를 같이 쓰지 않게"""
    return {stem(n).lower() for sub in ('images', '_removed/images') for n in images_of(root, sub)}


def label_candidates(rel):
    """사진 상대 경로 → 라벨 상대 경로 후보 (YOLO 구조 .../labels/이름.txt 먼저, 그다음 같은 폴더의 이름.txt)"""
    parts = rel.replace('\\', '/').split('/')
    parts[-1] = stem(parts[-1]) + '.txt'
    out = []
    for k in range(len(parts) - 2, -1, -1):
        if parts[k].lower() == 'images':
            out.append('/'.join(parts[:k] + ['labels'] + parts[k + 1:]))
            break
    return out + ['/'.join(parts)]


def image_stems_for_label(rel):
    """라벨 상대 경로 → 짝이 되는 사진의 저장 이름(확장자 뺀 것) 후보 (label_candidates 의 반대)"""
    parts = rel.replace('\\', '/').split('/')
    parts[-1] = stem(parts[-1])
    out = []
    for k in range(len(parts) - 2, -1, -1):
        if parts[k].lower() == 'labels':
            out.append('__'.join(parts[:k] + ['images'] + parts[k + 1:]))
            break
    return out + ['__'.join(parts)]


def can_take_label(root, n, rv):
    """가져온 라벨을 넣어도 되는 사진: 라벨·분석 결과·검수 기록이 아직 없음 (프로젝트에서 만든 라벨을 덮어쓰지 않게)"""
    return not os.path.isfile(os.path.join(root, 'labels', stem(n) + '.txt')) \
        and not os.path.isfile(os.path.join(root, '_auto', stem(n) + '.json')) \
        and not rv.get(n, {}).get('reviewed')


def check_label_names(src):
    """가져올 폴더(또는 바로 위 폴더)의 data.yaml 클래스 이름이 이 프로그램(0 lie 1 sit 2 stand)과 다르면 거절"""
    for d in (src, os.path.dirname(src)):
        p = os.path.join(d, 'data.yaml')
        if not os.path.isfile(p):
            continue
        try:
            import yaml
            with open(p, encoding='utf-8') as f:
                names = (yaml.safe_load(f) or {}).get('names')
        except Exception:
            return
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names is not None and [str(x) for x in names] != A.NAMES:
            raise ValueError(f'가져올 폴더의 data.yaml 클래스 이름이 {names} 이에요. 이 프로그램은 {A.NAMES} 순서만 받아요 (번호가 다른 자세를 가리키게 돼요)')
        return


def api_works():
    os.makedirs(WORK_DIR, exist_ok=True)
    out = []
    for d in sorted(os.listdir(WORK_DIR), key=natural_key):
        root = os.path.join(WORK_DIR, d)
        if not os.path.isdir(root):
            continue
        imgs = images_of(root)
        rv = load_review(root)
        an = {n for n in imgs if os.path.isfile(os.path.join(root, '_auto', stem(n) + '.json'))}
        lab = {n for n in imgs if os.path.isfile(os.path.join(root, 'labels', stem(n) + '.txt'))}
        rvd = {n for n in imgs if rv.get(n, {}).get('reviewed')}
        out.append({'name': d, 'images': len(imgs), 'analyzed': len(an), 'reviewed': len(rvd),
                    'pending': sum(1 for n in imgs if n not in an and n not in rvd and n not in lab),  # 'new' 분석 대상 (api_analyze 와 같은 기준)
                    'labeled': len(lab), 'reviewed_labeled': len(lab & rvd),
                    'removed': len(images_of(root, '_removed/images')),
                    'mtime': os.path.getmtime(root)})
    return {'works': out, 'work_dir': WORK_DIR, 'rule': A.RULE, 'names': A.NAMES}


RESERVED = re.compile(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$', re.I)
NAME_MAX = 60


def check_new_name(name):
    """새 프로젝트 이름: Windows 가 바꾸거나 못 만드는 이름은 거절 (화면 index.html 의 검사와 같게)"""
    name = (name or '').strip()
    work_root(name)
    if name.endswith('.'):
        raise ValueError('프로젝트 이름은 점(.)으로 끝날 수 없어요')
    if RESERVED.match(name):
        raise ValueError('CON, PRN, AUX, NUL, COM1~9, LPT1~9 는 Windows 가 쓰는 이름이라 프로젝트 이름으로 쓸 수 없어요')
    if len(name) > NAME_MAX:
        raise ValueError(f'프로젝트 이름은 {NAME_MAX}자까지 쓸 수 있어요')
    return name


def project_created(root):
    """프로젝트를 만든 시각 'YYYYMMDD_HHMMSS' (_project.json, 없으면 폴더 만든 시각)"""
    try:
        with open(os.path.join(root, '_project.json'), encoding='utf-8') as f:
            return json.load(f)['created']
    except (OSError, ValueError, KeyError):
        return time.strftime('%Y%m%d_%H%M%S', time.localtime(os.path.getctime(root))) if os.path.isdir(root) else ''


def api_new_work(body):
    check_new_name(body['name'])
    root = work_root(body['name'])
    if os.path.exists(root):
        raise ValueError('같은 이름의 프로젝트가 이미 있어요')
    os.makedirs(os.path.join(root, 'images'))
    A.write_yaml(root)
    with open(os.path.join(root, '_project.json'), 'w', encoding='utf-8') as f:  # 지난 내보내기를 이 프로젝트 것만 보이게
        json.dump({'created': time.strftime('%Y%m%d_%H%M%S')}, f)
    return {'ok': True, 'name': body['name'].strip()}


def api_delete_work(body):
    """프로젝트 폴더를 trash/이름_날짜시간 으로 옮김 (바로 지우지 않음, 되살리려면 work/ 로 다시 옮기면 됨)"""
    name = body['name'].strip()
    root = work_root(name)
    if not os.path.isdir(root):
        raise ValueError('그런 프로젝트가 없어요: ' + name)
    if JOB['busy'] and JOB['work'] == name:
        raise ValueError('이 프로젝트에서 ' + JOB['kind'] + ' 이(가) 진행 중이에요. 끝난 뒤 삭제하세요')
    dst = os.path.join(TRASH_DIR, name + '_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(TRASH_DIR, exist_ok=True)
    shutil.move(root, dst)
    return {'ok': True, 'trash': dst}


def api_state(q):
    root = work_root(q['work'][0])
    rv = load_review(root)
    imgs = [image_state(root, n, rv) for n in images_of(root)]
    counts = {}
    for i in imgs:
        for a in i['anns']:
            counts[a['c']] = counts.get(a['c'], 0) + 1
    return {'work': q['work'][0], 'root': root, 'images': imgs, 'counts': counts,
            'removed': images_of(root, '_removed/images')}


def api_auto(q):
    """편집 화면용: 자동 분석의 관절 좌표"""
    root = work_root(q['work'][0])
    auto = load_auto(root, safe_name(q['n'][0]))
    return {'persons': [{'box': p['box'], 'kp': p['kp'], 'cls': p['cls'], 'conf': p['conf'], 'why': p['why']}
                        for p in (auto or {}).get('persons', [])]}


def api_save(body):
    root = work_root(body['work'])
    if analyzing(body['work']):  # 분석이 끝나면 라벨을 덮어쓰므로 사람이 고친 것을 잃지 않게 막음
        raise ValueError('자동 분석이 진행 중이라 지금은 저장할 수 없어요. 고친 내용은 화면에 남아 있으니 끝난 뒤에 다시 저장해 주세요')
    n = safe_name(body['n'])
    w, h = float(body['w']), float(body['h'])
    lines = []
    for a in body['anns']:
        x1, y1, x2, y2 = [float(v) for v in a['box']]
        x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
        y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
        c = int(a['c'])
        if x2 - x1 < 1 or y2 - y1 < 1 or c not in (0, 1, 2):
            continue
        lines.append(A.yolo_line(c, [x1, y1, x2, y2], w, h))
    backup_label(root, n)
    os.makedirs(os.path.join(root, 'labels'), exist_ok=True)
    with open(os.path.join(root, 'labels', stem(n) + '.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(''.join(l + '\n' for l in lines))
    if 'reviewed' in body:
        rv = load_review(root)
        rv[n] = {'reviewed': bool(body['reviewed']), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        save_review(root, rv)
    return {'ok': True, 'boxes': len(lines), 'state': image_state(root, n, load_review(root))}


def api_mark(body):
    """여러 장을 한 번에 검수 완료/취소 (박스는 그대로)"""
    root = work_root(body['work'])
    if analyzing(body['work']):
        raise ValueError('자동 분석이 진행 중이라 지금은 검수 완료 표시를 바꿀 수 없어요. 끝난 뒤에 다시 해 주세요')
    rv = load_review(root)
    names = [n for n in map(safe_name, body['names']) if os.path.isfile(os.path.join(root, 'images', n))]
    for n in names:
        rv[n] = {'reviewed': bool(body['reviewed']), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    save_review(root, rv)
    return {'ok': True, 'n': len(names)}


def api_move(body):
    """to='removed': 사진+라벨+분석결과를 _removed/ 로, to='restore': 제자리로.
    이름마다 먼저 검사해서, 원본 사진이 없거나 목적지에 같은 이름 파일이 하나라도 있으면 그 사진은 통째로 건너뜀"""
    root = work_root(body['work'])
    to_removed = body['to'] == 'removed'
    if to_removed and analyzing(body['work']):
        raise ValueError('자동 분석이 진행 중이라 지금은 사진을 뺄 수 없어요. 끝난 뒤에 다시 해 주세요')
    moved, skipped = 0, []
    for n in body['names']:
        n = safe_name(n)
        pairs = []
        for sub, f in (('images', n), ('labels', stem(n) + '.txt'), ('_auto', stem(n) + '.json')):
            a, b = os.path.join(root, sub, f), os.path.join(root, '_removed', sub, f)
            pairs.append((a, b) if to_removed else (b, a))
        if not os.path.isfile(pairs[0][0]) or any(os.path.exists(d) for _, d in pairs):
            skipped.append(n)
            continue
        for src, dst in pairs:
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
        moved += 1
    return {'ok': True, 'moved': moved, 'skipped': skipped}


def api_upload(q, data):
    root = work_root(q['work'][0])
    if not os.path.isdir(root):  # 올리는 도중 프로젝트가 삭제됐으면 폴더를 새로 만들지 않음
        raise ValueError('그런 프로젝트가 없어요: ' + q['work'][0])
    rel = q.get('name', [''])[0].replace('\\', '/').strip('/')
    if '..' in rel.split('/'):
        raise ValueError('잘못된 경로')
    if rel.lower().endswith('.txt'):
        return upload_label(root, q['work'][0], rel, data)
    name = rel.replace('/', '__')  # 하위 폴더는 파일 이름 앞에 붙여서 겹치지 않게 (autolabel.import_images 와 같음)
    if os.path.splitext(name)[1].lower() not in A.IMG_EXT:
        return {'ok': False, 'skip': 'type'}
    dst = os.path.join(root, 'images', safe_name(name))
    if os.path.exists(dst):
        return {'ok': False, 'skip': 'exists'}
    if os.path.exists(os.path.join(root, '_removed', 'images', safe_name(name))):
        return {'ok': False, 'skip': 'removed'}  # 제외한 사진을 다시 넣지 않음
    if stem(name).lower() in stems_of(root):
        return {'ok': False, 'skip': 'stem'}  # 확장자만 다른 같은 이름 (라벨 파일이 겹침)
    if not image_ok(data):
        return {'ok': False, 'skip': 'broken'}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(data)
    A.write_yaml(root)
    return {'ok': True}


def upload_label(root, work, rel, data):
    """끌어다 놓은 라벨(.txt)을 이미 올라온 짝 사진의 labels/ 에 넣음 (화면이 사진을 먼저 올림)"""
    if analyzing(work):
        return {'ok': False, 'skip': 'busy'}
    by = {stem(n).lower(): n for n in images_of(root)}
    n = next((by[s.lower()] for s in image_stems_for_label(rel) if s.lower() in by), None)
    if not n:
        return {'ok': False, 'skip': 'nomatch'}  # classes.txt 처럼 짝 사진이 없는 파일
    if not can_take_label(root, n, load_review(root)):
        return {'ok': False, 'skip': 'haslabel'}
    os.makedirs(os.path.join(root, 'labels'), exist_ok=True)
    with open(os.path.join(root, 'labels', stem(n) + '.txt'), 'wb') as f:
        f.write(data)
    return {'ok': True, 'label': True}


def api_pick():
    """내 컴퓨터의 폴더 고르기 창 (서버를 실행한 컴퓨터에서만)"""
    import tkinter as tk
    from tkinter import filedialog
    r = tk.Tk()
    r.withdraw()
    r.attributes('-topmost', True)
    p = filedialog.askdirectory(title='사진 폴더 고르기')
    r.destroy()
    return {'path': os.path.normpath(p) if p else ''}


def api_open(body):
    p = os.path.realpath(body['path'])
    if not (p.startswith(os.path.realpath(WORK_DIR)) or p.startswith(os.path.realpath(EXPORT_DIR))):
        raise ValueError('열 수 없는 폴더')
    os.startfile(p)  # Windows 탐색기
    return {'ok': True}


# ---------------------------------------------------------------- 오래 걸리는 일 (한 번에 하나)
def start_job(kind, work, fn):
    with LOCK:
        if JOB['busy']:
            raise ValueError('이미 다른 작업이 진행 중이에요: ' + JOB['kind'])
        JOB.update(busy=True, kind=kind, work=work, done=0, total=0, msg='시작하는 중…', err='', result=None, seq=JOB['seq'] + 1)

    def body():
        try:
            JOB['result'] = fn()
            JOB['msg'] = '완료'
        except Exception as e:
            traceback.print_exc()
            JOB['err'] = str(e)
        finally:
            JOB['busy'] = False
    threading.Thread(target=body, daemon=True).start()
    return {'ok': True, 'seq': JOB['seq']}


def analyzing(work):
    """이 프로젝트에서 자동 분석이 진행 중인지"""
    return JOB['busy'] and JOB['kind'] == '자동 분석' and JOB['work'] == (work or '').strip()


def prog(i, n, name):
    JOB.update(done=i, total=n, msg=name)


def api_import(body):
    root = work_root(body['work'])
    src = os.path.realpath(body['path'].strip().strip('"'))
    if not os.path.isdir(src):
        raise ValueError('폴더가 없어요: ' + src)
    if src.startswith(os.path.realpath(root)):
        raise ValueError('프로젝트 폴더 안의 폴더는 가져올 수 없어요')
    check_label_names(src)

    def fn():
        JOB['msg'] = '사진 목록 읽는 중…'
        files = A.list_images(src)
        dst = os.path.join(root, 'images')
        rem = os.path.join(root, '_removed', 'images')
        os.makedirs(dst, exist_ok=True)
        c = s = sr = ss = sb = lb = 0
        stems = stems_of(root)
        rv = load_review(root)

        def take_label(rel, name):
            """원본 폴더에 짝 라벨이 있고 프로젝트에 아직 라벨이 없으면 복사 → 자동 분석 없이 바로 검수"""
            p = next((q for q in (os.path.join(src, x) for x in label_candidates(rel)) if os.path.isfile(q)), None)
            if not p or not can_take_label(root, name, rv):
                return 0
            os.makedirs(os.path.join(root, 'labels'), exist_ok=True)
            shutil.copy2(p, os.path.join(root, 'labels', stem(name) + '.txt'))
            return 1

        for i, p in enumerate(files):
            rel = os.path.relpath(p, src).replace('\\', '/')
            name = rel.replace('/', '__')
            d = os.path.join(dst, name)
            if os.path.exists(d):
                s += 1
                lb += take_label(rel, name)  # 사진만 먼저 가져왔던 폴더를 다시 가져오면 라벨만 채움
            elif os.path.exists(os.path.join(rem, name)):
                sr += 1  # 제외한 사진은 다시 넣지 않음
            elif stem(name).lower() in stems:
                ss += 1  # 확장자만 다른 같은 이름 (라벨 파일이 겹침)
            elif not image_ok(path=p):
                sb += 1  # 열 수 없는 사진
            else:
                shutil.copy2(p, d)
                stems.add(stem(name).lower())
                c += 1
                lb += take_label(rel, name)
            prog(i + 1, len(files), name)
        A.write_yaml(root)
        return {'copied': c, 'skipped': s, 'skipped_removed': sr, 'skipped_stem': ss, 'skipped_broken': sb, 'labels': lb}
    return start_job('사진 가져오기', body['work'], fn)


def api_analyze(body):
    root = work_root(body['work'])
    conf, imgsz = float(body.get('conf', 0.35)), int(body.get('imgsz', 960))
    mode = body.get('mode', 'new')  # new: 분석 안 한 사진만 / redo: 검수 안 한 사진 다시 / one: 지정 사진
    if mode == 'one' and not any(os.path.isfile(os.path.join(root, 'images', safe_name(n))) for n in body.get('names', [])):
        raise ValueError('그 사진이 프로젝트에 없어요')

    def fn():
        key = (conf, imgsz)
        if ANALYZER['key'] != key:
            JOB['msg'] = '모델 불러오는 중… (처음에는 시간이 걸려요)'
            ANALYZER['obj'] = A.Analyzer(conf=conf, imgsz=imgsz)
            ANALYZER['key'] = key
        rv = load_review(root)
        imgs = images_of(root)
        if mode == 'one':  # 프로젝트에 있는 사진만
            names = [n for n in map(safe_name, body['names']) if n in imgs]
        elif mode == 'redo':
            names = [n for n in imgs if not rv.get(n, {}).get('reviewed')]
        else:  # 새로 넣은 사진만: 라벨 파일이 이미 있거나 검수 완료한 사진은 건너뜀 (사람이 직접 만든 라벨을 지키려고)
            names = [n for n in imgs if not os.path.isfile(os.path.join(root, '_auto', stem(n) + '.json'))
                     and not os.path.isfile(os.path.join(root, 'labels', stem(n) + '.txt'))
                     and not rv.get(n, {}).get('reviewed')]
        if mode != 'new':
            for n in names:
                backup_label(root, n)
        t = time.time()
        A.run(root, ANALYZER['obj'], names=names, redo=True, progress=prog)
        # 분석 결과(_auto/*.json)가 이번에 새로 써지지 않은 사진 = 분석 못 한 사진 (열 수 없는 사진 등)
        auto_t = lambda n: os.path.getmtime(os.path.join(root, '_auto', stem(n) + '.json')) \
            if os.path.isfile(os.path.join(root, '_auto', stem(n) + '.json')) else 0
        ok = [n for n in names if auto_t(n) >= t - 1]
        if mode == 'one':  # 박스가 자동 결과로 바뀌었으니 검수 완료 표시를 풂
            rv = load_review(root)
            for n in ok:
                rv[n] = {'reviewed': False, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            save_review(root, rv)
        return {'analyzed': len(ok), 'failed': len(names) - len(ok), 'sec': round(time.time() - t, 1), 'mode': mode}
    return start_job('자동 분석', body['work'], fn)


def api_export(body):
    root = work_root(body['work'])
    only = body.get('only_reviewed', True)
    ratio = [max(0.0, float(x)) for x in body.get('split', [100, 0, 0])]
    seed = int(body.get('seed', 0))

    def fn():
        rv = load_review(root)
        imgs = [n for n in images_of(root) if os.path.isfile(os.path.join(root, 'labels', stem(n) + '.txt'))
                and (not only or rv.get(n, {}).get('reviewed'))]
        if not imgs:
            raise ValueError('내보낼 사진이 없어요. 검수 완료한 사진이 있는지, 또는 \'검수 완료한 사진만\' 체크를 확인해 주세요')
        random.Random(seed).shuffle(imgs)
        tot = sum(ratio) or 1
        n_tr = round(len(imgs) * ratio[0] / tot)
        n_va = round(len(imgs) * ratio[1] / tot)
        parts = {'train': imgs[:n_tr], 'valid': imgs[n_tr:n_tr + n_va], 'test': imgs[n_tr + n_va:]}
        out = os.path.join(EXPORT_DIR, f"{body['work']}_{time.strftime('%Y%m%d_%H%M%S')}")
        counts, k = [0, 0, 0], 0
        for sp, names in parts.items():
            for n in names:
                for sub, f in (('images', n), ('labels', stem(n) + '.txt')):
                    d = os.path.join(out, sp, sub)
                    os.makedirs(d, exist_ok=True)
                    shutil.copy2(os.path.join(root, sub, f), os.path.join(d, f))
                for line in open(os.path.join(root, 'labels', stem(n) + '.txt'), encoding='utf-8'):
                    if line.split():
                        counts[int(line.split()[0])] += 1
                k += 1
                prog(k, len(imgs), n)
        used = [sp for sp in parts if parts[sp]]
        val = 'valid' if parts['valid'] else used[0]
        with open(os.path.join(out, 'data.yaml'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"path: .\ntrain: train/images\nval: {val}/images\n")
            if parts['test']:
                f.write('test: test/images\n')
            f.write('\nnc: 3\nnames: ' + json.dumps(A.NAMES) + '\n')
        summary = {'work': body['work'], 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'only_reviewed': only,
                   'images': {sp: len(v) for sp, v in parts.items()},
                   'boxes': dict(zip(A.NAMES, counts)), 'rule': A.RULE,
                   'note': '자동 라벨(규칙) 결과를 사람이 검수한 데이터. 사진 단위 무작위 분할이라 같은 영상의 연속 프레임이 train/valid 에 섞일 수 있음.'}
        with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        JOB['msg'] = 'zip 만드는 중…'
        shutil.make_archive(out, 'zip', out)
        return dict(summary, path=out, zip=out + '.zip')
    return start_job('내보내기', body['work'], fn)


def api_exports(q):
    """이 프로젝트의 지난 내보내기 (exports/프로젝트이름_날짜_시간), 최신순"""
    work = q['work'][0].strip()
    since = project_created(work_root(work))  # 같은 이름으로 예전에 있던(삭제한) 프로젝트의 내보내기는 빼고
    pat = re.compile(re.escape(work) + r'_(\d{8}_\d{6})$')
    out = []
    if os.path.isdir(EXPORT_DIR):
        for d in os.listdir(EXPORT_DIR):
            m = pat.match(d)
            p = os.path.join(EXPORT_DIR, d)
            if not m or not os.path.isdir(p) or m.group(1) < since:
                continue
            try:
                with open(os.path.join(p, 'summary.json'), encoding='utf-8') as f:
                    sm = json.load(f)
            except (OSError, ValueError):
                continue
            out.append({'stamp': m.group(1), 'time': sm.get('time', ''), 'images': sm.get('images', {}),
                        'only_reviewed': sm.get('only_reviewed'), 'path': p})
    out.sort(key=lambda x: x['stamp'], reverse=True)
    return {'exports': out}


# ---------------------------------------------------------------- 사진
def thumb(path, w):
    key = (path, w, os.path.getmtime(path))
    if key not in THUMB_CACHE:
        im = Image.open(path)
        im.draft('RGB', (w, w))
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((w, w))
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=88)
        if len(THUMB_CACHE) > 3000:
            THUMB_CACHE.clear()
        THUMB_CACHE[key] = buf.getvalue()
    return THUMB_CACHE[key]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        if getattr(self, 'cookie', None):
            self.send_header('Set-Cookie', self.cookie)
        self.end_headers()
        self.wfile.write(body)

    def local(self):
        return self.client_address[0] in ('127.0.0.1', '::1')

    def allowed(self, q):
        """내 컴퓨터는 통과, 다른 컴퓨터는 접속 키(주소의 ?key= 또는 쿠키)가 맞아야 함"""
        if self.local():
            return True
        if not KEY:
            return False
        if q.get('key', [''])[0] == KEY:
            return True
        return f'ak={KEY}' in (self.headers.get('Cookie') or '').replace(' ', '').split(';')

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self.allowed(q):
            return self.send(403, '접속 키가 필요해요. key= 가 붙은 전체 주소를 받아 주세요.', 'text/plain; charset=utf-8')
        try:
            if u.path == '/':
                if KEY and q.get('key', [''])[0] == KEY:
                    self.cookie = f'ak={KEY}; Path=/; SameSite=Strict'
                return self.send(200, open(PAGE_FILE, encoding='utf-8').read(), 'text/html; charset=utf-8')
            if u.path == '/api/works':
                return self.send(200, api_works())
            if u.path == '/api/state':
                return self.send(200, api_state(q))
            if u.path == '/api/auto':
                return self.send(200, api_auto(q))
            if u.path == '/api/exports':
                return self.send(200, api_exports(q))
            if u.path == '/api/job':
                return self.send(200, {k: v for k, v in JOB.items()})
            if u.path == '/img':
                root = work_root(q['work'][0])
                sub = '_removed/images' if q.get('removed', ['0'])[0] == '1' else 'images'
                path = os.path.join(root, sub, safe_name(q['n'][0]))
                return self.send(200, thumb(path, int(q.get('w', ['400'])[0])), 'image/jpeg')
            self.send(404, {'error': 'not found'})
        except Exception as e:
            self.send(500, {'error': str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self.allowed(q):
            return self.send(403, {'error': '접속 키가 필요해요'})
        try:
            raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            if u.path == '/api/upload':
                with LOCK:
                    return self.send(200, api_upload(q, raw))
            body = json.loads(raw or b'{}')
            if u.path in ('/api/pick', '/api/open') and not self.local():
                raise ValueError('이 기능은 서버를 실행한 컴퓨터에서만 쓸 수 있어요')
            routes = {'/api/new': api_new_work, '/api/delete_work': api_delete_work, '/api/save': api_save, '/api/mark': api_mark, '/api/move': api_move,
                      '/api/import': api_import, '/api/analyze': api_analyze, '/api/export': api_export,
                      '/api/open': api_open, '/api/pick': lambda b: api_pick()}
            if u.path in routes:
                with LOCK if u.path not in ('/api/import', '/api/analyze', '/api/export', '/api/pick') else NOLOCK:
                    return self.send(200, routes[u.path](body))
            self.send(404, {'error': 'not found'})
        except Exception as e:
            self.send(500, {'error': str(e)})


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


NOLOCK = _NoLock()


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def main():
    global KEY
    ap = argparse.ArgumentParser(description='자동 라벨 + 검수 화면 (0=lie 1=sit 2=stand)')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--no-browser', action='store_true', help='브라우저 자동으로 안 열기')
    ap.add_argument('--share', action='store_true', help='같은 네트워크의 다른 컴퓨터도 접속 허용 (접속 키 필요)')
    ap.add_argument('--key', default=None, help='--share 접속 키 직접 정하기')
    a = ap.parse_args()
    os.makedirs(WORK_DIR, exist_ok=True)
    host = '0.0.0.0' if a.share else '127.0.0.1'
    srv = ThreadingHTTPServer((host, a.port), H)
    url = f'http://127.0.0.1:{a.port}/'
    print(f'작업 폴더: {WORK_DIR}\n자동 라벨 화면: {url}  (끝내려면 Ctrl+C)')
    if a.share:
        KEY = a.key or secrets.token_urlsafe(6)
        print(f'\n다른 사람 접속 주소:\n  http://{lan_ip()}:{a.port}/?key={KEY}')
        print('  ※ Windows 방화벽 창이 뜨면 "개인 네트워크" 허용. 주소를 아는 사람은 라벨을 고칠 수 있어요.')
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('종료')


PAGE_FILE = os.path.join(A.HERE, 'web', 'index.html')

if __name__ == '__main__':
    sys.exit(main())
