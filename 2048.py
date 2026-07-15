import time
import subprocess
import cv2
import numpy as np
import random
import os

# ------------------------------------------------------------------
# ОБМЕЖЕННЯ ТА КОНСТАНТИ АЛГОРИТМУ
# ------------------------------------------------------------------
FORBIDDEN_LOG = 11   # log2(2048) — плитка, яку не можна створювати
DANGER_LOG = 10       # log2(1024)

_LATE_GAME = False   # Вмикається автоматично після 1300 ходів

# ------------------------------------------------------------------
# НАЛАШТУВАННЯ АВТОМАТИЗАЦІЇ ТЕЛЕФОНУ (Підлаштуй під свій екран!)
# ------------------------------------------------------------------
SWIPE_CENTER_X = 540
SWIPE_CENTER_Y = 1150
SWIPE_DISTANCE = 300

# Координати клітинок (4х4) на скріншоті (x1, y1, x2, y2)
CELL_COORDINATES = [
    [(100, 700, 300, 900), (320, 700, 520, 900), (540, 700, 740, 900), (760, 700, 960, 900)],
    [(100, 920, 300, 1120), (320, 920, 520, 1120), (540, 920, 740, 1120), (760, 920, 960, 1120)],
    [(100, 1140, 300, 1340), (320, 1140, 520, 1340), (540, 1140, 740, 1340), (760, 1140, 960, 1340)],
    [(100, 1360, 300, 1560), (320, 1360, 520, 1560), (540, 1360, 740, 1560), (760, 1360, 960, 1560)]
]

TEMPLATE_FILES = {
    "templates/2.png": 1, "templates/4.png": 2, "templates/8.png": 3,
    "templates/16.png": 4, "templates/32.png": 5, "templates/64.png": 6,
    "templates/128.png": 7, "templates/256.png": 8, "templates/512.png": 9,
    "templates/1024.png": 10, "templates/2048.png": 11
}

# ------------------------------------------------------------------
# ВЗАЄМОДІЯ З ADB ТА OPENCV РОЗПІЗНАВАННЯ
# ------------------------------------------------------------------
def get_screen():
    pipe = subprocess.Popen(['adb', 'shell', 'screencap', '-p'], stdout=subprocess.PIPE)
    image_bytes = pipe.communicate()[0]
    image_bytes = image_bytes.replace(b'\r\n', b'\n')
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def send_move(direction):
    x1, y1 = SWIPE_CENTER_X, SWIPE_CENTER_Y
    if direction == "UP":    x2, y2 = x1, y1 - SWIPE_DISTANCE
    elif direction == "DOWN":  x2, y2 = x1, y1 + SWIPE_DISTANCE
    elif direction == "LEFT":  x2, y2 = x1 - SWIPE_DISTANCE, y1
    elif direction == "RIGHT": x2, y2 = x1 + SWIPE_DISTANCE, y1
    
    cmd = f"adb shell input swipe {x1} {y1} {x2} {y2} 100"
    subprocess.run(cmd.split(), stdout=subprocess.DEVNULL)

def scan_board_from_phone(templates):
    screen = get_screen()
    board = [0] * 16
    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            x1, y1, x2, y2 = CELL_COORDINATES[r][c]
            cell_img = screen[y1:y2, x1:x2]
            
            best_val = 0
            best_max_val = 0.85
            
            for t_img, log_val in templates.items():
                res = cv2.matchTemplate(cell_img, t_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_max_val:
                    best_max_val = max_val
                    best_val = log_val
            board[idx] = best_val
    return tuple(board)

# ------------------------------------------------------------------
# ПОБУДОВА LOOK-UP ТАБЛИЦЬ ДЛЯ МИТТЄВИХ ХОДІВ (БІТБОРД ТЕХНІКА)
# ------------------------------------------------------------------
def _merge_row(vals):
    non_zero = [v for v in vals if v != 0]
    new_vals, score, skip = [], 0, False
    for i in range(len(non_zero)):
        if skip:
            skip = False; continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            merged = non_zero[i] + 1
            new_vals.append(merged)
            score += (1 << merged)
            skip = True
        else:
            new_vals.append(non_zero[i])
    new_vals += [0] * (4 - len(new_vals))
    return tuple(new_vals), score

LEFT_TABLE = {}
for a in range(16):
    for b in range(16):
        for c in range(16):
            for d in range(16):
                row = (a, b, c, d)
                new_row, score = _merge_row(row)
                LEFT_TABLE[row] = (new_row, score, new_row != row)

RIGHT_TABLE = {r[::-1]: (nr[::-1], sc, mv) for r, (nr, sc, mv) in LEFT_TABLE.items()}

def _rows(board): return (board[0:4], board[4:8], board[8:12], board[12:16])
def _cols(board):
    return ((board[0], board[4], board[8], board[12]), (board[1], board[5], board[9], board[13]),
            (board[2], board[6], board[10], board[14]), (board[3], board[7], board[11], board[15]))

def apply_move(board, direction):
    moved_any, gained = False, 0
    if direction in ("LEFT", "RIGHT"):
        table = LEFT_TABLE if direction == "LEFT" else RIGHT_TABLE
        new_rows = []
        for r in _rows(board):
            nr, sc, mv = table[r]; new_rows.append(nr); gained += sc; moved_any = moved_any or mv
        return new_rows[0] + new_rows[1] + new_rows[2] + new_rows[3], moved_any, gained
    else:
        table = LEFT_TABLE if direction == "UP" else RIGHT_TABLE
        new_cols = []
        for c in _cols(board):
            nc, sc, mv = table[c]; new_cols.append(nc); gained += sc; moved_any = moved_any or mv
        # ВИПРАВЛЕНО: Return винесено з циклу for!
        return tuple(new_cols[c][r] for r in range(4) for c in range(4)), moved_any, gained

# ------------------------------------------------------------------
# ЕВРИСТИКА ОЦІНКИ ТА ЗМІЇНА МАТРИЦЯ (SNAKE PATTERN)
# ------------------------------------------------------------------
SNAKE_WEIGHTS = (
    4**15, 4**14, 4**13, 4**12,
    4**8,  4**9,  4**10, 4**11,
    4**7,  4**6,  4**5,  4**4,
    4**0,  4**1,  4**2,  4**3,
)
_SNAKE_NORM = float(sum(4**i for i in range(16)))

def _build_symmetry_index_maps():
    idx = list(range(16))
    def rot90(m): return [m[(3 - c) * 4 + r] for r in range(4) for c in range(4)]
    def mirror(m): return [m[r * 4 + (3 - c)] for r in range(4) for c in range(4)]
    maps = []
    cur = idx
    for _ in range(4):
        maps.append(tuple(cur)); maps.append(tuple(mirror(cur))); cur = rot90(cur)
    return maps

SYMMETRY_MAPS = _build_symmetry_index_maps()
_EVAL_CACHE = {}

def _snake_score(board):
    best = -1.0
    for m in SYMMETRY_MAPS:
        total = 0
        for i in range(16):
            lv = board[m[i]]
            if lv: total += SNAKE_WEIGHTS[i] * (1 << lv)
        if total > best: best = total
    return best / _SNAKE_NORM

def evaluate(board):
    cached = _EVAL_CACHE.get(board)
    if cached is not None: return cached

    if FORBIDDEN_LOG in board:
        _EVAL_CACHE[board] = -1e12
        return -1e12

    empty = board.count(0)
    score = empty * 270.0

    mono = 0
    for line in _rows(board) + _cols(board):
        inc = dec = 0
        for i in range(3):
            diff = line[i + 1] - line[i]
            if diff > 0: inc += diff
            else: dec -= diff
        mono -= min(inc, dec)
    score += mono * 40.0

    smooth = 0
    for r in range(4):
        for c in range(4):
            v = board[r * 4 + c]
            if c + 1 < 4: smooth -= abs(v - board[r * 4 + c + 1])
            if r + 1 < 4: smooth -= abs(v - board[(r + 1) * 4 + c])
    score += smooth * 4.0

    mergeable = 0
    for r in range(4):
        for c in range(4):
            v = board[r * 4 + c]
            if v == 0: continue
            if c + 1 < 4 and board[r * 4 + c + 1] == v: mergeable += 1
            if r + 1 < 4 and board[(r + 1) * 4 + c] == v: mergeable += 1
    score += mergeable * 120.0
    score += _snake_score(board) * 0.4

    if DANGER_LOG in board:
        danger_count = board.count(DANGER_LOG)
        if _LATE_GAME and empty <= 6:
            if danger_count >= 2: score += 300.0
            for r in range(4):
                for c in range(4):
                    if board[r * 4 + c] == DANGER_LOG:
                        if c + 1 < 4 and board[r * 4 + c + 1] == DANGER_LOG: score += 500.0
                        if r + 1 < 4 and board[(r + 1) * 4 + c] == DANGER_LOG: score += 500.0

    _EVAL_CACHE[board] = score
    return score

# ------------------------------------------------------------------
# ПОШУК КРАЩОГО ХОДУ (EXPECTIMAX)
# ------------------------------------------------------------------
def expectimax(board, depth, is_player, cache):
    key = (board, depth, is_player)
    cached = cache.get(key)
    if cached is not None: return cached

    if depth == 0 or FORBIDDEN_LOG in board:
        val = evaluate(board)
        cache[key] = val
        return val

    if is_player:
        best = -float('inf')
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            nb, moved, _ = apply_move(board, d)
            if moved:
                v = expectimax(nb, depth - 1, False, cache)
                if v > best: best = v
        if best == -float('inf'): best = evaluate(board)
        cache[key] = best
        return best
    else:
        empty = [i for i, v in enumerate(board) if v == 0]
        if not empty:
            val = evaluate(board)
            cache[key] = val
            return val
        sample = empty if len(empty) <= 7 else random.sample(empty, 7)
        total = 0.0
        for i in sample:
            b2 = board[:i] + (1,) + board[i + 1:]
            total += 0.9 * expectimax(b2, depth - 1, True, cache)
            b4 = board[:i] + (2,) + board[i + 1:]
            total += 0.1 * expectimax(b4, depth - 1, True, cache)
        val = total / len(sample)
        cache[key] = val
        return val

def choose_move(board):
    empty_count = board.count(0)
    danger = DANGER_LOG in board
    
    # Адаптивна глибина залежно від стану поля
    if empty_count <= 2: depth = 6
    elif empty_count <= 3: depth = 5
    elif empty_count <= 7: depth = 3
    else: depth = 2
    if danger: depth += 1
    
    cache = {}
    candidates = []
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
        nb, moved, _ = apply_move(board, d)
        if not moved: continue
        forbidden = FORBIDDEN_LOG in nb
        v = expectimax(nb, depth, False, cache)
        candidates.append((v, d, forbidden))

    if not candidates: return None
    safe = [c for c in candidates if not c[2]]
    pool = safe if safe else candidates
    _, best_d, _ = max(pool, key=lambda x: x[0])
    return best_d

# ------------------------------------------------------------------
# ГОЛОВНИЙ ЦИКЛ УПРАВЛІННЯ ТЕЛЕФОНОМ
# ------------------------------------------------------------------
def play_on_phone():
    global _LATE_GAME
    print("[🚀] Запуск швидкого бітборд-бота на телефоні...")
    
    # Визначаємо шлях до скрипта для побудови абсолютних шляхів
    # Визначаємо шлях до скрипта для побудови абсолютних шляхів
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Попередньо завантажуємо шаблони картинок
    loaded_templates = {}
    for p, v in TEMPLATE_FILES.items():
        # normpath виправить усі похилі риски під стандарт Windows (\ замість /)
        abs_path = os.path.normpath(os.path.join(script_dir, p))
        try:
            # Перевіряємо, чи файл взагалі існує за цим шляхом
            if not os.path.exists(abs_path):
                print(f"[❌] Файл фізично ВІДСУТНІЙ за шляхом: {abs_path}")
                continue
                
            img_array = np.fromfile(abs_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            img = None
            print(f"[💥] Помилка читання файлу: {e}")
            
        if img is None:
            print(f"[⚠️] Не вдалося завантажити шаблон: {abs_path}")
        else:
            loaded_templates[img] = v
            
    moves_count = 0
    _LATE_GAME = False
    _EVAL_CACHE.clear()

    while True:
        # Зчитуємо та конвертуємо поточний скріншот телефону у формат log2
        board = scan_board_from_phone(loaded_templates)
        
        # Перевірка на помилкову появу або випадкове злиття у 2048
        if FORBIDDEN_LOG in board:
            print("[💀] Аварійна зупинка! Виявлено або з'явиться плитка 2048.")
            break
            
        # Увімкнення фінальної стадії накопичення очок
        if not _LATE_GAME and moves_count >= 1300:
            _LATE_GAME = True
            _EVAL_CACHE.clear()
            print("[🔥] Режим пізньої гри (Late Game) активовано!")

        # Розраховуємо найкращий напрямок
        move = choose_move(board)
        if move is None:
            print("[🛑] Немає доступних ходів. Кінець гри.")
            break

        # Виводимо статус та робимо фізичний свайп на Android
        print(f"Хід {moves_count} -> Свайп: {move} | Макс плитка на екрані: {1 << max(board)}")
        send_move(move)
        moves_count += 1
        
        # Захист від переповнення кешу оцінок станів
        if len(_EVAL_CACHE) > 300000:
            _EVAL_CACHE.clear()
            
        # Невелика затримка під анімацію інтерфейсу
        time.sleep(0.15)

if __name__ == "__main__":
    play_on_phone()