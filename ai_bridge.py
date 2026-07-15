"""
Міст між нашим ботом і готовим C++ ШІ nneonneo/2048-ai.

Перед використанням:
1. git clone https://github.com/nneonneo/2048-ai.git
2. Зібрати bin/2048.dll (make-msvc.bat у Native Tools Command Prompt,
   або MinGW-інструкція з README проєкту).
3. Покласти зібраний 2048.dll поруч із цим файлом (або вказати
   AILIB_PATH нижче).

Формат дошки, який очікує DLL (băitboard nneonneo):
    64-бітне число, поле розбите на 16 ніблів по 4 біти.
    Нібл i (i = row*4 + col) містить ПОКАЗНИК степеня двійки:
        0 -> порожня клітинка
        1 -> 2      (2^1)
        2 -> 4      (2^2)
        3 -> 8      (2^3)
        ...
        n -> 2^n
    Рядок 0 (верхній) лежить у молодших 16 бітах, рядок 3 — у
    старших. Це стандартне кодування з оригінального проєкту.
"""

import ctypes
import math
import os

AILIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2048.dll")

_ailib = None
_MOVE_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]  # порядок відповідає ABI DLL


def _load_ailib():
    global _ailib
    if _ailib is not None:
        return _ailib

    if not os.path.isfile(AILIB_PATH):
        raise FileNotFoundError(
            f"Не знайдено {AILIB_PATH}. Зібрати DLL: git clone "
            "https://github.com/nneonneo/2048-ai.git, потім make-msvc.bat "
            "(Native Tools Command Prompt), і скопіювати bin/2048.dll сюди."
        )

    lib = ctypes.CDLL(AILIB_PATH)
    lib.init_tables()  # ОБОВ'ЯЗКОВО: без цього внутрішні таблиці
    # ходів/оцінок бібліотеки лишаються порожніми, і find_best_move
    # завжди вважає, що жоден хід нічого не змінює (саме це і
    # спостерігалось: "eval'd 0 moves ... maxdepth=0" для всіх
    # напрямків одразу).
    lib.find_best_move.argtypes = [ctypes.c_uint64]
    lib.find_best_move.restype = ctypes.c_int
    _ailib = lib
    return _ailib


def to_c_board(board):
    """
    Пакує наш board (список 4x4 значень 0/2/4/8/16/...) у 64-бітний
    bitboard, який очікує DLL.
    """
    packed = 0
    for r in range(4):
        for c in range(4):
            value = board[r][c]
            exponent = 0 if value == 0 else int(round(math.log2(value)))
            nibble_index = r * 4 + c
            packed |= (exponent & 0xF) << (4 * nibble_index)
    return ctypes.c_uint64(packed)


def choose_best_move(board):
    """
    Той самий інтерфейс, що й наш власний choose_best_move(board) з
    2048.py — щоб можна було підмінити один на інший однією зміною
    імпорту. Повертає "UP"/"DOWN"/"LEFT"/"RIGHT" або None, якщо ходів
    більше немає (game over).
    """
    lib = _load_ailib()
    c_board = to_c_board(board)
    move_index = lib.find_best_move(c_board)
    if move_index < 0 or move_index > 3:
        return None
    return _MOVE_NAMES[move_index]


if __name__ == "__main__":
    # Швидка самоперевірка на прикладі з офіційного README проєкту:
    # 16 128 256 1024
    # 16   8   2    0
    #  8   2   0    0
    #  0   4   0    0
    # Очікуваний хід (за README): "up"
    test_board = [
        [16, 128, 256, 1024],
        [16, 8, 2, 0],
        [8, 2, 0, 0],
        [0, 4, 0, 0],
    ]
    print("Тестова дошка:")
    for row in test_board:
        print([f"{v:5}" for v in row])
    print("Рекомендований хід:", choose_best_move(test_board))