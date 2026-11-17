#!/bin/bash
# Запуск на Mac/Linux: двойной клик (Mac) или ./start.command
cd "$(dirname "$0")" || exit 1
if [ ! -f .venv/installed.ok ]; then
  echo "Первый запуск: ставлю Python-библиотеки, это займёт 5-10 минут..."
  python3 -m venv .venv || { echo "Не найден python3. Установи Python 3.12 с python.org"; read -r; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "Не удалось установить библиотеки. Проверь интернет."; read -r; exit 1; }
  touch .venv/installed.ok
fi
.venv/bin/python turnstile_detector.py "$@"
echo
read -r -p "Нажми Enter, чтобы закрыть окно"
