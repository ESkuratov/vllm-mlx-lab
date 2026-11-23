#!/usr/bin/env bash
# Запуск всех экспериментов lab/ по очереди.
#
#   source ./vllm-env/bin/activate
#   ./lab/run_all.sh
#
# Лаборатории разложены по темам (01 model engineering/, 02 inference engineering/,
# 03 context engineering/, 04 eval engineering/, 05 observability engineering/),
# поэтому скрипты ищутся глобом по папкам.
# Модель: можно задать через LAB_MODEL.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/lab${PYTHONPATH:+:$PYTHONPATH}"

for script in lab/*/[0-9]*.py; do
    echo
    echo "########################################################################"
    echo "# $script"
    echo "########################################################################"
    python "$script"
done
