#!/usr/bin/env sh
set -eu

TEST_DIR="stage6-test"
# VER="old"
VER="new"

# 进入 test 目录执行，保证生成的 .s 与 .c 同目录同名
cd "$TEST_DIR"

# 收集并编译所有 .c -> 同名 .s
found=0
for c in ./*.c; do
  [ -e "$c" ] || continue
  found=1
  base="${c%.c}"
  s="${base}_${VER}.s"

  echo "Compiling: $c -> $s"
  python3 ../main.py --input "$c" --riscv > "$s"
done

if [ "$found" -eq 0 ]; then
  echo "No .c files found under $TEST_DIR"
  exit 1
fi

echo "Done."
