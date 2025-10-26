#!/usr/bin/env bash
# usage: bash run_example.sh AAPL
TICKER=${1:-AAPL}
python -m src.train --ticker $TICKER --epochs 20 --seq_len 60 --batch_size 32 --output_dir outputs/$TICKER
