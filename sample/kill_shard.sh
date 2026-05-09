#!/usr/bin/env bash
# Kill one (or all) streaming-shard workers by reading the pid recorded by
# launch_shards.sh.
#
# Usage:
#   sample/kill_shard.sh [--samples-root DIR] <shard_id|all>
#
# Sends SIGTERM first, waits up to 5s, then SIGKILL.
set -euo pipefail

SAMPLES_ROOT="samples"
TARGET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --samples-root) SAMPLES_ROOT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--samples-root DIR] <shard_id|all>" >&2
            exit 0
            ;;
        *) TARGET="$1"; shift ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    echo "missing target shard id (e.g. '0' or 'all')" >&2
    exit 1
fi

kill_one() {
    local shard="$1"
    local pid_file="$SAMPLES_ROOT/shard_$shard/pid"
    if [[ ! -f "$pid_file" ]]; then
        echo "[kill] shard $shard: no pid file at $pid_file" >&2
        return 0
    fi
    local pid
    pid="$(cat "$pid_file")"
    if [[ -z "$pid" ]]; then
        echo "[kill] shard $shard: empty pid file" >&2
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[kill] shard $shard: pid $pid already gone"
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[kill] shard $shard: pid $pid stopped (TERM)"
            return 0
        fi
        sleep 0.5
    done
    kill -KILL "$pid" 2>/dev/null || true
    echo "[kill] shard $shard: pid $pid killed (KILL)"
}

if [[ "$TARGET" == "all" ]]; then
    for d in "$SAMPLES_ROOT"/shard_*/; do
        s="$(basename "$d")"
        s="${s#shard_}"
        kill_one "$s"
    done
else
    kill_one "$TARGET"
fi
