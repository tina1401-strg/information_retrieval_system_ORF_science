#!/bin/bash

is_phone_number() {
    [[ "$1" =~ ^\+[0-9]{7,15}$ ]]
}

cleanup() {
    echo ""
    echo "Cleaning up local Signal data..."
    [ -n "$SIGNAL_TMPDIR" ] && rm -rf "$SIGNAL_TMPDIR"
    rm -rf ~/.local/share/signal-cli
    rm -rf ~/.config/signal-cli
    rm -rf /tmp/signal-cli-* 2>/dev/null || true
    echo "All Signal data deleted."
}

# ─── main ────────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
    # normal mode — no bot
    uv run ./main.py

elif [ "$1" = "--bot" ]; then
    SIGNAL_CLI="./signal-cli-0.14.5/bin/signal-cli"

    # verify signal-cli exists
    if [ ! -f "$SIGNAL_CLI" ]; then
        echo "Error: signal-cli not found at $SIGNAL_CLI"
        exit 1
    fi

    # create RAM-only temp directory
    SIGNAL_TMPDIR=$(mktemp -d -p /dev/shm)
    echo "Using RAM-only config dir: $SIGNAL_TMPDIR"

    # cleanup runs on any exit
    trap cleanup EXIT INT TERM

    # link device via QR code
    echo ""
    echo "Please scan the QR Code via your Signal App:"
    if ! "$SIGNAL_CLI" --config "$SIGNAL_TMPDIR" link -n "MyBot"; then
        echo "Error: failed to link Signal device"
        exit 1
    fi

    # start daemon
    echo ""
    echo "Starting Signal daemon..."
    "$SIGNAL_CLI" --config "$SIGNAL_TMPDIR" daemon --tcp 127.0.0.1:7583 &
    DAEMON_PID=$!
    echo "Signal daemon running (PID $DAEMON_PID)"
    sleep 2

    # run python app
    echo ""
    echo "Starting Python app..."
    uv run ./main.py --bot

    # python finished — prompt to unlink
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  IMPORTANT: Please unlink the bot from Signal:"
    echo "  1. Open Signal on your phone"
    echo "  2. Settings → Linked Devices"
    echo "  3. Remove 'MyBot' from the list"
    echo "════════════════════════════════════════════════════"
    echo ""
    read -r -p "Press ENTER after unlinking in Signal app..." _

    # stop daemon
    echo ""
    echo "Stopping Signal daemon..."
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true

    # cleanup runs automatically via trap

else
    echo "Unknown argument: '$1'"
    echo "Usage:"
    echo "  ./run.sh          — run normally"
    echo "  ./run.sh --bot    — run with Signal bot"
    exit 1
fi