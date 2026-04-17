#!/bin/bash
set -e  # exit on any error

is_phone_number() {
    # must start with + and contain only digits after
    [[ "$1" =~ ^\+[0-9]{7,15}$ ]]
}

cleanup() {
    echo ""
    echo "Cleaning up Signal data..."

    # unregister and delete from Signal servers
    if [ -n "$PHONE" ]; then
        signal-cli --config "$SIGNAL_TMPDIR" -a "$PHONE" unregister --delete-account 2>/dev/null || true
        signal-cli --config "$SIGNAL_TMPDIR" -a "$PHONE" deleteLocalAccountData 2>/dev/null || true
    fi

    # wipe all possible local storage locations
    [ -n "$SIGNAL_TMPDIR" ] && rm -rf "$SIGNAL_TMPDIR"
    rm -rf ~/.local/share/signal-cli
    rm -rf ~/.config/signal-cli
    rm -rf /tmp/signal-cli-* 2>/dev/null || true

    echo "All Signal data deleted."
}

# ─── main ───────────────────────────────────────────────────────────────────

if [ -z "$1" ]; then
    # normal mode — no bot
    cd ./src
    python ./main.py

elif [ "$1" = "--bot" ]; then

    # validate phone number argument
    if [ -z "$2" ]; then
        echo "Error: phone number required. Usage: ./run.sh --bot +XXXXXXXXXXX"
        exit 1
    fi

    if ! is_phone_number "$2"; then
        echo "Error: '$2' is not a valid phone number. Must start with + followed by 7-15 digits."
        exit 1
    fi

    PHONE="$2"
    SIGNAL_CLI="./signal-cli-0.14.0/bin/signal-cli"

    # create RAM-only temp directory — never touches disk
    SIGNAL_TMPDIR=$(mktemp -d -p /dev/shm)
    echo "Using RAM-only config dir: $SIGNAL_TMPDIR"

    # register cleanup on exit, interrupt, or error
    trap cleanup EXIT INT TERM

    # link device via QR code
    echo ""
    echo "Please scan the QR Code via your Signal App:"
    "$SIGNAL_CLI" --config "$SIGNAL_TMPDIR" link -n "MyBot"

    # start signal-cli daemon in background
    echo ""
    echo "Starting Signal daemon for $PHONE..."
    "$SIGNAL_CLI" --config "$SIGNAL_TMPDIR" -a "$PHONE" daemon --tcp 127.0.0.1:7583 &
    DAEMON_PID=$!
    echo "Signal daemon running (PID $DAEMON_PID)"

    # wait briefly for daemon to be ready
    sleep 2

    # run python app
    echo ""
    echo "Starting Python app..."
    cd ./src
    python ./main.py --bot

    # python finished — kill daemon
    echo ""
    echo "Python app finished. Stopping Signal daemon..."
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true

    # cleanup runs automatically via trap

else
    echo "Unknown argument: '$1'"
    echo "Usage:"
    echo "  ./run.sh              — run normally"
    echo "  ./run.sh --bot +43XXX — run with Signal bot"
    exit 1
fi