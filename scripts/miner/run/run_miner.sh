#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
NETUID="${NETUID:-126}"
NETWORK="${NETWORK:-finney}"
COLDKEY="${COLDKEY:?set COLDKEY in .env}"
HOTKEY="${HOTKEY:?set HOTKEY in .env}"
<<<<<<< HEAD
AXON_PORT="${AXON_PORT:-8101}"
PM2_NAME="${PM2_NAME:-poker_pdx_gbr}"
=======
AXON_PORT="${AXON_PORT:-8104}"
PM2_NAME="${PM2_NAME:-poker_pdx_nt}"
>>>>>>> 19a2a21 (first)
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"
export P44_CAPTURE="${P44_CAPTURE:-1}"
export P44_CAPTURE_DIR="$(pwd)/${P44_CAPTURE_DIR:-live-data}"
mkdir -p "$P44_CAPTURE_DIR"
<<<<<<< HEAD
export P44_MODEL_PATH="${P44_MODEL_PATH:-detection_model/artifacts/pdx-gbr.joblib}"
=======
export P44_MODEL_PATH="${P44_MODEL_PATH:-detection_model/artifacts/pdx-nt.joblib}"
>>>>>>> 19a2a21 (first)
export P44_REQUIRE_MODEL="${P44_REQUIRE_MODEL:-1}"
export POKER44_MODEL_ARTIFACT_SHA256="${POKER44_MODEL_ARTIFACT_SHA256:-}"
export P44_MANIFEST_REPO_URL="${P44_MANIFEST_REPO_URL:-}"
export P44_MANIFEST_REPO_COMMIT="${P44_MANIFEST_REPO_COMMIT:-}"
export BT_NO_PARSE_CLI_ARGS="${BT_NO_PARSE_CLI_ARGS:-false}"
export PYTHONPATH="$(pwd)"
if ! command -v pm2 >/dev/null 2>&1; then echo "pm2 not installed"; exit 1; fi
pm2 delete "$PM2_NAME" 2>/dev/null || true
ARGS=(--netuid "$NETUID" --wallet.name "$COLDKEY" --wallet.hotkey "$HOTKEY"
      --subtensor.network "$NETWORK" --axon.port "$AXON_PORT" --logging.debug)
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VH <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  ARGS+=(--blacklist.allowed_validator_hotkeys "${VH[@]}")
else
  ARGS+=(--blacklist.force_validator_permit)
fi
pm2 start neurons/miner.py --name "$PM2_NAME" --interpreter python3 -- "${ARGS[@]}"
pm2 save
echo "started $PM2_NAME on port $AXON_PORT"
