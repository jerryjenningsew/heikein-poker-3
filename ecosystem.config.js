const fs = require('fs');
const path = require('path');
function loadEnv(p) {
  const env = {};
  if (!fs.existsSync(p)) return env;
  for (const raw of fs.readFileSync(p, 'utf8').split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const i = line.indexOf('=');
    if (i < 0) continue;
    env[line.slice(0, i).trim()] = line.slice(i + 1).trim().replace(/^["']|["']$/g, '');
  }
  return env;
}
const dir = __dirname;
const e = loadEnv(path.join(dir, '.env'));
const allow = (e.ALLOWED_VALIDATOR_HOTKEYS || '').trim();
const args = ['--netuid', e.NETUID || '126',
  '--wallet.name', e.COLDKEY || '', '--wallet.hotkey', e.HOTKEY || '',
  '--subtensor.network', e.NETWORK || 'finney',
<<<<<<< HEAD
  '--axon.port', e.AXON_PORT || '8101', '--logging.debug'];
if (allow) args.push('--blacklist.allowed_validator_hotkeys', ...allow.split(/\s+/));
else args.push('--blacklist.force_validator_permit');
module.exports = { apps: [{
  name: e.PM2_NAME || 'poker_pdx_gbr',
=======
  '--axon.port', e.AXON_PORT || '8104', '--logging.debug'];
if (allow) args.push('--blacklist.allowed_validator_hotkeys', ...allow.split(/\s+/));
else args.push('--blacklist.force_validator_permit');
module.exports = { apps: [{
  name: e.PM2_NAME || 'poker_pdx_nt',
>>>>>>> 19a2a21 (first)
  script: 'neurons/miner.py', interpreter: 'python3', cwd: dir, args,
  env: Object.assign({
    BT_NO_PARSE_CLI_ARGS: 'false',
    PYTHONPATH: dir,
<<<<<<< HEAD
    P44_MODEL_PATH: e.P44_MODEL_PATH || 'detection_model/artifacts/pdx-gbr.joblib',
=======
    P44_MODEL_PATH: e.P44_MODEL_PATH || 'detection_model/artifacts/pdx-nt.joblib',
>>>>>>> 19a2a21 (first)
    P44_REQUIRE_MODEL: e.P44_REQUIRE_MODEL || '1',
    P44_CAPTURE: e.P44_CAPTURE || '1',
    P44_CAPTURE_DIR: path.join(dir, e.P44_CAPTURE_DIR || 'live-data'),
  }, e),
}]};
