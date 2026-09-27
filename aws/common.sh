# Shared helpers for the aws/*.sh scripts (run from anywhere; uses the repo root).
set -euo pipefail
# Git Bash on Windows does not always have the AWS CLI v2 install folder on PATH
for d in "/c/Program Files/Amazon/AWSCLIV2" "$(cygpath -u "${LOCALAPPDATA:-C:/}" 2>/dev/null)/Programs/Amazon/AWSCLIV2"; do
  if ! command -v aws >/dev/null 2>&1 && [ -x "$d/aws" ]; then export PATH="$PATH:$d"; fi
done
[ "${RENDER_ONLY:-0}" = "1" ] || command -v aws >/dev/null 2>&1 || { echo "AWS CLI v2 not found — install it first (see aws/README.md)"; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source aws/config.env
export AWS_DEFAULT_REGION="$AWS_REGION"
if [ -z "${BUCKET:-}" ]; then
  ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
  BUCKET="ber-${ACCOUNT}-${AWS_REGION}"
fi
ROLE=ber-ec2-role
PROFILE=ber-ec2-profile
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
