#!/usr/bin/env bash
# Show a run's instance state, status marker and the tail of its log.
#   aws/status.sh RUN_NAME [LINES]
[ $# -ge 1 ] || { echo "usage: aws/status.sh RUN_NAME [LINES]"; exit 1; }
RUN_NAME="$1"; LINES="${2:-25}"
source "$(dirname "$0")/common.sh"
aws ec2 describe-instances --filters "Name=tag:ber-run,Values=$RUN_NAME" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' --output text
echo "status: $(aws s3 cp "s3://$BUCKET/runs/$RUN_NAME/STATUS" - 2>/dev/null || echo 'booting (no status yet)')"
aws s3 cp "s3://$BUCKET/runs/$RUN_NAME/run.log" - 2>/dev/null | grep -v DeprecationWarning | tail -n "$LINES" || true
