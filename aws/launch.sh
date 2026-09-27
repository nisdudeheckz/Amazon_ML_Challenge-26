#!/usr/bin/env bash
# Launch one pipeline run on a fresh EC2 instance (it terminates itself when done).
#
#   aws/launch.sh RUN_NAME [STEP ...]            steps default to: all
#   FROM_RUN=full-1 aws/launch.sh s2-try stage2  reuse work/ of an earlier run (KEEP_WORK=1)
#
# Settings come from aws/config.env; any of them can be overridden on the command line,
# e.g.  INSTANCE_TYPE=c7i.8xlarge BER_JOBS=2 aws/launch.sh full-1
#
#   RENDER_ONLY=1 aws/launch.sh RUN [STEP ...]   only build code.tar.gz + userdata.sh into
#       aws/.render/RUN/ (no AWS calls) — for launching through other AWS tooling
[ $# -ge 1 ] || { echo "usage: aws/launch.sh RUN_NAME [STEP ...]"; exit 1; }
RUN_NAME="$1"; shift
STEPS="${*:-all}"
_overrides="$(env | grep -E '^(INSTANCE_TYPE|SPOT|VOLUME_GB|MAX_HOURS|AUTO_SHUTDOWN|KEEP_WORK|BER_[A-Z_]+)=' || true)"
RENDER_ONLY="${RENDER_ONLY:-0}"
source "$(dirname "$0")/common.sh"
while IFS='=' read -r k v; do [ -n "$k" ] && printf -v "$k" '%s' "$v"; done <<< "$_overrides"
FROM_RUN="${FROM_RUN:-}"
S3="s3://$BUCKET/runs/$RUN_NAME"

# ---- package the current code (exactly what the run will execute) + user-data
tmp="$(mktemp -d)"
tar czf "$tmp/code.tar.gz" --exclude=__pycache__ src requirements.txt
{   # user-data = settings + aws/remote_run.sh (LF line endings)
  echo '#!/bin/bash'
  for k in RUN_NAME BUCKET AWS_REGION STEPS FROM_RUN MAX_HOURS AUTO_SHUTDOWN KEEP_WORK            BER_JOBS BER_THREADS BER_MAX_DF BER_TRAIN_FRAC BER_SELF_TRAIN; do
    printf '%s=%q
' "${k/RUN_NAME/RUN}" "${!k}"
  done
  tr -d '' < aws/remote_run.sh
} > "$tmp/userdata.sh"
if [ "$RENDER_ONLY" = "1" ]; then
  out="aws/.render/$RUN_NAME"; mkdir -p "$out"; mv "$tmp/code.tar.gz" "$tmp/userdata.sh" "$out/"; rm -rf "$tmp"
  log "rendered $out/code.tar.gz and $out/userdata.sh (instance $INSTANCE_TYPE, profile $PROFILE, ${VOLUME_GB} GB, spot=$SPOT)"
  exit 0
fi

if aws s3api head-object --bucket "$BUCKET" --key "runs/$RUN_NAME/code.tar.gz" >/dev/null 2>&1; then
  echo "run '$RUN_NAME' already exists in s3://$BUCKET/runs/ — pick another name"; exit 1
fi
aws s3 cp "$tmp/code.tar.gz" "$S3/code.tar.gz" --only-show-errors
log "uploaded code to $S3/code.tar.gz"

# MSYS_NO_PATHCONV: stop Git Bash on Windows from rewriting "/aws/..." and "/dev/xvda" as paths
AMI=$(MSYS_NO_PATHCONV=1 aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
      --query Parameter.Value --output text)
market=()
[ "$SPOT" = "1" ] && market=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')

cd "$tmp"   # file:// paths relative to here work for both Windows and Linux aws CLIs
IID=$(MSYS_NO_PATHCONV=1 aws ec2 run-instances \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --count 1 \
  --iam-instance-profile "Name=$PROFILE" \
  --user-data file://userdata.sh \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
  --metadata-options HttpTokens=required \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ber-$RUN_NAME},{Key=ber-run,Value=$RUN_NAME}]" \
  "${market[@]}" --query 'Instances[0].InstanceId' --output text)
cd "$ROOT"; rm -rf "$tmp"
log "launched $IID ($INSTANCE_TYPE, spot=$SPOT) for run '$RUN_NAME', steps: $STEPS${FROM_RUN:+, from run $FROM_RUN}"
log "watch it with:  aws/status.sh $RUN_NAME      results:  aws/fetch.sh $RUN_NAME"
