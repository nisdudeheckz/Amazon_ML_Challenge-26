#!/usr/bin/env bash
# Terminate the instance(s) of a run immediately (results already in S3 are kept).
#   aws/stop.sh RUN_NAME          |   aws/stop.sh --all   (every ber-* instance)
[ $# -ge 1 ] || { echo "usage: aws/stop.sh RUN_NAME | --all"; exit 1; }
source "$(dirname "$0")/common.sh"
if [ "$1" = "--all" ]; then filter="Name=tag-key,Values=ber-run"; else filter="Name=tag:ber-run,Values=$1"; fi
ids=$(aws ec2 describe-instances --filters "$filter" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query 'Reservations[].Instances[].InstanceId' --output text)
[ -z "$ids" ] && { log "no running instances"; exit 0; }
# shellcheck disable=SC2086
aws ec2 terminate-instances --instance-ids $ids --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
