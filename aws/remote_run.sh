# EC2 user-data body (Amazon Linux 2023). launch.sh prepends the run settings:
#   RUN BUCKET AWS_REGION STEPS FROM_RUN MAX_HOURS AUTO_SHUTDOWN KEEP_WORK BER_*
# Everything is logged to /var/log/ber-run.log, mirrored to s3://$BUCKET/runs/$RUN/run.log.
exec > /var/log/ber-run.log 2>&1
set -uo pipefail
export AWS_DEFAULT_REGION="$AWS_REGION" HOME=/root PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export BER_JOBS BER_MAX_DF BER_TRAIN_FRAC BER_SELF_TRAIN
[ -n "${BER_THREADS:-}" ] && export BER_THREADS
S3="s3://$BUCKET/runs/$RUN"
echo "[$(date -u +%FT%TZ)] run $RUN on $(nproc) vCPU / $(free -g | awk '/Mem/{print $2}') GiB, steps: $STEPS"

# safety net: hard stop after MAX_HOURS, and a log mirror every minute
( sleep $((MAX_HOURS * 3600)); echo "MAX_HOURS reached"; aws s3 cp /var/log/ber-run.log "$S3/run.log" --only-show-errors
  echo TIMEOUT | aws s3 cp - "$S3/STATUS"; shutdown -h now ) &
( while true; do sleep 60; aws s3 cp /var/log/ber-run.log "$S3/run.log" --only-show-errors; done ) &

finish() {
  status=$?
  echo "[$(date -u +%FT%TZ)] finished with exit code $status"
  cd /opt/ber || true
  aws s3 cp output "$S3/output" --recursive --only-show-errors || true
  for f in work/*.json; do [ -f "$f" ] && aws s3 cp "$f" "$S3/reports/$(basename "$f")" --only-show-errors; done
  if [ "$KEEP_WORK" = "1" ] && [ -d work ]; then
    echo "uploading work/ ..."; aws s3 sync work "$S3/work" --only-show-errors || true
  fi
  aws s3 cp /var/log/ber-run.log "$S3/run.log" --only-show-errors
  { [ "$status" = "0" ] && echo DONE || echo "FAILED $status"; } | aws s3 cp - "$S3/STATUS"
  [ "$AUTO_SHUTDOWN" = "1" ] && shutdown -h now
}
trap finish EXIT
set -e

echo RUNNING | aws s3 cp - "$S3/STATUS"
dnf install -y -q python3.11 python3.11-pip libgomp tar gzip
mkdir -p /opt/ber && cd /opt/ber
aws s3 cp "s3://$BUCKET/data/dataset.tar.gz" - | tar xz
aws s3 cp "$S3/code.tar.gz" - | tar xz
python3.11 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
if [ -n "$FROM_RUN" ]; then
  echo "restoring work/ from run $FROM_RUN ..."
  aws s3 sync "s3://$BUCKET/runs/$FROM_RUN/work" work --only-show-errors
fi
cd src
for step in $STEPS; do
  echo "[$(date -u +%FT%TZ)] ===== step $step"
  ../.venv/bin/python -m ber.run "$step" --data-dir ../dataset --work-dir ../work --out-dir ../output
done
