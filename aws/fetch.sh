#!/usr/bin/env bash
# Download a finished run: output TSVs, reports, log and the exact code it ran.
#   aws/fetch.sh RUN_NAME        -> output-aws/RUN_NAME/
[ $# -ge 1 ] || { echo "usage: aws/fetch.sh RUN_NAME"; exit 1; }
RUN_NAME="$1"
source "$(dirname "$0")/common.sh"
S3="s3://$BUCKET/runs/$RUN_NAME"
DEST="output-aws/$RUN_NAME"
mkdir -p "$DEST/code"
echo "status: $(aws s3 cp "$S3/STATUS" - 2>/dev/null || echo unknown)"
aws s3 cp "$S3/output" "$DEST" --recursive --only-show-errors
aws s3 cp "$S3/reports" "$DEST" --recursive --only-show-errors || true
aws s3 cp "$S3/run.log" "$DEST/run.log" --only-show-errors || true
aws s3 cp "$S3/code.tar.gz" - | tar xz -C "$DEST/code"
ls -la "$DEST"
report="$DEST/stage2_report.json"; [ -f "$report" ] || report="$DEST/train_report.json"
cat <<EOF

Package it as the next submission with:
  .venv/Scripts/python.exe scripts/make_submission.py --output-dir $DEST --work-dir $DEST \\
      --src $DEST/code/src --report $report
EOF
