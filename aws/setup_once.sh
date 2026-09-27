#!/usr/bin/env bash
# One-time setup: S3 bucket, IAM role for the instances, dataset upload, quota check.
#   aws/setup_once.sh            (run from Git Bash, after `aws configure`)
source "$(dirname "$0")/common.sh"

log "account $(aws sts get-caller-identity --query Account --output text), region $AWS_REGION, bucket $BUCKET"

# ---- S3 bucket (private by default)
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  log "bucket exists"
else
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  log "created bucket"
fi

# ---- IAM role: instances may read/write this bucket and be reached through SSM (no SSH)
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
  }' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  log "created role $ROLE"
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name ber-bucket-access --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\", \"s3:ListBucket\", \"s3:DeleteObject\"],
                   \"Resource\": [\"arn:aws:s3:::$BUCKET\", \"arn:aws:s3:::$BUCKET/*\"]}]
}"
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE"
  log "created instance profile $PROFILE"
fi

# ---- dataset (compressed once, ~2.4 GB of TSVs)
if aws s3api head-object --bucket "$BUCKET" --key data/dataset.tar.gz >/dev/null 2>&1; then
  log "dataset already uploaded"
else
  log "compressing dataset ..."
  tar czf dataset.tar.gz -C student_resource dataset
  log "uploading $(du -h dataset.tar.gz | cut -f1) ..."
  aws s3 cp dataset.tar.gz "s3://$BUCKET/data/dataset.tar.gz"
  rm -f dataset.tar.gz
fi

# ---- EC2 vCPU quotas (new accounts are often limited; request an increase if too low)
od=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query Quota.Value --output text 2>/dev/null || echo "?")
sp=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-34B43A08 --query Quota.Value --output text 2>/dev/null || echo "?")
log "vCPU quota: on-demand standard = $od, spot standard = $sp  (c7i.16xlarge needs 64, c7i.8xlarge 32)"
log "setup done"
