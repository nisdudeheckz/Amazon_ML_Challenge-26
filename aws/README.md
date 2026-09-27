# Running the pipeline on AWS (EC2 + S3)

Each run starts a fresh CPU instance. The instance:

- pulls the dataset and **a snapshot of the current `src/`** from your S3 bucket;
- runs the pipeline and mirrors its log to S3 every minute;
- uploads the outputs to S3;
- **terminates itself**.

There is no SSH, no open ports and no long-running cost. The instance also terminates itself after `MAX_HOURS` no matter what, as a safety net.

```
local (Git Bash)                        S3 bucket                         EC2 (per run)
aws/setup_once.sh  ── dataset ──▶  data/dataset.tar.gz  ──────────▶  /opt/ber/dataset
aws/launch.sh RUN  ── code ─────▶  runs/RUN/code.tar.gz ──────────▶  /opt/ber/src ─▶ python -m ber.run ...
aws/status.sh RUN  ◀───────────── runs/RUN/{STATUS,run.log} ◀──────  log mirror (1/min)
aws/fetch.sh RUN   ◀───────────── runs/RUN/{output,reports,work} ◀── on exit, then shutdown
```

## One-time setup (you)

1. **Check the credits.** In the AWS console, open *Billing → Credits* and confirm the $200 is applied and covers **EC2** and **S3**. Promotional credits usually do. If they only cover specific services (for example SageMaker), stop here and tell me.
2. **Set a safety budget.** Create a $150 cost budget with an e-mail alert under *Billing → Budgets*.
3. **Install AWS CLI v2** for Windows: https://awscli.amazonaws.com/AWSCLIV2.msi
4. **Create CLI credentials.**
   1. In *IAM → Users*, create a user such as `ber-cli` with the `AdministratorAccess` policy. Don't use root keys.
   2. Create an access key for it.
   3. Run `aws configure`:
      - enter the key and secret;
      - region `us-east-1`;
      - output `json`.

   Keep the keys to yourself: they only go into `aws configure`.
5. **Check the vCPU quota.** New accounts are often limited to fewer vCPUs than a 64-vCPU instance needs. `aws/setup_once.sh` prints your quota. If it's below 64, request an increase in *Service Quotas → Amazon EC2 → "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances"*; approval can take a few hours. Meanwhile use `INSTANCE_TYPE=c7i.8xlarge` (32 vCPU).

Then, from the repo root in Git Bash:

```bash
aws/setup_once.sh
```

This creates the private bucket `ber-<account>-us-east-1` and the instance role (S3 access and SSM only), uploads the dataset (≈0.6 GB compressed), and prints the quotas.

## Runs

```bash
aws/launch.sh full-1                          # whole pipeline: prepare -> candidates -> enrich/stage2
aws/status.sh full-1                          # instance state + last log lines (re-run any time)
aws/fetch.sh  full-1                          # outputs, reports, log, exact code -> output-aws/full-1/

FROM_RUN=full-1 aws/launch.sh s2-exp stage2   # reuse full-1's work/ and only redo stage 2 (~30 min)
BER_MAX_DF=15000 aws/launch.sh full-2         # any config.env setting can be overridden per run
aws/stop.sh full-1                            # kill a run early (aws/stop.sh --all kills every run)
```

`aws/fetch.sh` prints the `scripts/make_submission.py` command that packages the downloaded run as the next `Submissions/Submission-n` from the exact code it ran.

## Cost guide (us-east-1 on-demand; check the EC2 pricing page)

| instance | vCPU / RAM | ≈ $/h | full pipeline | ≈ $ per full run |
|---|---|---|---|---|
| **r7i.2xlarge** (default; this account's 8-vCPU quota) | 8 / 64 GiB | 0.53 | ~6–8 h (stage-2-only runs ~1–1.5 h) | ~4 (stage-2-only ~$1) |
| c7i.8xlarge | 32 / 64 GiB | 1.43 | ~2 h | ~3 |
| c7i.16xlarge | 64 / 128 GiB | 2.86 | ~1–1.5 h | ~3–4 |

- A stage-2-only rerun (`FROM_RUN=...`) costs about $1–1.5.
- S3 storage (~15 GB per kept run) costs ~$0.35 a month.
- Downloading a run's outputs (~0.4 GB) costs ~$0.04.
- `SPOT=1` is usually 50–70% cheaper, but an interrupted run has to be relaunched.

The laptop-measured runtimes will be replaced with measured cloud numbers after the first run.
