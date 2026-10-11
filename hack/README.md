# Kubernetes hack Guidelines

This document describes how to use the scripts from the [`hack`](.) directory
and briefly explains what they do.

## Overview

The [`hack`](.) directory contains scripts that support Kubernetes development,
enhance code robustness, and improve development efficiency.
The explanations and descriptions of these scripts are helpful for contributors.
For details, refer to the following guidelines.

## Key scripts

* [`verify-all.sh`](verify-all.sh): This script is a vestigial redirection. Please do not add "real" logic. Use `make verify` instead.
* [`update-all.sh`](update-all.sh): This script is a vestigial redirection. Please do not add "real" logic.
The `true` target of this makerule is `hack/make-rules/update.sh`. Use `make update` instead.

## Attention
Run all scripts from the Kubernetes root directory.
**Run `hack/verify-all.sh` before submitting a PR. If anything fails, run `hack/update-all.sh`.**

