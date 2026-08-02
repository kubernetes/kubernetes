# Kubernetes hack GuideLines

This document describes how you can use the scripts from [`hack`](.) directory
and gives a brief introduction and explanation of these scripts.

## Overview

The [`hack`](.) directory contains scripts that ensure continuous development of kubernetes,
enhance the robustness of the code, improve development efficiency, etc.
The explanations and descriptions of these scripts are helpful for contributors.
For details, refer to the following guidelines.

## Key scripts

* [`verify-all.sh`](verify-all.sh): This script is a vestigial redirection. Do not add "real" logic. Runs as `make verify`.
* [`update-all.sh`](update-all.sh): This script is a vestigial redirection. Do not add "real" logic.
The `true` target of this makerule is `hack/make-rules/update.sh`. Runs as `make update`.

## Attention
Run all scripts from the Kubernetes root directory.
**Run `hack/verify-all.sh` before submitting a PR. If anything fails, run `hack/update-all.sh`**.



