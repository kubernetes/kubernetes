---
name: gh-pr
description: GitHub PR operations using gh CLI. Use when creating PRs, checking CI status, viewing reviews, or managing pull requests.
allowed-tools:
  - Bash
  - Read
---

# GitHub PR Operations

Manage GitHub pull requests using the `gh` CLI for Kubernetes development.

## Instructions

When the user wants to work with PRs:

1. **Identify the operation** (create, status, checks, review)
2. **Run the appropriate gh command**
3. **Report results clearly**

## Common Operations

### Create a PR
```bash
# Interactive creation
gh pr create

# With title and body
gh pr create --title "kubelet: add NodeInfo caching" --body "Description here"

# With labels
gh pr create --title "kubelet: fix bug" --label "kind/bug" --label "sig/node"

# Draft PR
gh pr create --draft --title "WIP: feature"
```

### Check PR Status
```bash
# View your PRs
gh pr status

# View specific PR
gh pr view <number>

# View PR in browser
gh pr view <number> --web
```

### Check CI Status
```bash
# View CI checks for current branch's PR
gh pr checks

# View specific PR's checks
gh pr checks <number>

# Watch checks until complete
gh pr checks --watch
```

### View Reviews
```bash
# View PR comments
gh pr view <number> --comments

# View review status
gh pr view <number> --json reviews
```

### Update PR
```bash
# Add reviewers
gh pr edit <number> --add-reviewer @username

# Add labels
gh pr edit <number> --add-label "kind/feature"

# Update title
gh pr edit <number> --title "New title"
```

### Merge PR
```bash
# Squash merge (Kubernetes standard)
gh pr merge <number> --squash

# Merge with auto-merge when checks pass
gh pr merge <number> --squash --auto
```

## Kubernetes-Specific Commands

### Add SIG Labels (via comment)
```bash
# Labels are typically added via bot commands in PR comments
gh pr comment <number> --body "/sig node"
gh pr comment <number> --body "/kind feature"
gh pr comment <number> --body "/area kubelet"
```

### Request Review
```bash
gh pr edit <number> --add-reviewer @reviewer1,@reviewer2
```

### Check Prow CI
```bash
# View all checks including Prow jobs
gh pr checks <number>

# Re-run failed tests (via comment)
gh pr comment <number> --body "/retest"
```

## Useful Queries

### List Open PRs
```bash
# Your open PRs
gh pr list --author @me

# PRs needing review
gh pr list --search "review-requested:@me"

# Kubelet PRs
gh pr list --label "area/kubelet"
```

### Search PRs
```bash
# PRs by label
gh pr list --label "sig/node" --state open

# PRs mentioning an issue
gh pr list --search "132858"
```

## Examples

User: "Create a PR for my changes"
```bash
gh pr create --title "kubelet: improve NodeInfo caching" --body "$(cat <<'EOF'
## What this PR does
Implements NodeInfo caching for efficient pod admission.

## Why we need it
Reduces O(n*m) computation on every admission.

Fixes #132858
EOF
)"
```

User: "Check if my PR's CI is passing"
```bash
gh pr checks
```

User: "Add the sig/node label to PR 12345"
```bash
gh pr comment 12345 --body "/sig node"
```

## Tips

- Use `--web` to open PRs in browser for complex operations
- Kubernetes uses Prow for CI - retest with `/retest` comment
- Labels are managed by bot commands, not direct `--add-label`
- Always squash merge to maintain clean history
