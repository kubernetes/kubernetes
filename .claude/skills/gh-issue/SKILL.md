---
name: gh-issue
description: View and manage GitHub issues using gh CLI. Use when viewing issues, searching for related issues, or understanding issue context.
allowed-tools:
  - Bash
  - Read
  - WebFetch
---

# GitHub Issue Operations

View and manage GitHub issues for Kubernetes development.

## Instructions

When the user wants to work with issues:

1. **Identify the operation** (view, search, comment)
2. **Run the appropriate gh command**
3. **Summarize key information** from the issue

## Common Operations

### View an Issue
```bash
# View issue details
gh issue view <number>

# View in browser
gh issue view <number> --web

# View with comments
gh issue view <number> --comments
```

### Search Issues
```bash
# Search by keyword
gh issue list --search "NodeInfo cache"

# Filter by label
gh issue list --label "area/kubelet"
gh issue list --label "sig/node"

# Filter by state
gh issue list --state open --label "kind/bug"

# Your assigned issues
gh issue list --assignee @me
```

### Issue Details in JSON
```bash
# Get structured data
gh issue view <number> --json title,body,labels,comments

# Get just labels
gh issue view <number> --json labels --jq '.labels[].name'
```

### Comment on Issue
```bash
# Add a comment
gh issue comment <number> --body "Working on this"
```

## Kubernetes-Specific Searches

### Find Good First Issues
```bash
gh issue list --label "good first issue" --state open
```

### Find Kubelet Issues
```bash
gh issue list --label "area/kubelet" --state open
gh issue list --label "sig/node" --state open
```

### Find Feature Requests
```bash
gh issue list --label "kind/feature" --label "sig/node" --state open
```

### Find Bugs
```bash
gh issue list --label "kind/bug" --label "area/kubelet" --state open
```

## Useful Queries

### Issues by Priority
```bash
# Critical issues
gh issue list --label "priority/critical-urgent" --state open

# High priority
gh issue list --label "priority/important-soon" --state open
```

### Issues Needing Help
```bash
gh issue list --label "help wanted" --state open
gh issue list --label "good first issue" --state open
```

### Related to Specific Topic
```bash
# NodeInfo caching
gh issue list --search "NodeInfo" --label "area/kubelet"

# Pod admission
gh issue list --search "admission" --label "sig/node"
```

## Examples

User: "Show me issue 132858"
```bash
gh issue view 132858
```

User: "Find kubelet bugs"
```bash
gh issue list --label "kind/bug" --label "area/kubelet" --state open --limit 20
```

User: "Search for caching issues"
```bash
gh issue list --search "cache" --label "area/kubelet" --state open
```

## Understanding Issue Labels

| Label | Meaning |
|-------|---------|
| `sig/node` | SIG Node ownership |
| `area/kubelet` | Kubelet component |
| `kind/bug` | Bug report |
| `kind/feature` | Feature request |
| `kind/cleanup` | Code cleanup |
| `priority/critical-urgent` | Must fix immediately |
| `priority/important-soon` | Fix in next release |
| `good first issue` | Suitable for new contributors |
| `help wanted` | Community help welcome |

## Tips

- Use `--web` to open complex issues in browser
- Use `--json` with `--jq` for scripting
- Labels are the primary way to find relevant issues
- KEPs (Kubernetes Enhancement Proposals) track major features
