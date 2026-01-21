---
name: kubelet-verify
description: Run verification checks before submitting PRs. Use when checking code formatting, linting, or running pre-submit validations.
allowed-tools:
  - Bash
  - Read
---

# Kubelet Verification

Run code verification checks required before submitting Kubernetes PRs.

## Instructions

When the user wants to verify their code:

1. **Quick verification** for fast feedback:
```bash
make quick-verify
```

2. **Full verification** before final PR:
```bash
hack/verify-all.sh
```

3. **If verification fails**, run updates:
```bash
hack/update-all.sh
```

## Verification Commands

### Quick Checks
```bash
# Fast verification (most common issues)
make quick-verify

# Just Go formatting
hack/verify-gofmt.sh

# Just linting
make lint
```

### Full Verification
```bash
# All checks (slow but thorough)
hack/verify-all.sh
```

### Individual Checks
```bash
# Go formatting
hack/verify-gofmt.sh

# Import organization
hack/verify-imports.sh

# License headers
hack/verify-boilerplate.sh

# Generated code
hack/verify-codegen.sh

# Linting with golangci-lint
hack/verify-golangci-lint.sh

# Vendor dependencies
hack/verify-vendor.sh
```

### Auto-Fix Issues
```bash
# Fix all auto-fixable issues
hack/update-all.sh

# Fix specific issues
hack/update-gofmt.sh         # Format code
hack/update-codegen.sh       # Regenerate code
hack/update-vendor.sh        # Update vendor
```

## Common Issues and Fixes

| Verification | Fix Command |
|--------------|-------------|
| gofmt | `hack/update-gofmt.sh` or `gofmt -w .` |
| imports | `hack/update-import-aliases.sh` |
| codegen | `hack/update-codegen.sh` |
| boilerplate | Add license header manually |
| vendor | `hack/update-vendor.sh` |

## Pre-PR Checklist

1. Run `hack/verify-gofmt.sh` - formatting
2. Run `make lint` - linting
3. Run tests - `go test ./pkg/kubelet/...`
4. Run `hack/verify-all.sh` - full verification

## Examples

User: "Check if my code is ready for PR"
```bash
# Quick check first
make quick-verify

# If that passes, full verification
hack/verify-all.sh
```

User: "Fix formatting issues"
```bash
hack/update-gofmt.sh
```

User: "Run the linter"
```bash
make lint
```

## Tips

- Always run verification before pushing
- `hack/update-all.sh` fixes most issues automatically
- License headers must be added manually
- Generated code changes require `hack/update-codegen.sh`
