# Package: tolerations

## Purpose
Provides utilities for working with pod tolerations, including merging and whitelist verification.

## Key Functions
- `VerifyAgainstWhitelist()` - Checks if tolerations satisfy a whitelist
- `MergeTolerations()` - Merges two toleration sets, removing redundant entries

## Superset Logic
A toleration `ss` is a superset of `t` if:
- Keys match (or ss.Key is empty with Exists operator)
- Effects match (or ss.Effect is empty)
- For NoExecute: ss.TolerationSeconds >= t.TolerationSeconds
- For Equal operator: values must match
- For Exists operator: always matches values

## Design Patterns
- Whitelist verification ensures pods only use allowed tolerations
- Merge removes tolerations that are subsets of others
- Preserves first toleration when duplicates exist
- Uses semantic equality from k8s.io/apimachinery
- Used by admission controllers and pod mutation webhooks
