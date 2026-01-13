# Kubernetes Metrics Stability and Graduation Guidelines Summary

## Overview

This document summarizes the official Kubernetes metrics stability and graduation guidelines based on findings in the Kubernetes repository and linked documentation.

## Sources

1. **KEP-1209**: Control Plane Metrics Stability
   - Referenced throughout the codebase at: `https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/kubernetes-control-plane-metrics-stability.md`
   - Not included in the kubernetes/kubernetes repository but extensively referenced

2. **Metric Instrumentation Documentation**: 
   - Located at: `https://github.com/kubernetes/community/blob/9de615daee15bdee370a825dc0ef150eac189700/contributors/devel/sig-instrumentation/metric-instrumentation.md`

## Stability Level Definitions

From `staging/src/k8s.io/component-base/metrics/opts.go`:

### ALPHA
- **Stability guarantees**: None
- **Label changes**: Labels may be arbitrarily added/removed
- **Removal**: Metric may be deleted at any time
- **Default**: All new metrics default to ALPHA if not specified

### BETA
- **Stability guarantees**: Governed by the deprecation policy outlined in the control plane metrics stability KEP
- **Label changes**: 
  - Labels **cannot be removed** during Beta lifetime
  - Labels **can be added** while metric is in Beta stage
  - This ensures existing dashboards/alerts continue to work while allowing future enhancements
- **Deprecation period**: 1 minor release after deprecation before being hidden
  (from `staging/src/k8s.io/component-base/metrics/registry.go`)

### STABLE
- **Stability guarantees**: Guaranteed not to be mutated
- **Label changes**: No labels can be added or removed during lifetime
- **Removal**: Governed by the deprecation policy in the control plane metrics stability KEP
- **Deprecation period**: 3 minor releases after deprecation before being hidden

## Graduating from Alpha to Beta - Exact Criteria

From `https://github.com/kubernetes/community/blob/9de615daee15bdee370a825dc0ef150eac189700/contributors/devel/sig-instrumentation/metric-instrumentation.md#graduating-to-beta`:

### 1. Testing Requirement
The metric must have a corresponding test that validates:
- The metric is **registered and emitted correctly**
- The metric has the **expected labels and values** under known conditions

### 2. Documentation
- Ensure the metric has a **clear and accurate help text description**
- The help text should clearly explain the metric's purpose and usage

### 3. API Review
**Documentation Text**: The "Graduating to Beta" section includes: "API Review: Graduating a metric to Stable requires an API review by SIG Instrumentation, as it represents a contractual API agreement. See the [API Review](/contributors/devel/sig-instrumentation/metric-stability.md#api-review) section in the metrics stability documentation."

**Note**: This text explicitly mentions "Graduating a metric to Stable" (not Beta), suggesting it may be a copy-paste error from the Stable graduation section.

**From metric-stability.md API Review documentation**:
- API Review uses a verification script to flag **stable metric changes** for review by SIG Instrumentation approvers
- Graduating a metric to a **stable state** is a contractual API agreement requiring api-review to sig-instrumentation
- The API Review process documentation focuses on "stable state" metrics

**Interpretation**: The API Review documentation and tooling specifically target stable metrics. Beta graduation may not require formal API review, though the component owner must acknowledge support. The documentation text appears inconsistent and may need clarification from SIG Instrumentation.

## Additional Requirements from Code Comments

From various metric files (e.g., `pkg/volume/util/metrics.go`, `staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go`):

### Component Owner Responsibility
- Promoting the stability level is the **responsibility of the component owner**
- Requires **explicitly acknowledging support for the metric across multiple releases**
- Must be done in accordance with the metric stability policy

## Graduating from Beta to Stable - For Reference

The metric must meet all requirements for Beta graduation, plus:

1. **Testing requirement**: 
   - Must be included in the [stable metrics list](https://github.com/kubernetes/kubernetes/blob/master/test/instrumentation/testdata/stable-metrics-list.yaml)
   - See [instrumentation test README](https://github.com/kubernetes/kubernetes/tree/master/test/instrumentation/README.md) for generation steps

2. **Stability validation**: 
   - Should have been at Beta stability for **at least one release** to ensure sufficient production validation

3. **API Review**: 
   - Requires an API review by SIG Instrumentation
   - Represents a contractual API agreement

## Implementation Details in Repository

### Test Infrastructure
- Location: `test/instrumentation/`
- Contains regression tests for controlling the list of stable metrics
- Changes to stable metrics require review by sig-instrumentation
- Uses golden test files in `test/instrumentation/testdata/`

### Documentation Generation
- Auto-generates metrics documentation
- Categorizes metrics by stability level (Alpha, Beta, Stable)
- Template in `test/instrumentation/documentation/main.go`

### Deprecation Implementation
- BETA metrics: 1 minor release deprecation period (from `registry.go`)
- STABLE metrics: 3 minor releases deprecation period
- Deprecated metrics are prefixed with deprecation notice
- Hidden metrics can be enabled via command line flag

## Summary of Alpha to Beta Promotion Criteria

To promote a metric from Alpha to Beta, you must:

1. ✅ **Create tests** that validate:
   - Metric registration and emission
   - Expected labels and values under known conditions

2. ✅ **Provide clear documentation** with accurate help text

3. ✅ **Get component owner acknowledgment** for supporting the metric across multiple releases

4. ⚠️ **API Review** (Documentation mentions API review but text refers to "Stable" - appears inconsistent. API review process specifically targets stable metrics. Component owner acknowledgment is definitely required.)

5. ✅ **Understand API guarantees**: Once promoted to Beta, labels cannot be removed (but can be added), providing stability for existing dashboards/alerts
