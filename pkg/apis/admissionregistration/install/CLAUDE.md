# Package: install

## Purpose
Installs the admissionregistration API group into the legacy scheme, making it available for encoding/decoding admission webhook and policy configurations.

## Key Functions
- `Install(scheme *runtime.Scheme)` - Registers all admissionregistration types and versions
- `init()` - Automatically installs to the legacy scheme on package import

## Registered Versions
1. Internal version (hub) - `k8s.io/kubernetes/pkg/apis/admissionregistration`
2. v1beta1 - Mutating/ValidatingWebhookConfiguration, ValidatingAdmissionPolicy (beta)
3. v1alpha1 - MutatingAdmissionPolicy (alpha, introduced v1.32)
4. v1 - Mutating/ValidatingWebhookConfiguration (stable), ValidatingAdmissionPolicy (stable)

## Version Priority
v1 > v1beta1 > v1alpha1

## Design Notes
- Uses the standard install pattern for Kubernetes API groups
- Panics on registration errors via `utilruntime.Must()`
