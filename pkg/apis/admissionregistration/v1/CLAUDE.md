# Package: v1

## Purpose
Provides conversion, defaulting, and registration functions for the v1 stable version of the admissionregistration API.

## Key Defaulting Functions
- `SetDefaults_ValidatingWebhook` - Sets FailurePolicy=Fail, MatchPolicy=Equivalent, TimeoutSeconds=10, empty selectors
- `SetDefaults_MutatingWebhook` - Same as ValidatingWebhook plus ReinvocationPolicy=Never
- `SetDefaults_Rule` - Sets Scope=AllScopes
- `SetDefaults_ServiceReference` - Sets Port=443
- `SetDefaults_ValidatingAdmissionPolicySpec` - Sets FailurePolicy=Fail
- `SetDefaults_MatchResources` - Sets MatchPolicy=Equivalent, empty selectors

## Key Differences from v1beta1
- FailurePolicy defaults to Fail (not Ignore)
- MatchPolicy defaults to Equivalent (not Exact)
- TimeoutSeconds defaults to 10 (not 30)
- SideEffects is required (no default)
- AdmissionReviewVersions is required (no default)

## Build Tags
- `+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/admissionregistration`
- `+k8s:defaulter-gen=TypeMeta`
- `+groupName=admissionregistration.k8s.io`
