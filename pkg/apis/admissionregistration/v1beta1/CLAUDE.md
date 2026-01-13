# Package: v1beta1

## Purpose
Provides conversion, defaulting, and registration functions for the v1beta1 version of the admissionregistration API.

## Key Defaulting Functions
- `SetDefaults_ValidatingWebhook` - Sets FailurePolicy=Ignore, MatchPolicy=Exact, SideEffects=Unknown, TimeoutSeconds=30, AdmissionReviewVersions=[v1beta1]
- `SetDefaults_MutatingWebhook` - Same as ValidatingWebhook plus ReinvocationPolicy=Never
- `SetDefaults_ServiceReference` - Sets Port=443
- `SetDefaults_ValidatingAdmissionPolicySpec` - Sets FailurePolicy=Fail
- `SetDefaults_MatchResources` - Sets MatchPolicy=Equivalent, empty selectors
- `SetDefaults_MutatingAdmissionPolicySpec` - Sets FailurePolicy=Fail

## Key Differences from v1
- FailurePolicy defaults to Ignore (v1 uses Fail)
- MatchPolicy defaults to Exact (v1 uses Equivalent)
- TimeoutSeconds defaults to 30 (v1 uses 10)
- SideEffects defaults to Unknown (v1 requires explicit value)
- AdmissionReviewVersions defaults to [v1beta1] (v1 requires explicit value)

## Build Tags
- `+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/admissionregistration`
- `+k8s:defaulter-gen=TypeMeta`
- `+groupName=admissionregistration.k8s.io`
