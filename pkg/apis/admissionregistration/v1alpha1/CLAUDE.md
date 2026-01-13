# Package: v1alpha1

## Purpose
Provides conversion, defaulting, and registration functions for the v1alpha1 version of the admissionregistration API, primarily for MutatingAdmissionPolicy (introduced in v1.32).

## Key Defaulting Functions
- `SetDefaults_ValidatingAdmissionPolicySpec` - Sets FailurePolicy=Fail
- `SetDefaults_MatchResources` - Sets MatchPolicy=Equivalent, empty namespace/object selectors
- `SetDefaults_ParamRef` - Sets ParameterNotFoundAction=Deny
- `SetDefaults_MutatingAdmissionPolicySpec` - Sets FailurePolicy=Fail

## API Resources
- ValidatingAdmissionPolicy / ValidatingAdmissionPolicyBinding (also available in v1beta1, v1)
- MutatingAdmissionPolicy / MutatingAdmissionPolicyBinding (alpha, v1alpha1 only)

## Build Tags
- `+k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/admissionregistration`
- `+k8s:defaulter-gen=TypeMeta`
- `+groupName=admissionregistration.k8s.io`
