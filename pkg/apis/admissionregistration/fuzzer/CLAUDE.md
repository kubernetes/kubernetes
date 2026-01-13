# Package: fuzzer

## Purpose
Provides fuzzer functions for the admissionregistration API group used in property-based testing, ensuring required fields have valid default values.

## Key Variables
- `Funcs` - Returns fuzzer functions for admissionregistration types

## Fuzzer Functions
- `Rule` - Ensures Scope defaults to AllScopes
- `ValidatingWebhook` - Sets defaults for FailurePolicy (Fail), MatchPolicy (Exact), SideEffects (Unknown), TimeoutSeconds (30), AdmissionReviewVersions (v1beta1)
- `MutatingWebhook` - Same as ValidatingWebhook plus ReinvocationPolicy (Never)
- `ValidatingAdmissionPolicySpec` - Sets FailurePolicy to Fail
- `ValidatingAdmissionPolicyBindingSpec` - Sets ValidationActions to [Deny]
- `MatchResources` - Sets MatchPolicy to Exact
- `ParamRef` - Sets ParameterNotFoundAction to Deny
- `MutatingAdmissionPolicySpec` - Sets FailurePolicy to Fail, ReinvocationPolicy to Never
- `Mutation` - Randomly chooses between JSONPatch and ApplyConfiguration patch types

## Design Notes
- Ensures fuzzed objects pass validation by setting required defaults
- Particularly important for pointer fields that have no natural default
