# Package: validation

## Purpose
Provides comprehensive validation functions for admissionregistration API types including webhooks, CEL-based admission policies, and their bindings.

## Key Validation Functions

### Webhook Validation
- `ValidateValidatingWebhookConfiguration` / `ValidateMutatingWebhookConfiguration` - Full webhook config validation
- `ValidateValidatingWebhookConfigurationUpdate` / `ValidateMutatingWebhookConfigurationUpdate` - Update validation

### CEL Policy Validation
- `ValidateValidatingAdmissionPolicy` / `ValidateValidatingAdmissionPolicyUpdate` - Validate CEL policies
- `ValidateValidatingAdmissionPolicyBinding` / `ValidateValidatingAdmissionPolicyBindingUpdate` - Validate policy bindings
- `ValidateMutatingAdmissionPolicy` / `ValidateMutatingAdmissionPolicyUpdate` - Validate mutating policies
- `ValidateMutatingAdmissionPolicyBinding` / `ValidateMutatingAdmissionPolicyBindingUpdate` - Validate mutating bindings

## Validated Elements
- Rules (APIGroups, APIVersions, Resources, Scope)
- WebhookClientConfig (URL format, Service reference, CABundle)
- AdmissionReviewVersions (must include supported versions)
- MatchConditions (CEL expression syntax, cost limits)
- Validations/Mutations (CEL expression compilation, message expressions)
- FailurePolicy, MatchPolicy, SideEffects
- NamespaceSelector, ObjectSelector
- Timeout constraints (1-30 seconds)

## Key Variables
- `AcceptedAdmissionReviewVersions` - [v1, v1beta1] - versions the API server accepts
