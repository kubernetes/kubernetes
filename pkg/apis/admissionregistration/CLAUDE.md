# Package: admissionregistration

## Purpose
Defines the internal types for admission webhook and admission policy configuration, used to configure how the API server invokes admission webhooks and evaluates CEL-based admission policies.

## Key Types

### Webhook Configuration
- `ValidatingWebhookConfiguration` / `ValidatingWebhook` - Configuration for validating admission webhooks
- `MutatingWebhookConfiguration` / `MutatingWebhook` - Configuration for mutating admission webhooks
- `WebhookClientConfig` - How to connect to the webhook (URL or Service reference)
- `RuleWithOperations` - What resources and operations trigger the webhook

### CEL-Based Admission Policies
- `ValidatingAdmissionPolicy` / `ValidatingAdmissionPolicySpec` - CEL-based validation without webhooks
- `ValidatingAdmissionPolicyBinding` - Binds a policy to resources with parameters
- `MutatingAdmissionPolicy` / `MutatingAdmissionPolicySpec` - CEL-based mutation (introduced v1.32)
- `MutatingAdmissionPolicyBinding` - Binds a mutating policy to resources

### Supporting Types
- `Rule` - API groups, versions, resources, and scope
- `MatchResources` - Namespace/object selectors and resource rules
- `MatchCondition` - CEL conditions for fine-grained matching
- `Validation` - CEL expression with message and reason
- `Mutation` / `ApplyConfiguration` / `JSONPatch` - Mutation specifications
- `ParamRef` / `ParamKind` - Parameter references for policies

## Key Constants
- `FailurePolicyType`: Ignore, Fail
- `MatchPolicyType`: Exact, Equivalent
- `SideEffectClass`: Unknown, None, Some, NoneOnDryRun
- `ValidationAction`: Deny, Warn, Audit
- `OperationType`: CREATE, UPDATE, DELETE, CONNECT
- `ReinvocationPolicyType`: Never, IfNeeded
