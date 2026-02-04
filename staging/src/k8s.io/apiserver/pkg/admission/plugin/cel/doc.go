/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package cel provides shared core functionality for both validating and mutating admission policies.
// It bridges the generic admission control interfaces with the CEL runtime, handling compilation,
// composition, and execution of CEL expressions against admission requests.
//
// # Core Components
//
// Compilation (compile.go)
//
// The Compiler interface and its implementations are the engine room of the plugin. They are responsible
// for transforming raw CEL string expressions into executable programs that can be run against admission requests.
//
//   - Environment: The plugin builds upon the base CEL environment defined in k8s.io/apiserver/pkg/cel/environment.
//     It extends this base with admission-specific variables like request (AdmissionRequest attributes), namespace
//     (the namespace object), and authorizer.
//
// Composition (composition.go)
//
// Composition is a powerful feature that allows policy authors to define reusable logic and avoid repetition.
//
//   - CompositedCompiler: This is a sophisticated component that manages the complexity of having multiple types of
//     expressions within a single policy. A single ValidatingAdmissionPolicy might contain match conditions, parameter
//     definitions, variable definitions, validation rules, and audit annotations. The CompositedCompiler orchestrates
//     the compilation of all these parts. It acts as a central factory, initializing the correct environment for each
//     context (e.g., ensuring params are available only if paramKind is set) and managing the dependencies between them.
//   - Variables & Lifting: ValidatingAdmissionPolicy and MutatingAdmissionPolicy support a variables list. These are
//     named CEL expressions. The composition logic handles "lifting" these variables into the scope of subsequent
//     expressions. For example, a variable isStable defined as object.spec.replicas < 5 becomes available as a typed
//     boolean variable in all subsequent validation rules. This enables modular policy design.
//   - Lazy Evaluation: To ensure efficiency, variable evaluation is lazy. The expression for a variable is not executed
//     immediately upon policy evaluation. Instead, it is computed only the first time it is referenced by another rule.
//     If a request is denied by a precondition and the complex variable is never used, its cost is never incurred.
//
// # Type Checking Mechanics
//
// The plugin architecture supports strict, schema-aware type checking to ensure policy validity before execution.
//
//   - Environment Construction: A specialized type-checking environment is constructed (via buildEnvSet) which differs
//     from the runtime environment. It replaces dynamic types (DynType) with specific DeclType definitions derived from
//     OpenAPI schemas.
//   - Strongly Typed Variables: Key variables like object, oldObject, and params are strongly typed in this mode.
//     This allows the compiler to catch field access errors (e.g., typos, invalid types) at compile time.
//   - Type Overwrite: The TypeChecker (specifically in staging/src/k8s.io/apiserver/pkg/admission/plugin/policy/validating/typechecking.go)
//     iterates through all GroupVersionKinds (GVKs) matched by a policy. For each GVK, it injects the specific schema types into the compiler context,
//     effectively "overwriting" the generic types for that specific pass.
//   - Limit: To ensure performance, a hard limit (e.g., 10) is often placed on the number of distinct GVKs checked per policy.
//
// # ValidatingAdmissionPolicy
//
// ValidatingAdmissionPolicy (VAP) allows administrators to define validation rules using CEL, offering a declarative
// alternative to ValidatingAdmissionWebhooks.
//
//   - Implementation: The core logic resides in staging/src/k8s.io/apiserver/pkg/admission/plugin/policy/validating.
//   - Type Checking: The TypeChecker (in staging/src/k8s.io/apiserver/pkg/admission/plugin/policy/validating/typechecking.go) performs a multi-pass validation. It resolves all
//     GroupVersionKinds (GVKs) that the policy matches (via matchConstraints). It then compiles the policy's expressions
//     individually against each resolved GVK. For example, if a policy matches both apps/v1 Deployment and apps/v1 StatefulSet,
//     the checker verifies that object.spec.replicas exists and has a compatible type in both schemas. If object.spec.template
//     is referenced but only exists in one, type checking will report an error for the incompatible GVK.
//   - Execution: The dispatcher evaluates the policy. It handles the retrieval of the param resource (if configured) and
//     the namespace object, injecting them into the execution context alongside the admission request and the object under test.
//   - Cost Management: A strict runtime cost budget is enforced. If the evaluation complexity exceeds the limit, the policy
//     evaluation is aborted to protect the API server.
//
// # MutatingAdmissionPolicy
//
// MutatingAdmissionPolicy (MAP) enables modifying objects (setting defaults, injecting sidecars) using CEL.
//
//   - Implementation: Located in staging/src/k8s.io/apiserver/pkg/admission/plugin/policy/mutating.
//   - Mutations: MAP supports two styles of mutation:
//     1. JSONPatch: The expression returns a list of standard JSON Patch operations (add, remove, replace).
//     2. ApplyConfiguration: The expression returns a structured object (an "Apply Configuration") that is merged onto the
//     existing object, similar to a Server-Side Apply operation.
//   - Reinvocation: This handles the complex interaction between multiple admission plugins. If reinvocationPolicy is set
//     to IfNeeded, the policy allows itself to be re-run if a subsequent admission plugin (e.g., a mutating webhook or
//     another policy) modifies the object. The dispatcher tracks the object state and re-triggers the policy to ensure its
//     mutations (like adding a mandatory label) are still present after other plugins have run. A limit on the number of
//     reinvocations prevents infinite loops.
//   - Dispatcher: The dispatcher orchestrates this flow, applying the generated patches to the in-flight object version
//     and managing the "dirty" state of the object attributes.
//
// # Shared Mechanics
//
//   - Filter/Matcher: Both policies use a common Matcher to determine applicability. This evaluates matchConstraints
//     (resource, group, user), matchResources (namespace selectors, object selectors), and paramRef binding validity.
//   - Match Conditions: These are high-performance CEL filters evaluated before the main policy logic or parameter retrieval.
//     If a matchCondition evaluates to false or error, the policy is skipped immediately. This allows for very granular
//     filtering (e.g., "only validate Pods with a specific annotation") without incurring the cost of loading parameters
//     or running complex validation rules.
//   - Configuration: Policies are configured via the API resources ValidatingAdmissionPolicy and MutatingAdmissionPolicy,
//     and bound to specific resources via ...Binding resources.
//   - Metrics: Standardized metrics are collected for policy evaluation counts, latency (broken down by check vs. execution),
//     and error rates, enabling observability into the admission control pipeline.
package cel
