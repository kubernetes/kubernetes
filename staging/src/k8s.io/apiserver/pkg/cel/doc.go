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

// Package cel provides a comprehensive overview of the Common Expression Language (CEL) integration within Kubernetes,
// specifically focusing on the k8s.io/apiserver/pkg/cel package.
//
// # Context & History
//
// CEL was selected for Kubernetes to handle CRD validation and admission control after a thorough evaluation of alternatives.
//
//   - Alternatives Considered:
//
//   - WebAssembly (Wasm): While powerful, Wasm was deemed too complex for the target use cases. It introduces significant
//     operational overhead for managing binaries, potential "cold start" latency issues, and a more complex developer
//     workflow compared to in-line expressions.
//
//   - Rego (OPA): Rego is a robust policy language but operates as a separate engine. Integrating it would require
//     external webhooks or a heavier embedding, whereas CEL offers a lightweight, expression-based syntax that fits
//     naturally within Kubernetes YAML manifests.
//
//   - Why CEL?
//
//   - Low Overhead: CEL is designed for microseconds-level execution, making it suitable for high-throughput admission
//     control where every millisecond counts.
//
//   - Type Safety: CEL provides compile-time type checking. This allows the API server to reject invalid policies
//     before they are accepted, preventing runtime errors during admission.
//
//   - Simplicity: It allows logic to be embedded directly in CRD definitions (validation rules) or Policy objects,
//     removing the need to build, deploy, and manage separate webhook infrastructure.
//
// # Core Architecture
//
// # Use Cases
//
// CEL is currently utilized in several key areas of Kubernetes, becoming the standard for in-process extensibility:
//  1. CRD Validation: The initial use case, allowing CRD authors to define complex validation rules (e.g., "minReplicas < maxReplicas")
//     that OpenAPI schema validation cannot express.
//  2. Admission Control: ValidatingAdmissionPolicy and MutatingAdmissionPolicy allow cluster administrators to enforce policy
//     and modify objects on built-in types (like Pods, PVCs) without writing Go code or deploying webhooks.
//  3. Webhook Match Conditions: Allows filtering requests before they are sent to a remote webhook, reducing unnecessary traffic and latency.
//  4. AuthN/AuthZ: Used in SubjectAccessReview and other authorization paths to manage dynamic access controls.
//  5. Dynamic Resource Allocation (DRA): Used to configure complex resource parameters and perform matching logic within the scheduler.
//
// # Type System Integration
//
// The integration uses CelGo to bridge Kubernetes native Go types with CEL's runtime values.
//
//   - ref.Val Interface: Kubernetes internal types (both structured Go structs and unstructured map[string]interface{}) are
//     adapted to implement CEL's ref.Val interface via staging/src/k8s.io/apiserver/pkg/cel/types.go and value.go.
//   - Lazy Evaluation: This is a critical performance optimization. Instead of converting a whole Kubernetes object (which can be very large)
//     into a CEL-native map before evaluation, the integration uses reflection-based wrappers. Fields are accessed and converted
//     only when the CEL expression actually reads them. This "pay-for-what-you-use" model significantly reduces memory allocation
//     and CPU usage. Implementation details can be found in staging/src/k8s.io/apiserver/pkg/cel/lazy.
//   - Compile-Time Safety: The system uses the OpenAPI schema information (decl.Type) to inform the CEL compiler about the structure
//     of the input variables. This allows the compiler to catch errors like accessing non-existent fields or comparing incompatible
//     types (e.g., string vs int) at configuration time.
//
// Compilation & Optimization
//
//   - Lifecycle: CEL expressions are not interpreted raw at runtime. They are compiled into an executable program representation
//     when the API object (CRD, ValidatingAdmissionPolicy) is written to the API server. This compilation step performs syntax
//     parsing and type checking.
//   - Caching: The compiled programs are stored in an in-memory cache within the API server. This means the heavy lifting of
//     parsing and checking is done once (amortized), and runtime evaluation involves executing the efficient, pre-compiled program.
//     These cached programs are thread-safe and shared across concurrent requests.
//   - Two-Step Process for CRDs:
//     1. Validator Construction: When a CRD is loaded, a validator is constructed based on its OpenAPIV3 schema.
//     2. Compilation: Any CEL validation rules within the schema are found and compiled. These rules are then stored alongside the schema validator.
//   - Admission Policies: For ValidatingAdmissionPolicy, an informer-based controller monitors the policy resources. When a policy
//     is added or updated, the controller asynchronously compiles the expressions and updates the ready status of the policy.
//   - Ahead-of-Time (AOT): Currently, there is no AOT compilation to native machine code (Go/Assembly). Serialization of compiled
//     results to proto is supported by the library but is not currently used for execution persistence in Kubernetes.
//
// # Available CEL Libraries
//
// Kubernetes provides a rich set of CEL libraries, managed via EnvSet in pkg/cel/environment. Libraries are versioned to ensure backward compatibility.
//
// Standard Libraries (Core)
//
//   - URLs: URL parsing and manipulation (Always available).
//     Methods: url, isURL, getScheme, getHost, getHostname, getPort, getEscapedPath, getQuery.
//     Cost: Parsing functions (url, isURL) cost is proportional to string length. Accessors cost 1 (nominal).
//   - Regex: Regular expression matching (Always available).
//     Methods: find, findAll.
//     Cost: Proportional to string length * regex length.
//   - Lists: List manipulation utilities (Always available).
//     Methods: isSorted, sum, min, max, indexOf, lastIndexOf.
//     Cost: Proportional to list size (O(n) traversal).
//
// Kubernetes-Specific Libraries
//
//   - Authz: Authorization checks (Introduced v1.27).
//     Methods: path, group, serviceAccount, resource, subresource, namespace, name, check, allowed, reason, errored, error.
//     Cost: check() has a fixed cost of 350,000. Builders and accessors have a nominal cost of 1.
//   - AuthzSelectors: Authorization with field/label selectors (Introduced v1.31).
//     Methods: fieldSelector, labelSelector.
//     Cost: Proportional to the length of the selector string.
//   - Quantity: Support for Kubernetes resource quantities (Introduced v1.28).
//     Methods: quantity, isQuantity, sign, add, sub, isGreaterThan, isLessThan, compareTo, asInteger, asApproximateFloat, isInteger.
//     Cost: Parsing (quantity) is proportional to string length. Arithmetic and comparisons have a nominal cost of 1.
//   - IP / CIDR: IP address and CIDR block handling (Introduced v1.30).
//     Methods: ip, isIP, ip.isCanonical, family, isUnspecified, isLoopback, isLinkLocalMulticast, isLinkLocalUnicast,
//     isGlobalUnicast, cidr, isCIDR, containsIP, containsCIDR, prefixLength, masked.
//     Cost: Parsing is proportional to string length. Accessors cost 1. Containment checks (containsIP, containsCIDR)
//     are proportional to input size.
//   - Format: Standardized string formatting (Introduced v1.31).
//     Methods: validate, format.named.
//     Cost: validate cost is proportional to string length * estimated regex length. format.named costs 1.
//   - Semver: Semantic versioning support (Introduced v1.33).
//     Methods: semver, isSemver, major, minor, patch, compareTo, isGreaterThan, isLessThan.
//     Cost: Parsing is proportional to string length. Comparisons and accessors have a nominal cost of 1.
//   - JSONPatch: Creation of JSON Patch operations (Used in MutatingAdmissionPolicy).
//     Methods: jsonpatch.escapeKey.
//     Cost: Proportional to string length (traversal).
//
// Extension Libraries
//
//   - Strings: String manipulation (Version 2 active since v1.29).
//     Methods: includes split, join, replace, lowerAscii, upperAscii, substring, trim.
//     Cost: Generally proportional to string traversal + result construction cost.
//   - Sets: Set operations (Introduced v1.29).
//   - TwoVarComprehensions: Support for two-variable comprehensions (Introduced v1.32).
//   - Lists Extension: Extended list features (Version 3 active since v1.34).
//
// Core Features
//
//   - CrossTypeNumericComparisons: Enabled since v1.28.
//   - OptionalTypes: Enabled since v1.28.
//   - DefaultUTCTimeZone: Enabled by default.
//
// # The Cost System
//
// A critical component for running user-defined logic in the API server is resource protection. Since CEL executes in the
// same process as the API server, a runaway expression could degrade the performance of the entire control plane. The Cost
// System mitigates this risk by assigning a cost to every operation and enforcing strict budgets.
//
//   - Purpose: The primary goal is to prevent Denial of Service (DoS) attacks or accidental performance degradation caused
//     by computationally expensive expressions. Unlike webhooks, which run externally and mainly affect the request latency
//     via I/O blocking, CEL runs locally and consumes shared CPU and memory resources.
//   - Cost Unit: Costs are abstract units, but they are roughly calibrated such that 1 cost unit corresponds to approximately
//     50 nanoseconds of execution time on a reference CPU. This provides a deterministic metric independent of momentary system load.
//   - Estimation (Static): Before a policy is accepted, the system performs a static cost estimation. It calculates the worst-case
//     running time by analyzing the AST. It uses the MaxItems and MaxLength constraints from the CRD schema to bound the cost of
//     list traversals and string operations. If the estimated cost exceeds the limit, the policy is rejected at configuration time.
//   - Enforcement (Runtime): During execution, the interpreter tracks the actual accumulated cost. If the cost exceeds the
//     RuntimeCELCostBudget (defined in staging/src/k8s.io/apiserver/pkg/apis/cel/config.go), execution is halted immediately,
//     and the validation fails (or defaults to the failure policy).
//
// High-Cost Operations
//
//   - Traversal: Iterating over lists or maps (e.g., list.all(...), list.map(...)) has a cost proportional to the size of the data structure (O(N)).
//   - String Operations: Operations like replace, split, or parsing large strings have costs proportional to the string length.
//   - Regex: Functions like find or findAll have costs proportional to string_length * regex_complexity. While the RE2 library
//     prevents catastrophic backtracking, complex regexes on large inputs are still expensive.
//   - Allocations: Creating new lists or strings (e.g., list1 + list2) incurs costs for memory allocation and copying.
//
// # Best Practices for Cost Reduction
//
// To ensure policies are accepted by the API server and do not impact cluster performance, authors should follow these guidelines
// to minimize both estimated (static) and actual (runtime) costs.
//
//   - Define Schema Constraints (Crucial):
//     Why: The static cost estimator calculates the worst-case execution time. If a list does not have a maxItems limit, the
//     estimator must assume it could be infinitely large (up to system limits), resulting in a massive estimated cost that
//     likely exceeds the budget.
//     Action: Always define maxItems for arrays and maxLength for strings in your CRD or API schema. This allows the estimator
//     to mathematically bound the cost of traversals (list.all(...)) and string operations, keeping the estimated cost within the limit.
//
//   - Leverage Short-Circuiting:
//     Why: CEL evaluates logical && (AND) and || (OR) operators using short-circuit logic. If the left-hand side determines
//     the result, the right-hand side is never executed/billed.
//     Action: Place cheap, simple checks before expensive ones.
//     Bad: object.spec.items.all(i, i.matches('^complex-regex$')) && has(object.spec.items)
//     Good: has(object.spec.items) && object.spec.items.size() < 100 && object.spec.items.all(i, i.matches('^complex-regex$'))
//
//   - Use Efficient Libraries:
//     Why: Native library functions are optimized in Go. Manual implementations in CEL are often slower and costlier.
//     Action:
//     Use the Sets library (sets.contains) for membership checks instead of iterating lists.
//     Use startsWith / endsWith for simple string matching instead of full Regex matches.
//     Use ip and cidr libraries for network checks instead of string parsing.
//
//   - Avoid Redundancy with Variables:
//     Why: Repeatedly calculating an expensive value (e.g., parsing a complex annotation) multiplies the cost.
//     Action: In ValidatingAdmissionPolicy, define the expensive calculation once in the variables section. The variable is
//     lazily evaluated once, and subsequent references cost almost nothing.
//
// # Development Patterns
//
// # Backward Compatibility
//
// Kubernetes maintains strict backward compatibility for CEL, managed in staging/src/k8s.io/apiserver/pkg/cel/environment:
//   - Base Environments: Capabilities are mapped to Kubernetes versions via EnvSet.
//   - Rollback Support: New expressions can only utilize features available in $N-1$ versions.
//   - Forward Compatibility: Stored environments must be forward-compatible.
//
// # Mutation
//
// Mutating Admission Policies (Beta) allow CEL to define JSON patches or server-side apply operations.
//   - Logic: Implemented in staging/src/k8s.io/apiserver/pkg/admission/plugin/policy/mutating/dispatcher.go.
//   - Cascading Changes: A retry mechanism re-runs policies if changes occur (randomized order) to handle dependencies between policies.
//
// # Limitations
//
//   - No Cross-Resource Lookups: Restricted for security and performance (except for limited namespace metadata access via namespaceObject).
//   - List Filtering: CEL is generally not used for filtering large lists of objects due to the high cost of compilation and type-checking relative to evaluation.
//   - String Length: Soft limits are encouraged to keep expressions readable and manageable.
package cel
