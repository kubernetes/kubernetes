/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"context"
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types/ref"

	v1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/environment"
)

type ExpressionAccessor interface {
	GetExpression() string
	ReturnTypes() []*cel.Type
}

// NamedExpressionAccessor extends NamedExpressionAccessor with a name.
type NamedExpressionAccessor interface {
	ExpressionAccessor

	GetName() string // follows the naming convention of ExpressionAccessor
}

// EvaluationResult contains the minimal required fields and metadata of a cel evaluation
type EvaluationResult struct {
	EvalResult         ref.Val
	ExpressionAccessor ExpressionAccessor
	Elapsed            time.Duration
	Error              error
}

// OptionalVariableDeclarations declares which optional CEL variables
// are declared for an expression.
type OptionalVariableDeclarations struct {
	// HasParams specifies if the "params" variable is declared.
	// The "params" variable may still be bound to "null" when declared.
	HasParams bool
	// HasAuthorizer specifies if the "authorizer" and "authorizer.requestResource"
	// variables are declared. When declared, the authorizer variables are
	// expected to be non-null.
	HasAuthorizer bool
	// HasPatchTypes specifies if JSONPatch, Object, Object.metadata and similar types are available in CEL. These can be used
	// to initialize the typed objects in CEL required to create patches.
	HasPatchTypes bool
}

// ConditionCompiler contains a function to assist with converting types and values to/from CEL-typed values.
type ConditionCompiler interface {
	// CompileCondition is used for the cel expression compilation
	CompileCondition(expressions []ExpressionAccessor, optionalDecls OptionalVariableDeclarations, envType environment.Type) ConditionEvaluator
}

// OptionalVariableBindings provides expression bindings for optional CEL variables.
type OptionalVariableBindings struct {
	// VersionedParams provides the "params" variable binding. This variable binding may
	// be set to nil even when OptionalVariableDeclarations.HashParams is set to true.
	VersionedParams runtime.Object
	// Authorizer provides the authorizer used for the "authorizer" and
	// "authorizer.requestResource" variable bindings. If the expression was compiled with
	// OptionalVariableDeclarations.HasAuthorizer set to true this must be non-nil.
	Authorizer authorizer.Authorizer
}

// ConditionEvaluator contains the result of compiling a CEL expression
// that evaluates to a condition. This is used both for validation and pre-conditions.
// It expects the inbound object to already have been converted to the version expected
// by the underlying CEL code (which is indicated by the match criteria of a policy definition).
// versionedParams may be nil.
type ConditionEvaluator interface {
	// ForInput converts compiled CEL-typed values into evaluated CEL-typed value.
	// runtimeCELCostBudget was added for testing purpose only. Callers should always use const RuntimeCELCostBudget from k8s.io/apiserver/pkg/apis/cel/config.go as input.
	// If cost budget is calculated, the condition should return the remaining budget.
	ForInput(ctx context.Context, versionedAttr *admission.VersionedAttributes, request *v1.AdmissionRequest, optionalVars OptionalVariableBindings, namespace *corev1.Namespace, runtimeCELCostBudget int64) ([]EvaluationResult, int64, error)

	// CompilationErrors returns a list of errors from the compilation of the mutatingEvaluator
	CompilationErrors() []error
}

// MutatingCompiler contains a function to assist with converting types and values to/from CEL-typed values.
type MutatingCompiler interface {
	// CompileMutatingEvaluator is used for the cel expression compilation
	CompileMutatingEvaluator(expression ExpressionAccessor, optionalDecls OptionalVariableDeclarations, envType environment.Type) MutatingEvaluator
}

// MutatingEvaluator contains the result of compiling a CEL expression
// that evaluates to a mutation.
// It expects the inbound object to already have been converted to the version expected
// by the underlying CEL code (which is indicated by the match criteria of a policy definition).
// versionedParams may be nil.
type MutatingEvaluator interface {
	// ForInput converts compiled CEL-typed values into a CEL-typed value representing a mutation.
	// runtimeCELCostBudget was added for testing purpose only. Callers should always use const RuntimeCELCostBudget from k8s.io/apiserver/pkg/apis/cel/config.go as input.
	// If cost budget is calculated, the condition should return the remaining budget.
	ForInput(ctx context.Context, versionedAttr *admission.VersionedAttributes, request *v1.AdmissionRequest, optionalVars OptionalVariableBindings, namespace *corev1.Namespace, runtimeCELCostBudget int64) (EvaluationResult, int64, error)

	// CompilationErrors returns a list of errors from the compilation of the mutatingEvaluator
	CompilationErrors() []error
}
