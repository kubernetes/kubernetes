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
	"time"

	"github.com/google/cel-go/common/types/ref"

	v1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
)

var _ ExpressionAccessor = &MatchCondition{}

type ExpressionAccessor interface {
	GetExpression() string
}

// EvaluationResult contains the minimal required fields and metadata of a cel evaluation
type EvaluationResult struct {
	EvalResult         ref.Val
	ExpressionAccessor ExpressionAccessor
	Elapsed            time.Duration
	Error              error
}

// MatchCondition contains the inputs needed to compile, evaluate and match a cel expression
type MatchCondition struct {
	Expression string
}

func (v *MatchCondition) GetExpression() string {
	return v.Expression
}

// FilterCompiler contains a function to assist with converting types and values to/from CEL-typed values.
type FilterCompiler interface {
	// Compile is used for the cel expression compilation
	Compile(expressions []ExpressionAccessor, hasParam bool) Filter
}

// Filter contains a function to evaluate compiled CEL-typed values
// It expects the inbound object to already have been converted to the version expected
// by the underlying CEL code (which is indicated by the match criteria of a policy definition).
// versionedParams may be nil.
type Filter interface {
	// ForInput converts compiled CEL-typed values into evaluated CEL-typed values
	ForInput(versionedAttr *generic.VersionedAttributes, versionedParams runtime.Object, request *v1.AdmissionRequest) ([]EvaluationResult, error)

	// CompilationErrors returns a list of errors from the compilation of the evaluator
	CompilationErrors() []error
}
