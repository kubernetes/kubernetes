/*
Copyright 2024 The Kubernetes Authors.

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

	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/cel/environment"
)

// mutatingCompiler provides a MutatingCompiler implementation.
type mutatingCompiler struct {
	compiler Compiler
}

// CompileMutatingEvaluator compiles a CEL expression for admission plugins and returns an MutatingEvaluator for executing the
// compiled CEL expression.
func (p *mutatingCompiler) CompileMutatingEvaluator(expressionAccessor ExpressionAccessor, options OptionalVariableDeclarations, mode environment.Type) MutatingEvaluator {
	compilationResult := p.compiler.CompileCELExpression(expressionAccessor, options, mode)
	return NewMutatingEvaluator(compilationResult)
}

type mutatingEvaluator struct {
	compilationResult CompilationResult
}

func NewMutatingEvaluator(compilationResult CompilationResult) MutatingEvaluator {
	return &mutatingEvaluator{compilationResult}
}

// ForInput evaluates the compiled CEL expression and returns an evaluation result
// errors per evaluation are returned in the evaluation result
// runtimeCELCostBudget was added for testing purpose only. Callers should always use const RuntimeCELCostBudget from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func (p *mutatingEvaluator) ForInput(ctx context.Context, versionedAttr *admission.VersionedAttributes, request *admissionv1.AdmissionRequest, inputs OptionalVariableBindings, namespace *v1.Namespace, runtimeCELCostBudget int64) (EvaluationResult, int64, error) {
	// if this activation supports composition, we will need the compositionCtx. It may be nil.
	compositionCtx, _ := ctx.(CompositionContext)

	activation, err := newActivation(compositionCtx, versionedAttr, request, inputs, namespace)
	if err != nil {
		return EvaluationResult{}, -1, err
	}
	evaluation, remainingBudget, err := activation.Evaluate(ctx, compositionCtx, p.compilationResult, runtimeCELCostBudget)
	if err != nil {
		return evaluation, -1, err
	}
	return evaluation, remainingBudget, nil

}

// CompilationErrors returns a list of all the errors from the compilation of the mutatingEvaluator
func (p *mutatingEvaluator) CompilationErrors() (compilationErrors []error) {
	if p.compilationResult.Error != nil {
		return []error{p.compilationResult.Error}
	}
	return nil
}
