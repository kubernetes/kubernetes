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

// evaluatorCompiler implement the interface PatchCompiler.
type evaluatorCompiler struct {
	compiler Compiler
}

// NewEvaluatorCompiler creates a CEL compiler to compile CEL expressions for admission plugins.
func NewEvaluatorCompiler(env *environment.EnvSet) EvaluatorCompiler {
	return &evaluatorCompiler{compiler: NewCompiler(env)}
}

// CompileEvaluator compiles a CEL expression for admission plugins and returns an Evaluator for executing the
// compiled CEL expression.
func (p *evaluatorCompiler) CompileEvaluator(expressionAccessor ExpressionAccessor, options OptionalVariableDeclarations, mode environment.Type) Evaluator {
	compilationResult := p.compiler.CompileCELExpression(expressionAccessor, options, mode)
	return NewEvaluator(compilationResult)
}

type evaluator struct {
	compilationResult CompilationResult
}

func NewEvaluator(compilationResult CompilationResult) Evaluator {
	return &evaluator{compilationResult}
}

// ForInput evaluates the compiled CEL expression converting it to an Evaluation
// errors per evaluation are returned on the Evaluation object
// runtimeCELCostBudget was added for testing purpose only. Callers should always use const RuntimeCELCostBudget from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func (p *evaluator) ForInput(ctx context.Context, versionedAttr *admission.VersionedAttributes, request *admissionv1.AdmissionRequest, inputs OptionalVariableBindings, namespace *v1.Namespace, runtimeCELCostBudget int64) (EvaluationResult, int64, error) {
	activation, err := newActivation(ctx, versionedAttr, request, inputs, namespace)
	if err != nil {
		return EvaluationResult{}, -1, err
	}
	evaluation, remainingBudget, err := evaluateWithActivation(ctx, activation, p.compilationResult, runtimeCELCostBudget)
	if err != nil {
		return evaluation, -1, err
	}
	return evaluation, remainingBudget, nil

}

// CompilationErrors returns a list of all the errors from the compilation of the evaluator
func (p *evaluator) CompilationErrors() (compilationErrors []error) {
	if p.compilationResult.Error != nil {
		return []error{p.compilationResult.Error}
	}
	return nil
}
