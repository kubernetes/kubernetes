/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

var _ ClaimsMapper = &mapper{}
var _ UserMapper = &mapper{}

// mapper implements the ClaimsMapper and UserMapper interface.
type mapper struct {
	compilationResults []CompilationResult
}

// CELMapper is a struct that holds the compiled expressions for
// username, groups, uid, extra, claimValidation and userValidation
type CELMapper struct {
	Username             ClaimsMapper
	Groups               ClaimsMapper
	UID                  ClaimsMapper
	Extra                ClaimsMapper
	ClaimValidationRules ClaimsMapper
	UserValidationRules  UserMapper
}

// NewClaimsMapper returns a new ClaimsMapper.
func NewClaimsMapper(compilationResults []CompilationResult) ClaimsMapper {
	return &mapper{
		compilationResults: compilationResults,
	}
}

// NewUserMapper returns a new UserMapper.
func NewUserMapper(compilationResults []CompilationResult) UserMapper {
	return &mapper{
		compilationResults: compilationResults,
	}
}

// EvalClaimMapping evaluates the given claim mapping expression and returns a EvaluationResult.
func (m *mapper) EvalClaimMapping(ctx context.Context, claims *unstructured.Unstructured) (EvaluationResult, error) {
	results, err := m.eval(ctx, map[string]interface{}{claimsVarName: claims.Object})
	if err != nil {
		return EvaluationResult{}, err
	}
	if len(results) != 1 {
		return EvaluationResult{}, fmt.Errorf("expected 1 evaluation result, got %d", len(results))
	}
	return results[0], nil
}

// EvalClaimMappings evaluates the given expressions and returns a list of EvaluationResult.
func (m *mapper) EvalClaimMappings(ctx context.Context, claims *unstructured.Unstructured) ([]EvaluationResult, error) {
	return m.eval(ctx, map[string]interface{}{claimsVarName: claims.Object})
}

// EvalUser evaluates the given user expressions and returns a list of EvaluationResult.
func (m *mapper) EvalUser(ctx context.Context, userInfo *unstructured.Unstructured) ([]EvaluationResult, error) {
	return m.eval(ctx, map[string]interface{}{userVarName: userInfo.Object})
}

func (m *mapper) eval(ctx context.Context, input map[string]interface{}) ([]EvaluationResult, error) {
	evaluations := make([]EvaluationResult, len(m.compilationResults))

	for i, compilationResult := range m.compilationResults {
		var evaluation = &evaluations[i]
		evaluation.ExpressionAccessor = compilationResult.ExpressionAccessor

		evalResult, _, err := compilationResult.Program.ContextEval(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("expression '%s' resulted in error: %w", compilationResult.ExpressionAccessor.GetExpression(), err)
		}

		evaluation.EvalResult = evalResult
	}

	return evaluations, nil
}
