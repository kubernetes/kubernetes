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

	celgo "github.com/google/cel-go/cel"
	authorizationv1 "k8s.io/api/authorization/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

type CELMatcher struct {
	CompilationResults []CompilationResult
}

// eval evaluates the given SubjectAccessReview against all cel matchCondition expression
func (c *CELMatcher) Eval(ctx context.Context, r *authorizationv1.SubjectAccessReview) (bool, error) {
	var evalErrors []error
	specValObject, err := convertObjectToUnstructured(&r.Spec)
	if err != nil {
		return false, fmt.Errorf("authz celMatcher eval error: convert SubjectAccessReviewSpec object to unstructured failed: %w", err)
	}
	va := map[string]interface{}{
		"request": specValObject,
	}
	for _, compilationResult := range c.CompilationResults {
		evalResult, _, err := compilationResult.Program.ContextEval(ctx, va)
		if err != nil {
			evalErrors = append(evalErrors, fmt.Errorf("cel evaluation error: expression '%v' resulted in error: %w", compilationResult.ExpressionAccessor.GetExpression(), err))
			continue
		}
		if evalResult.Type() != celgo.BoolType {
			evalErrors = append(evalErrors, fmt.Errorf("cel evaluation error: expression '%v' eval result type should be bool but got %W", compilationResult.ExpressionAccessor.GetExpression(), evalResult.Type()))
			continue
		}
		match, ok := evalResult.Value().(bool)
		if !ok {
			evalErrors = append(evalErrors, fmt.Errorf("cel evaluation error: expression '%v' eval result value should be bool but got %W", compilationResult.ExpressionAccessor.GetExpression(), evalResult.Value()))
			continue
		}
		// If at least one matchCondition successfully evaluates to FALSE,
		// return early
		if !match {
			return false, nil
		}
	}
	// if there is any error, return
	if len(evalErrors) > 0 {
		return false, utilerrors.NewAggregate(evalErrors)
	}
	// return ALL matchConditions evaluate to TRUE successfully without error
	return true, nil
}

func convertObjectToUnstructured(obj *authorizationv1.SubjectAccessReviewSpec) (map[string]interface{}, error) {
	if obj == nil {
		return nil, nil
	}
	ret, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return ret, nil
}
