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
	"time"

	celgo "github.com/google/cel-go/cel"

	authorizationv1 "k8s.io/api/authorization/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

type CELMatcher struct {
	CompilationResults []CompilationResult

	// These track if any expressions use fieldSelector and labelSelector,
	// so construction of data passed to the CEL expression can be optimized if those fields are unused.
	UsesLabelSelector bool
	UsesFieldSelector bool

	// These are optional fields which can be populated if metrics reporting is desired
	Metrics        MatcherMetrics
	AuthorizerType string
	AuthorizerName string
}

// eval evaluates the given SubjectAccessReview against all cel matchCondition expression
func (c *CELMatcher) Eval(ctx context.Context, r *authorizationv1.SubjectAccessReview) (bool, error) {
	var evalErrors []error

	metrics := c.Metrics
	if metrics == nil {
		metrics = NoopMatcherMetrics{}
	}
	start := time.Now()
	defer func() {
		metrics.RecordAuthorizationMatchConditionEvaluation(ctx, c.AuthorizerType, c.AuthorizerName, time.Since(start))
		if len(evalErrors) > 0 {
			metrics.RecordAuthorizationMatchConditionEvaluationFailure(ctx, c.AuthorizerType, c.AuthorizerName)
		}
	}()

	va := map[string]interface{}{
		"request": convertObjectToUnstructured(&r.Spec, c.UsesFieldSelector, c.UsesLabelSelector),
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
			metrics.RecordAuthorizationMatchConditionExclusion(ctx, c.AuthorizerType, c.AuthorizerName)
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
