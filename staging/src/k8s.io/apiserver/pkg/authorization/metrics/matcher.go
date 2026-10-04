/*
Copyright The Kubernetes Authors.

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

package metrics

import (
	"context"
	"time"
)

// MatcherMetrics defines methods for reporting matchCondition metrics
type MatcherMetrics interface {
	// RecordAuthorizationMatchConditionEvaluation records the total time taken to evaluate matchConditions for an Authorize() call to the given authorizer
	RecordAuthorizationMatchConditionEvaluation(ctx context.Context, authorizerType, authorizerName string, elapsed time.Duration)
	// RecordAuthorizationMatchConditionEvaluationFailure increments if any evaluation error was encountered evaluating matchConditions for an Authorize() call to the given authorizer
	RecordAuthorizationMatchConditionEvaluationFailure(ctx context.Context, authorizerType, authorizerName string)
	// RecordAuthorizationMatchConditionExclusion records increments when at least one matchCondition evaluates to false and excludes an Authorize() call to the given authorizer
	RecordAuthorizationMatchConditionExclusion(ctx context.Context, authorizerType, authorizerName string)
}

type NoopMatcherMetrics struct{}

func (NoopMatcherMetrics) RecordAuthorizationMatchConditionEvaluation(ctx context.Context, authorizerType, authorizerName string, elapsed time.Duration) {
}
func (NoopMatcherMetrics) RecordAuthorizationMatchConditionEvaluationFailure(ctx context.Context, authorizerType, authorizerName string) {
}
func (NoopMatcherMetrics) RecordAuthorizationMatchConditionExclusion(ctx context.Context, authorizerType, authorizerName string) {
}
