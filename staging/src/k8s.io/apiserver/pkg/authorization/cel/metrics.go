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
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
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

type matcherMetrics struct{}

func NewMatcherMetrics() MatcherMetrics {
	RegisterMetrics()
	return matcherMetrics{}
}

const (
	namespace = "apiserver"
	subsystem = "authorization"
)

var (
	authorizationMatchConditionEvaluationErrorsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "match_condition_evaluation_errors_total",
			Help:           "Total number of errors when an authorization webhook encounters a match condition error split by authorizer type and name.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"type", "name"},
	)
	authorizationMatchConditionExclusionsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "match_condition_exclusions_total",
			Help:           "Total number of exclusions when an authorization webhook is skipped because match conditions exclude it.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"type", "name"},
	)
	authorizationMatchConditionEvaluationSeconds = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "match_condition_evaluation_seconds",
			Help:           "Authorization match condition evaluation time in seconds, split by authorizer type and name.",
			Buckets:        []float64{0.001, 0.005, 0.01, 0.025, 0.1, 0.2, 0.25},
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"type", "name"},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(authorizationMatchConditionEvaluationErrorsTotal)
		legacyregistry.MustRegister(authorizationMatchConditionExclusionsTotal)
		legacyregistry.MustRegister(authorizationMatchConditionEvaluationSeconds)
	})
}

func ResetMetricsForTest() {
	authorizationMatchConditionEvaluationErrorsTotal.Reset()
	authorizationMatchConditionExclusionsTotal.Reset()
	authorizationMatchConditionEvaluationSeconds.Reset()
}

func (matcherMetrics) RecordAuthorizationMatchConditionEvaluationFailure(ctx context.Context, authorizerType, authorizerName string) {
	authorizationMatchConditionEvaluationErrorsTotal.WithContext(ctx).WithLabelValues(authorizerType, authorizerName).Inc()
}

func (matcherMetrics) RecordAuthorizationMatchConditionExclusion(ctx context.Context, authorizerType, authorizerName string) {
	authorizationMatchConditionExclusionsTotal.WithContext(ctx).WithLabelValues(authorizerType, authorizerName).Inc()
}

func (matcherMetrics) RecordAuthorizationMatchConditionEvaluation(ctx context.Context, authorizerType, authorizerName string, elapsed time.Duration) {
	elapsedSeconds := elapsed.Seconds()
	authorizationMatchConditionEvaluationSeconds.WithContext(ctx).WithLabelValues(authorizerType, authorizerName).Observe(elapsedSeconds)
}
