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

package metrics

import (
	"context"
	"sync"

	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "authorization"
)

var (
	authorizationDecisionsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "decisions_total",
			Help:           "Total number of terminal decisions made by an authorizer split by authorizer type, name, and decision.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"type", "name", "decision"},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(authorizationDecisionsTotal)
	})
}

func ResetMetricsForTest() {
	authorizationDecisionsTotal.Reset()
}

func RecordAuthorizationDecision(authorizerType, authorizerName, decision string) {
	authorizationDecisionsTotal.WithLabelValues(authorizerType, authorizerName, decision).Inc()
}

func InstrumentedAuthorizer(authorizerType string, authorizerName string, delegate authorizer.Authorizer) authorizer.Authorizer {
	RegisterMetrics()
	return &instrumentedAuthorizer{
		authorizerType: string(authorizerType),
		authorizerName: authorizerName,
		delegate:       delegate,
	}
}

type instrumentedAuthorizer struct {
	authorizerType string
	authorizerName string
	delegate       authorizer.Authorizer
}

func (a *instrumentedAuthorizer) Authorize(ctx context.Context, attributes authorizer.Attributes) (authorizer.Decision, string, error) {
	decision, reason, err := a.delegate.Authorize(ctx, attributes)
	switch decision {
	case authorizer.DecisionNoOpinion:
		// non-terminal, not reported
	case authorizer.DecisionAllow:
		// matches SubjectAccessReview status.allowed field name
		RecordAuthorizationDecision(a.authorizerType, a.authorizerName, "allowed")
	case authorizer.DecisionDeny:
		// matches SubjectAccessReview status.denied field name
		RecordAuthorizationDecision(a.authorizerType, a.authorizerName, "denied")
	default:
		RecordAuthorizationDecision(a.authorizerType, a.authorizerName, "unknown")
	}
	return decision, reason, err
}
