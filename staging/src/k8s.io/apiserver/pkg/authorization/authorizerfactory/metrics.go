/*
Copyright 2021 The Kubernetes Authors.

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

package authorizerfactory

import (
	"context"
	"sync"

	celmetrics "k8s.io/apiserver/pkg/authorization/cel"
	webhookmetrics "k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var registerMetrics sync.Once

// RegisterMetrics registers authorizer metrics.
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(requestTotal)
		legacyregistry.MustRegister(requestLatency)
	})
}

var (
	requestTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_delegated_authz_request_total",
			Help:           "Number of HTTP requests partitioned by status code.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"code"},
	)

	requestLatency = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name:           "apiserver_delegated_authz_request_duration_seconds",
			Help:           "Request latency in seconds. Broken down by status code.",
			Buckets:        []float64{0.25, 0.5, 0.7, 1, 1.5, 3, 5, 10},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"code"},
	)
)

var _ = webhookmetrics.AuthorizerMetrics(delegatingAuthorizerMetrics{})

type delegatingAuthorizerMetrics struct {
	// no-op for webhook metrics for now, delegating authorization reports original total/latency metrics
	webhookmetrics.NoopWebhookMetrics
	// no-op for matchCondition metrics for now, delegating authorization doesn't configure match conditions
	celmetrics.NoopMatcherMetrics
}

func NewDelegatingAuthorizerMetrics() delegatingAuthorizerMetrics {
	RegisterMetrics()
	return delegatingAuthorizerMetrics{}
}

// RecordRequestTotal increments the total number of requests for the delegated authorization.
func (delegatingAuthorizerMetrics) RecordRequestTotal(ctx context.Context, code string) {
	requestTotal.WithContext(ctx).WithLabelValues(code).Add(1)
}

// RecordRequestLatency measures request latency in seconds for the delegated authorization. Broken down by status code.
func (delegatingAuthorizerMetrics) RecordRequestLatency(ctx context.Context, code string, latency float64) {
	requestLatency.WithContext(ctx).WithLabelValues(code).Observe(latency)
}
