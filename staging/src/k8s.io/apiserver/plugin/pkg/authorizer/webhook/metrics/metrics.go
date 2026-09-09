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

package metrics

import (
	"context"
	"sync"

	"k8s.io/apiserver/pkg/authorization/cel"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// AuthorizerMetrics specifies a set of methods that are used to register various metrics for the webhook authorizer
type AuthorizerMetrics interface {
	// Request total and latency metrics
	RequestMetrics
	// Webhook count, latency, and fail open metrics
	WebhookMetrics
	// match condition metrics
	cel.MatcherMetrics
}

type NoopAuthorizerMetrics struct {
	NoopRequestMetrics
	NoopWebhookMetrics
	cel.NoopMatcherMetrics
}

type RequestMetrics interface {
	// RecordRequestTotal increments the total number of requests for the webhook authorizer
	RecordRequestTotal(ctx context.Context, code string)

	// RecordRequestLatency measures request latency in seconds for webhooks. Broken down by status code.
	RecordRequestLatency(ctx context.Context, code string, latency float64)
}

type NoopRequestMetrics struct{}

func (NoopRequestMetrics) RecordRequestTotal(context.Context, string)            {}
func (NoopRequestMetrics) RecordRequestLatency(context.Context, string, float64) {}

type WebhookMetrics interface {
	// RecordWebhookEvaluation increments with each round-trip of a webhook authorizer.
	// result is one of:
	// - canceled: the call invoking the webhook request was canceled
	// - timeout: the webhook request timed out
	// - error: the webhook response completed and was invalid
	// - success: the webhook response completed and was well-formed
	RecordWebhookEvaluation(ctx context.Context, name, result string)
	// RecordWebhookDuration records latency for each round-trip of a webhook authorizer.
	// result is one of:
	// - canceled: the call invoking the webhook request was canceled
	// - timeout: the webhook request timed out
	// - error: the webhook response completed and was invalid
	// - success: the webhook response completed and was well-formed
	RecordWebhookDuration(ctx context.Context, name, result string, duration float64)
	// RecordWebhookFailOpen increments when a webhook timeout or error results in a fail open
	// of a request which has not been canceled.
	// result is one of:
	// - timeout: the webhook request timed out
	// - error: the webhook response completed and was invalid
	RecordWebhookFailOpen(ctx context.Context, name, result string)
}

type NoopWebhookMetrics struct{}

func (NoopWebhookMetrics) RecordWebhookEvaluation(ctx context.Context, name, result string) {}
func (NoopWebhookMetrics) RecordWebhookDuration(ctx context.Context, name, result string, duration float64) {
}
func (NoopWebhookMetrics) RecordWebhookFailOpen(ctx context.Context, name, result string) {}

var registerWebhookMetrics sync.Once

// RegisterMetrics registers authorizer metrics.
func RegisterWebhookMetrics() {
	registerWebhookMetrics.Do(func() {
		legacyregistry.MustRegister(webhookEvaluations)
		legacyregistry.MustRegister(webhookDuration)
		legacyregistry.MustRegister(webhookFailOpen)
	})
}

func ResetMetricsForTest() {
	webhookEvaluations.Reset()
	webhookDuration.Reset()
	webhookFailOpen.Reset()
}

const (
	namespace = "apiserver"
	subsystem = "authorization"
)

var (
	webhookEvaluations = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "webhook_evaluations_total",
			Help:           "Round-trips to authorization webhooks.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"name", "result"},
	)

	webhookDuration = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "webhook_duration_seconds",
			Help:           "Request latency in seconds.",
			Buckets:        compbasemetrics.DefBuckets,
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"name", "result"},
	)

	webhookFailOpen = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "webhook_evaluations_fail_open_total",
			Help:           "NoOpinion results due to webhook timeout or error.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"name", "result"},
	)
)

type webhookMetrics struct{}

func NewWebhookMetrics() WebhookMetrics {
	RegisterWebhookMetrics()
	return webhookMetrics{}
}

func ResetWebhookMetricsForTest() {
	webhookEvaluations.Reset()
	webhookDuration.Reset()
	webhookFailOpen.Reset()
}

func (webhookMetrics) RecordWebhookEvaluation(ctx context.Context, name, result string) {
	webhookEvaluations.WithContext(ctx).WithLabelValues(name, result).Inc()
}
func (webhookMetrics) RecordWebhookDuration(ctx context.Context, name, result string, duration float64) {
	webhookDuration.WithContext(ctx).WithLabelValues(name, result).Observe(duration)
}
func (webhookMetrics) RecordWebhookFailOpen(ctx context.Context, name, result string) {
	webhookFailOpen.WithContext(ctx).WithLabelValues(name, result).Inc()
}
