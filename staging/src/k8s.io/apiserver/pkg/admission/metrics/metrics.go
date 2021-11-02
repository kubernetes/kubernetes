/*
Copyright 2017 The Kubernetes Authors.

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
	"strconv"
	"time"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// WebhookRejectionErrorType defines different error types that happen in a webhook rejection.
type WebhookRejectionErrorType string

const (
	namespace = "apiserver"
	subsystem = "admission"

	// WebhookRejectionCallingWebhookError identifies a calling webhook error which causes
	// a webhook admission to reject a request
	WebhookRejectionCallingWebhookError WebhookRejectionErrorType = "calling_webhook_error"
	// WebhookRejectionAPIServerInternalError identifies an apiserver internal error which
	// causes a webhook admission to reject a request
	WebhookRejectionAPIServerInternalError WebhookRejectionErrorType = "apiserver_internal_error"
	// WebhookRejectionNoError identifies a webhook properly rejected a request
	WebhookRejectionNoError WebhookRejectionErrorType = "no_error"
)

var (
	latencySummaryMaxAge = 5 * time.Hour

	// Metrics provides access to all admission metrics.
	Metrics = newAdmissionMetrics()
)

// ObserverFunc is a func that emits metrics.
type ObserverFunc func(ctx context.Context, elapsed time.Duration, rejected bool, attr admission.Attributes, stepType string, extraLabels ...string)

const (
	stepValidate = "validate"
	stepAdmit    = "admit"
)

// WithControllerMetrics is a decorator for named admission handlers.
func WithControllerMetrics(i admission.Interface, name string) admission.Interface {
	return WithMetrics(i, Metrics.ObserveAdmissionController, name)
}

// WithStepMetrics is a decorator for a whole admission phase, i.e. admit or validation.admission step.
func WithStepMetrics(i admission.Interface) admission.Interface {
	return WithMetrics(i, Metrics.ObserveAdmissionStep)
}

// WithMetrics is a decorator for admission handlers with a generic observer func.
func WithMetrics(i admission.Interface, observer ObserverFunc, extraLabels ...string) admission.Interface {
	return &pluginHandlerWithMetrics{
		Interface:   i,
		observer:    observer,
		extraLabels: extraLabels,
	}
}

// pluginHandlerWithMetrics decorates a admission handler with metrics.
type pluginHandlerWithMetrics struct {
	admission.Interface
	observer    ObserverFunc
	extraLabels []string
}

// Admit performs a mutating admission control check and emit metrics.
func (p pluginHandlerWithMetrics) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	mutatingHandler, ok := p.Interface.(admission.MutationInterface)
	if !ok {
		return nil
	}

	start := time.Now()
	err := mutatingHandler.Admit(ctx, a, o)
	p.observer(ctx, time.Since(start), err != nil, a, stepAdmit, p.extraLabels...)
	return err
}

// Validate performs a non-mutating admission control check and emits metrics.
func (p pluginHandlerWithMetrics) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	validatingHandler, ok := p.Interface.(admission.ValidationInterface)
	if !ok {
		return nil
	}

	start := time.Now()
	err := validatingHandler.Validate(ctx, a, o)
	p.observer(ctx, time.Since(start), err != nil, a, stepValidate, p.extraLabels...)
	return err
}

// AdmissionMetrics instruments admission with prometheus metrics.
type AdmissionMetrics struct {
	step             *metricSet
	controller       *metricSet
	webhook          *metricSet
	webhookRejection *metrics.CounterVec
	webhookRequest   *metrics.CounterVec
}

// newAdmissionMetrics create a new AdmissionMetrics, configured with default metric names.
func newAdmissionMetrics() *AdmissionMetrics {
	// Admission metrics for a step of the admission flow. The entire admission flow is broken down into a series of steps
	// Each step is identified by a distinct type label value.
	// Use buckets ranging from 5 ms to 2.5 seconds.
	step := &metricSet{
		latencies: metrics.NewHistogramVec(
			&metrics.HistogramOpts{
				Namespace:      namespace,
				Subsystem:      subsystem,
				Name:           "step_admission_duration_seconds",
				Help:           "Admission sub-step latency histogram in seconds, broken out for each operation and API resource and step type (validate or admit).",
				Buckets:        []float64{0.005, 0.025, 0.1, 0.5, 1.0, 2.5},
				StabilityLevel: metrics.STABLE,
			},
			[]string{"type", "operation", "rejected"},
		),

		latenciesSummary: metrics.NewSummaryVec(
			&metrics.SummaryOpts{
				Namespace:      namespace,
				Subsystem:      subsystem,
				Name:           "step_admission_duration_seconds_summary",
				Help:           "Admission sub-step latency summary in seconds, broken out for each operation and API resource and step type (validate or admit).",
				MaxAge:         latencySummaryMaxAge,
				StabilityLevel: metrics.ALPHA,
			},
			[]string{"type", "operation", "rejected"},
		),
	}

	// Built-in admission controller metrics. Each admission controller is identified by name.
	// Use buckets ranging from 5 ms to 2.5 seconds.
	controller := &metricSet{
		latencies: metrics.NewHistogramVec(
			&metrics.HistogramOpts{
				Namespace:      namespace,
				Subsystem:      subsystem,
				Name:           "controller_admission_duration_seconds",
				Help:           "Admission controller latency histogram in seconds, identified by name and broken out for each operation and API resource and type (validate or admit).",
				Buckets:        []float64{0.005, 0.025, 0.1, 0.5, 1.0, 2.5},
				StabilityLevel: metrics.STABLE,
			},
			[]string{"name", "type", "operation", "rejected"},
		),

		latenciesSummary: nil,
	}

	// Admission webhook metrics. Each webhook is identified by name.
	// Use buckets ranging from 5 ms to 2.5 seconds (admission webhooks timeout at 30 seconds by default).
	webhook := &metricSet{
		latencies: metrics.NewHistogramVec(
			&metrics.HistogramOpts{
				Namespace:      namespace,
				Subsystem:      subsystem,
				Name:           "webhook_admission_duration_seconds",
				Help:           "Admission webhook latency histogram in seconds, identified by name and broken out for each operation and API resource and type (validate or admit).",
				Buckets:        []float64{0.005, 0.025, 0.1, 0.5, 1.0, 2.5},
				StabilityLevel: metrics.STABLE,
			},
			[]string{"name", "type", "operation", "rejected"},
		),

		latenciesSummary: nil,
	}

	webhookRejection := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "webhook_rejection_count",
			Help:           "Admission webhook rejection count, identified by name and broken out for each admission type (validating or admit) and operation. Additional labels specify an error type (calling_webhook_error or apiserver_internal_error if an error occurred; no_error otherwise) and optionally a non-zero rejection code if the webhook rejects the request with an HTTP status code (honored by the apiserver when the code is greater or equal to 400). Codes greater than 600 are truncated to 600, to keep the metrics cardinality bounded.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "type", "operation", "error_type", "rejection_code"})

	webhookRequest := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "webhook_request_total",
			Help:           "Admission webhook request total, identified by name and broken out for each admission type (validating or mutating) and operation. Additional labels specify whether the request was rejected or not and an HTTP status code. Codes greater than 600 are truncated to 600, to keep the metrics cardinality bounded.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "type", "operation", "code", "rejected"})

	step.mustRegister()
	controller.mustRegister()
	webhook.mustRegister()
	legacyregistry.MustRegister(webhookRejection)
	legacyregistry.MustRegister(webhookRequest)
	return &AdmissionMetrics{step: step, controller: controller, webhook: webhook, webhookRejection: webhookRejection, webhookRequest: webhookRequest}
}

func (m *AdmissionMetrics) reset() {
	m.step.reset()
	m.controller.reset()
	m.webhook.reset()
}

// ObserveAdmissionStep records admission related metrics for a admission step, identified by step type.
func (m *AdmissionMetrics) ObserveAdmissionStep(ctx context.Context, elapsed time.Duration, rejected bool, attr admission.Attributes, stepType string, extraLabels ...string) {
	m.step.observe(ctx, elapsed, append(extraLabels, stepType, string(attr.GetOperation()), strconv.FormatBool(rejected))...)
}

// ObserveAdmissionController records admission related metrics for a built-in admission controller, identified by it's plugin handler name.
func (m *AdmissionMetrics) ObserveAdmissionController(ctx context.Context, elapsed time.Duration, rejected bool, attr admission.Attributes, stepType string, extraLabels ...string) {
	m.controller.observe(ctx, elapsed, append(extraLabels, stepType, string(attr.GetOperation()), strconv.FormatBool(rejected))...)
}

// ObserveWebhook records admission related metrics for a admission webhook.
func (m *AdmissionMetrics) ObserveWebhook(ctx context.Context, name string, elapsed time.Duration, rejected bool, attr admission.Attributes, stepType string, code int) {
	// We truncate codes greater than 600 to keep the cardinality bounded.
	if code > 600 {
		code = 600
	}
	m.webhookRequest.WithContext(ctx).WithLabelValues(name, stepType, string(attr.GetOperation()), strconv.Itoa(code), strconv.FormatBool(rejected)).Inc()
	m.webhook.observe(ctx, elapsed, name, stepType, string(attr.GetOperation()), strconv.FormatBool(rejected))
}

// ObserveWebhookRejection records admission related metrics for an admission webhook rejection.
func (m *AdmissionMetrics) ObserveWebhookRejection(ctx context.Context, name, stepType, operation string, errorType WebhookRejectionErrorType, rejectionCode int) {
	// We truncate codes greater than 600 to keep the cardinality bounded.
	// This should be rarely done by a malfunctioning webhook server.
	if rejectionCode > 600 {
		rejectionCode = 600
	}
	m.webhookRejection.WithContext(ctx).WithLabelValues(name, stepType, operation, string(errorType), strconv.Itoa(rejectionCode)).Inc()
}

type metricSet struct {
	latencies        *metrics.HistogramVec
	latenciesSummary *metrics.SummaryVec
}

// MustRegister registers all the prometheus metrics in the metricSet.
func (m *metricSet) mustRegister() {
	legacyregistry.MustRegister(m.latencies)
	if m.latenciesSummary != nil {
		legacyregistry.MustRegister(m.latenciesSummary)
	}
}

// Reset resets all the prometheus metrics in the metricSet.
func (m *metricSet) reset() {
	m.latencies.Reset()
	if m.latenciesSummary != nil {
		m.latenciesSummary.Reset()
	}
}

// Observe records an observed admission event to all metrics in the metricSet.
func (m *metricSet) observe(ctx context.Context, elapsed time.Duration, labels ...string) {
	elapsedSeconds := elapsed.Seconds()
	m.latencies.WithContext(ctx).WithLabelValues(labels...).Observe(elapsedSeconds)
	if m.latenciesSummary != nil {
		m.latenciesSummary.WithContext(ctx).WithLabelValues(labels...).Observe(elapsedSeconds)
	}
}
