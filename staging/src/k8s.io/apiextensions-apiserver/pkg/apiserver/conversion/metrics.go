/*
Copyright 2019 The Kubernetes Authors.

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

package conversion

import (
	"context"
	"strconv"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	latencyBuckets = metrics.ExponentialBuckets(0.001, 2, 15)
)

// converterMetricFactory holds metrics for all CRD converters
type converterMetricFactory struct {
	// A map from a converter name to it's metric. Allows the converterMetric to be created
	// again with the same metric for a specific converter (e.g. 'webhook').
	durations   map[string]*metrics.HistogramVec
	factoryLock sync.Mutex
}

func newConverterMetricFactory() *converterMetricFactory {
	return &converterMetricFactory{durations: map[string]*metrics.HistogramVec{}, factoryLock: sync.Mutex{}}
}

var _ crConverterInterface = &converterMetric{}

type converterMetric struct {
	delegate  crConverterInterface
	latencies *metrics.HistogramVec
	crdName   string
}

func (c *converterMetricFactory) addMetrics(crdName string, converter crConverterInterface) (crConverterInterface, error) {
	c.factoryLock.Lock()
	defer c.factoryLock.Unlock()
	metric, exists := c.durations["webhook"]
	if !exists {
		metric = metrics.NewHistogramVec(
			&metrics.HistogramOpts{
				Name:           "apiserver_crd_webhook_conversion_duration_seconds",
				Help:           "CRD webhook conversion duration in seconds",
				Buckets:        latencyBuckets,
				StabilityLevel: metrics.ALPHA,
			},
			[]string{"crd_name", "from_version", "to_version", "succeeded"})
		err := legacyregistry.Register(metric)
		if err != nil {
			return nil, err
		}
		c.durations["webhook"] = metric
	}
	return &converterMetric{latencies: metric, delegate: converter, crdName: crdName}, nil
}

func (m *converterMetric) Convert(in runtime.Object, targetGV schema.GroupVersion) (runtime.Object, error) {
	start := time.Now()
	obj, err := m.delegate.Convert(in, targetGV)
	fromVersion := in.GetObjectKind().GroupVersionKind().Version
	toVersion := targetGV.Version

	// only record this observation if the version is different
	if fromVersion != toVersion {
		m.latencies.WithLabelValues(
			m.crdName, fromVersion, toVersion, strconv.FormatBool(err == nil)).Observe(time.Since(start).Seconds())
	}
	return obj, err
}

type WebhookConversionErrorType string

const (
	WebhookConversionCallFailure                   WebhookConversionErrorType = "webhook_conversion_call_failure"
	WebhookConversionMalformedResponseFailure      WebhookConversionErrorType = "webhook_conversion_malformed_response_failure"
	WebhookConversionPartialResponseFailure        WebhookConversionErrorType = "webhook_conversion_partial_response_failure"
	WebhookConversionInvalidConvertedObjectFailure WebhookConversionErrorType = "webhook_conversion_invalid_converted_object_failure"
	WebhookConversionNoObjectsReturnedFailure      WebhookConversionErrorType = "webhook_conversion_no_objects_returned_failure"
)

var (
	Metrics = newWebhookConversionMetrics()
)

// WebhookConversionMetrics instruments webhook conversion with prometheus metrics.
type WebhookConversionMetrics struct {
	webhookConversionRequest *metrics.CounterVec
	webhookConversionLatency *metrics.HistogramVec
}

func newWebhookConversionMetrics() *WebhookConversionMetrics {
	webhookConversionRequest := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "webhook_conversion_requests",
			Help:           "Counter for webhook conversion requests with success/failure and failure error type",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result", "failure_type"})

	webhookConversionLatency := metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name:           "webhook_conversion_duration_seconds",
			Help:           "Webhook conversion request latency",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result", "failure_type"},
	)

	legacyregistry.MustRegister(webhookConversionRequest)
	legacyregistry.MustRegister(webhookConversionLatency)

	return &WebhookConversionMetrics{webhookConversionRequest: webhookConversionRequest, webhookConversionLatency: webhookConversionLatency}
}

// Observe successful request
func (m *WebhookConversionMetrics) ObserveWebhookConversionSuccess(ctx context.Context, elapsed time.Duration) {
	result := "success"
	m.webhookConversionRequest.WithContext(ctx).WithLabelValues(result, "").Inc()
	m.observe(ctx, elapsed, result, "")
}

// Observe failure with failure type
func (m *WebhookConversionMetrics) ObserveWebhookConversionFailure(ctx context.Context, elapsed time.Duration, errorType WebhookConversionErrorType) {
	result := "failure"
	m.webhookConversionRequest.WithContext(ctx).WithLabelValues(result, string(errorType)).Inc()
	m.observe(ctx, elapsed, result, errorType)
}

// Observe latency
func (m *WebhookConversionMetrics) observe(ctx context.Context, elapsed time.Duration, result string, errorType WebhookConversionErrorType) {
	elapsedSeconds := elapsed.Seconds()
	m.webhookConversionLatency.WithContext(ctx).WithLabelValues(result, string(errorType)).Observe(elapsedSeconds)
}
