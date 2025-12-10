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
				Name:           "apiserver_crd_conversion_webhook_duration_seconds",
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

type ConversionWebhookErrorType string

const (
	ConversionWebhookCallFailure                   ConversionWebhookErrorType = "conversion_webhook_call_failure"
	ConversionWebhookMalformedResponseFailure      ConversionWebhookErrorType = "conversion_webhook_malformed_response_failure"
	ConversionWebhookPartialResponseFailure        ConversionWebhookErrorType = "conversion_webhook_partial_response_failure"
	ConversionWebhookInvalidConvertedObjectFailure ConversionWebhookErrorType = "conversion_webhook_invalid_converted_object_failure"
	ConversionWebhookNoObjectsReturnedFailure      ConversionWebhookErrorType = "conversion_webhook_no_objects_returned_failure"
)

var (
	Metrics   = newConversionWebhookMetrics()
	namespace = "apiserver"
)

// ConversionWebhookMetrics instruments webhook conversion with prometheus metrics.
type ConversionWebhookMetrics struct {
	conversionWebhookRequest *metrics.CounterVec
	conversionWebhookLatency *metrics.HistogramVec
}

func newConversionWebhookMetrics() *ConversionWebhookMetrics {
	conversionWebhookRequest := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "conversion_webhook_request_total",
			Namespace:      namespace,
			Help:           "Counter for conversion webhook requests with success/failure and failure error type",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result", "failure_type"})

	conversionWebhookLatency := metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name:      "conversion_webhook_duration_seconds",
			Namespace: namespace,
			Help:      "Conversion webhook request latency",
			// Various buckets from 5 ms to 60 seconds
			Buckets:        []float64{0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 45, 60},
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result", "failure_type"},
	)

	legacyregistry.MustRegister(conversionWebhookRequest)
	legacyregistry.MustRegister(conversionWebhookLatency)

	return &ConversionWebhookMetrics{conversionWebhookRequest: conversionWebhookRequest, conversionWebhookLatency: conversionWebhookLatency}
}

// Observe successful request
func (m *ConversionWebhookMetrics) ObserveConversionWebhookSuccess(ctx context.Context, elapsed time.Duration) {
	result := "success"
	m.conversionWebhookRequest.WithContext(ctx).WithLabelValues(result, "").Inc()
	m.observe(ctx, elapsed, result, "")
}

// Observe failure with failure type
func (m *ConversionWebhookMetrics) ObserveConversionWebhookFailure(ctx context.Context, elapsed time.Duration, errorType ConversionWebhookErrorType) {
	result := "failure"
	m.conversionWebhookRequest.WithContext(ctx).WithLabelValues(result, string(errorType)).Inc()
	m.observe(ctx, elapsed, result, errorType)
}

// Observe latency
func (m *ConversionWebhookMetrics) observe(ctx context.Context, elapsed time.Duration, result string, errorType ConversionWebhookErrorType) {
	elapsedSeconds := elapsed.Seconds()
	m.conversionWebhookLatency.WithContext(ctx).WithLabelValues(result, string(errorType)).Observe(elapsedSeconds)
}
