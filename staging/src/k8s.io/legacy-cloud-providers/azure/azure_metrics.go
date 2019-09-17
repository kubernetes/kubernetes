// +build !providerless

/*
Copyright 2018 The Kubernetes Authors.

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

package azure

import (
	"strings"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

type apiCallMetrics struct {
	latency          *metrics.HistogramVec
	errors           *metrics.CounterVec
	rateLimitedCount *metrics.CounterVec
}

var (
	metricLabels = []string{
		"request",         // API function that is being invoked
		"resource_group",  // Resource group of the resource being monitored
		"subscription_id", // Subscription ID of the resource being monitored
		"source",          // Oeration source(optional)
	}

	apiMetrics = registerAPIMetrics(metricLabels...)
)

type metricContext struct {
	start      time.Time
	attributes []string
}

func newMetricContext(prefix, request, resourceGroup, subscriptionID, source string) *metricContext {
	return &metricContext{
		start:      time.Now(),
		attributes: []string{prefix + "_" + request, strings.ToLower(resourceGroup), subscriptionID, source},
	}
}
func (mc *metricContext) RateLimitedCount() {
	apiMetrics.rateLimitedCount.WithLabelValues(mc.attributes...).Inc()
}

func (mc *metricContext) Observe(err error) error {
	apiMetrics.latency.WithLabelValues(mc.attributes...).Observe(
		time.Since(mc.start).Seconds())
	if err != nil {
		apiMetrics.errors.WithLabelValues(mc.attributes...).Inc()
	}

	return err
}

func registerAPIMetrics(attributes ...string) *apiCallMetrics {
	metrics := &apiCallMetrics{
		latency: metrics.NewHistogramVec(
			&metrics.HistogramOpts{
				Name:           "cloudprovider_azure_api_request_duration_seconds",
				Help:           "Latency of an Azure API call",
				StabilityLevel: metrics.ALPHA,
			},
			attributes,
		),
		errors: metrics.NewCounterVec(
			&metrics.CounterOpts{
				Name:           "cloudprovider_azure_api_request_errors",
				Help:           "Number of errors for an Azure API call",
				StabilityLevel: metrics.ALPHA,
			},
			attributes,
		),
		rateLimitedCount: metrics.NewCounterVec(
			&metrics.CounterOpts{
				Name:           "cloudprovider_azure_api_request_ratelimited_count",
				Help:           "Number of rate limited Azure API calls",
				StabilityLevel: metrics.ALPHA,
			},
			attributes,
		),
	}

	legacyregistry.MustRegister(metrics.latency)
	legacyregistry.MustRegister(metrics.errors)
	legacyregistry.MustRegister(metrics.rateLimitedCount)

	return metrics
}
