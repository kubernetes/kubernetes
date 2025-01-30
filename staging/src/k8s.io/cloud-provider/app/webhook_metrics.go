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

package app

import (
	"context"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	// subSystemName is the name of this subsystem name used for prometheus metrics.
	subSystemName = "cloud_provider_webhook"
)

type registerables []metrics.Registerable

// init registers all metrics
func init() {
	for _, metric := range toRegister {
		legacyregistry.MustRegister(metric)
	}
}

var (
	requestTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "request_total",
			Subsystem:      subSystemName,
			Help:           "Number of HTTP requests partitioned by status code.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code", "webhook"},
	)

	requestLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name:           "request_duration_seconds",
			Subsystem:      subSystemName,
			Help:           "Request latency in seconds. Broken down by status code.",
			Buckets:        []float64{0.25, 0.5, 0.7, 1, 1.5, 3, 5, 10},
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code", "webhook"},
	)

	toRegister = registerables{
		requestTotal,
		requestLatency,
	}
)

// RecordRequestTotal increments the total number of requests for the webhook.
func recordRequestTotal(ctx context.Context, code string, webhookName string) {
	requestTotal.WithContext(ctx).With(map[string]string{"code": code, "webhook": webhookName}).Add(1)
}

// RecordRequestLatency measures request latency in seconds for the delegated authorization. Broken down by status code.
func recordRequestLatency(ctx context.Context, code string, webhookName string, latency float64) {
	requestLatency.WithContext(ctx).With(map[string]string{"code": code, "webhook": webhookName}).Observe(latency)
}
