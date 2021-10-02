//go:build !providerless
// +build !providerless

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

package aws

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	awsAPIMetric = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Name:           "cloudprovider_aws_api_request_duration_seconds",
			Help:           "Latency of AWS API calls",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"request"})

	awsAPIErrorMetric = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "cloudprovider_aws_api_request_errors",
			Help:           "AWS API errors",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"request"})

	awsAPIThrottlesMetric = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "cloudprovider_aws_api_throttled_requests_total",
			Help:           "AWS API throttled requests",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_name"})
)

func recordAWSMetric(actionName string, timeTaken float64, err error) {
	if err != nil {
		awsAPIErrorMetric.With(metrics.Labels{"request": actionName}).Inc()
	} else {
		awsAPIMetric.With(metrics.Labels{"request": actionName}).Observe(timeTaken)
	}
}

func recordAWSThrottlesMetric(operation string) {
	awsAPIThrottlesMetric.With(metrics.Labels{"operation_name": operation}).Inc()
}

var registerOnce sync.Once

func registerMetrics() {
	registerOnce.Do(func() {
		legacyregistry.MustRegister(awsAPIMetric)
		legacyregistry.MustRegister(awsAPIErrorMetric)
		legacyregistry.MustRegister(awsAPIThrottlesMetric)
	})
}
