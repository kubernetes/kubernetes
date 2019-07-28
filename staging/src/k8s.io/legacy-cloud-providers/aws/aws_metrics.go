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

	"github.com/prometheus/client_golang/prometheus"
)

var (
	awsAPIMetric = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "cloudprovider_aws_api_request_duration_seconds",
			Help: "Latency of AWS API calls",
		},
		[]string{"request"})

	awsAPIErrorMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cloudprovider_aws_api_request_errors",
			Help: "AWS API errors",
		},
		[]string{"request"})

	awsAPIThrottlesMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cloudprovider_aws_api_throttled_requests_total",
			Help: "AWS API throttled requests",
		},
		[]string{"operation_name"})
)

func recordAWSMetric(actionName string, timeTaken float64, err error) {
	if err != nil {
		awsAPIErrorMetric.With(prometheus.Labels{"request": actionName}).Inc()
	} else {
		awsAPIMetric.With(prometheus.Labels{"request": actionName}).Observe(timeTaken)
	}
}

func recordAWSThrottlesMetric(operation string) {
	awsAPIThrottlesMetric.With(prometheus.Labels{"operation_name": operation}).Inc()
}

var registerOnce sync.Once

func registerMetrics() {
	registerOnce.Do(func() {
		prometheus.MustRegister(awsAPIMetric)
		prometheus.MustRegister(awsAPIErrorMetric)
		prometheus.MustRegister(awsAPIThrottlesMetric)
	})
}
