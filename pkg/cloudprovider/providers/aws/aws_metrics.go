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

import "github.com/prometheus/client_golang/prometheus"

const (
	// OperationAttachDisk measures the time from when we call AttachVolume to when we find, via polling the API, that
	// the disk has been attached
	OperationAttachDisk = "attach"
	OperationDetachDisk = "detach"
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

	awsAttachDetachMetric = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "cloudprovider_aws_attach_detach_duration",
			Help: "Latency of AWS volume to enter the attached or detached state",
		},
		[]string{"operation", "node", "volume"},
	)
	awsAttachDetachErrorMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cloudprovider_aws_attach_detach_errors",
			Help: "AWS attach or detach errors",
		},
		[]string{"operation", "node", "volume"},
	)
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

func recordAWSAttachDetachMetric(actionName string, node string, volume string, timeTaken float64, err error) {
	if err != nil {
		awsAttachDetachErrorMetric.With(prometheus.Labels{"operation": actionName, "node": node, "volume": volume}).Inc()
	} else {
		awsAttachDetachMetric.With(prometheus.Labels{"operation": actionName, "node": node, "volume": volume}).Observe(timeTaken)
	}
}

func registerMetrics() {
	prometheus.MustRegister(awsAPIMetric)
	prometheus.MustRegister(awsAPIErrorMetric)
	prometheus.MustRegister(awsAPIThrottlesMetric)
	prometheus.MustRegister(awsAttachDetachMetric)
	prometheus.MustRegister(awsAttachDetachErrorMetric)
}
