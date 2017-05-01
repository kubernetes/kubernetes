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

var awsApiMetric = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Name: "cloudprovider_aws_api_request_duration_seconds",
		Help: "Latency of aws api call",
	},
	[]string{"request"},
)

var awsApiErrorMetric = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "cloudprovider_aws_api_request_errors",
		Help: "AWS Api errors",
	},
	[]string{"request"},
)

func registerMetrics() {
	prometheus.MustRegister(awsApiMetric)
	prometheus.MustRegister(awsApiErrorMetric)
}
