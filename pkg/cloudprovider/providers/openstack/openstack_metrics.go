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

package openstack

import "github.com/prometheus/client_golang/prometheus"

const (
	OpenstackSubsystem         = "openstack"
	OpenstackOperationKey      = "cloudprovider_openstack_api_request_duration_seconds"
	OpenstackOperationErrorKey = "cloudprovider_openstack_api_request_errors"
)

var (
	OpenstackOperationsLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: OpenstackSubsystem,
			Name:      OpenstackOperationKey,
			Help:      "Latency of openstack api call",
		},
		[]string{"request"},
	)

	OpenstackApiRequestErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: OpenstackSubsystem,
			Name:      OpenstackOperationErrorKey,
			Help:      "Cumulative number of openstack Api call errors",
		},
		[]string{"request"},
	)
)

func RegisterMetrics() {
	prometheus.MustRegister(OpenstackOperationsLatency)
	prometheus.MustRegister(OpenstackApiRequestErrors)
}
