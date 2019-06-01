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

package operationexecutor

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	statusSuccess     = "success"
	statusFailUnknown = "fail-unknown"
)

var pluginOperationMetric = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "plugin_operation_duration_seconds",
		Help:    "plugin operation duration",
		Buckets: []float64{.1, .25, .5, 1, 2.5, 5, 10, 15, 25, 50, 120, 300, 600},
	},
	[]string{"socket_path", "operation_name"},
)

var pluginOperationErrorMetric = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "plugin_operation_errors_total",
		Help: "plugin operation errors",
	},
	[]string{"socket_path", "operation_name"},
)

var pluginOperationStatusMetric = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "plugin_operation_status_count",
		Help: "plugin operation return statuses count",
	},
	[]string{"socket_path", "operation_name", "status"},
)

func init() {
	registerMetrics()
}

func registerMetrics() {
	prometheus.MustRegister(pluginOperationMetric)
	prometheus.MustRegister(pluginOperationErrorMetric)
	prometheus.MustRegister(pluginOperationStatusMetric)
}

// operationCompleteHook returns a hook to call when an operation is completed
func operationCompleteHook(plugin, operationName string) func(*error) {
	requestTime := time.Now()
	opComplete := func(err *error) {
		timeTaken := time.Since(requestTime).Seconds()
		// Create metric with operation name and plugin name
		status := statusSuccess
		if *err != nil {
			// TODO: Establish well-known error codes to be able to distinguish
			// user configuration errors from system errors.
			status = statusFailUnknown
			pluginOperationErrorMetric.WithLabelValues(plugin, operationName).Inc()
		} else {
			pluginOperationMetric.WithLabelValues(plugin, operationName).Observe(timeTaken)
		}
		pluginOperationStatusMetric.WithLabelValues(plugin, operationName, status).Inc()
	}
	return opComplete
}
