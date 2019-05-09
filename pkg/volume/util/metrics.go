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

package util

import (
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	statusSuccess     = "success"
	statusFailUnknown = "fail-unknown"
)

var storageOperationMetric = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "storage_operation_duration_seconds",
		Help:    "Storage operation duration",
		Buckets: []float64{.1, .25, .5, 1, 2.5, 5, 10, 15, 25, 50, 120, 300, 600},
	},
	[]string{"volume_plugin", "operation_name"},
)

var storageOperationErrorMetric = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "storage_operation_errors_total",
		Help: "Storage operation errors",
	},
	[]string{"volume_plugin", "operation_name"},
)

var storageOperationStatusMetric = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "storage_operation_status_count",
		Help: "Storage operation return statuses count",
	},
	[]string{"volume_plugin", "operation_name", "status"},
)

func init() {
	registerMetrics()
}

func registerMetrics() {
	prometheus.MustRegister(storageOperationMetric)
	prometheus.MustRegister(storageOperationErrorMetric)
	prometheus.MustRegister(storageOperationStatusMetric)
}

// OperationCompleteHook returns a hook to call when an operation is completed
func OperationCompleteHook(plugin, operationName string) func(*error) {
	requestTime := time.Now()
	opComplete := func(err *error) {
		timeTaken := time.Since(requestTime).Seconds()
		// Create metric with operation name and plugin name
		status := statusSuccess
		if *err != nil {
			// TODO: Establish well-known error codes to be able to distinguish
			// user configuration errors from system errors.
			status = statusFailUnknown
			storageOperationErrorMetric.WithLabelValues(plugin, operationName).Inc()
		} else {
			storageOperationMetric.WithLabelValues(plugin, operationName).Observe(timeTaken)
		}
		storageOperationStatusMetric.WithLabelValues(plugin, operationName, status).Inc()
	}
	return opComplete
}

// GetFullQualifiedPluginNameForVolume returns full qualified plugin name for
// given volume. For CSI plugin, it appends plugin driver name at the end of
// plugin name, e.g. kubernetes.io/csi:csi-hostpath. It helps to distinguish
// between metrics emitted for CSI volumes which may be handled by different
// CSI plugin drivers.
func GetFullQualifiedPluginNameForVolume(pluginName string, spec *volume.Spec) string {
	if spec != nil && spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CSI != nil {
		return fmt.Sprintf("%s:%s", pluginName, spec.PersistentVolume.Spec.CSI.Driver)
	}
	return pluginName
}
