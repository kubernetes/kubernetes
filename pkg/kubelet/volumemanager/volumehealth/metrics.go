/*
Copyright The Kubernetes Authors.

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

package volumehealth

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	registerMetrics sync.Once

	csiNodeStorageHealthStatus = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Name:           "csi_node_storage_health_status",
			Help:           "CSI node storage backend health conditions currently reported (1 = present).",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"driver_name", "status", "reason"},
	)
)

func registerHealthMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(csiNodeStorageHealthStatus)
	})
}

func setStorageHealthGauges(driverName string, conditions []storageHealthKey) {
	// Clear previous gauges for this driver by setting known keys; callers pass the full current set.
	// We only set present conditions to 1. Recovery is reflected by absence of the condition in the
	// next update; DeletePartialMatch clears prior series for the driver.
	csiNodeStorageHealthStatus.DeletePartialMatch(map[string]string{"driver_name": driverName})
	for _, c := range conditions {
		csiNodeStorageHealthStatus.WithLabelValues(driverName, c.status, c.reason).Set(1)
	}
}

type storageHealthKey struct {
	status string
	reason string
}
