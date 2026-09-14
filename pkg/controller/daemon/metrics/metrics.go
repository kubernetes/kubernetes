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

package metrics

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const DaemonControllerSubsystem = "daemonset_controller"

var (
	DaemonsetRequeueSkips = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      DaemonControllerSubsystem,
			Name:           "stale_sync_skips_total",
			Help:           "Total number of DaemonSet syncs skipped due to a stale watch cache.",
			StabilityLevel: metrics.ALPHA,
		},
		// These are the labels (dimensions)
		[]string{"group", "resource"},
	)
)

var registerMetrics sync.Once

// Register registers DaemonSet Controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(DaemonsetRequeueSkips)
	})
}
