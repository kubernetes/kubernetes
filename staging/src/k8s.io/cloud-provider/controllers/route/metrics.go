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

package route

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	// subsystem is the name of this subsystem used for prometheus metrics.
	subsystem = "route_controller"
)

var registration sync.Once

var (
	routeSyncCount = metrics.NewCounter(&metrics.CounterOpts{
		Name:           "route_sync_total",
		Subsystem:      subsystem,
		Help:           "A metric counting the amount of times routes have been synced with the cloud provider.",
		StabilityLevel: metrics.BETA,
	})
)

func registerMetrics() {
	registration.Do(func() {
		legacyregistry.MustRegister(routeSyncCount)
	})
}
