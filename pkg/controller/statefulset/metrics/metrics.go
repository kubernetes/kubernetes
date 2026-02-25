/*
Copyright 2025 The Kubernetes Authors.

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

const StatefulSetControllerSubsystem = "statefulset_controller"

var (
	// MaxUnavailable tracks the current .spec.updateStrategy.rollingUpdate.maxUnavailable value with the
	// MaxUnavailableStatefulSet feature enabled. This gauge reflects the configured maximum number of pods
	// that can be unavailable during rolling updates, providing visibility into the availability constraints.
	// The metric is set to 1 by default.
	//
	// Sample monitoring queries:
	// - Current maxUnavailable setting: statefulset_max_unavailable
	// - Compare with actual unavailable: statefulset_unavailable_replicas - statefulset_max_unavailable
	// - Alert when exceeding limit: statefulset_unavailable_replicas > statefulset_max_unavailable
	// - Monitor configuration changes: changes(statefulset_max_unavailable[1h])
	MaxUnavailable = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      StatefulSetControllerSubsystem,
			Name:           "statefulset_max_unavailable",
			Help:           "Maximum number of unavailable pods allowed during StatefulSet rolling updates",
			StabilityLevel: metrics.ALPHA,
		}, []string{"statefulset_namespace", "statefulset_name", "pod_management_policy"},
	)

	// UnavailableReplicas tracks the current number of unavailable pods in a StatefulSet.
	// This gauge reflects the real-time count of pods that are either missing or unavailable
	// (i.e., not ready for .spec.minReadySeconds).
	//
	// Sample monitoring queries:
	// - Current unavailable pods: statefulset_unavailable_replicas
	// - Availability percentage: (statefulset_replicas - statefulset_unavailable_replicas) / statefulset_replicas * 100
	// - Alert on high unavailability: statefulset_unavailable_replicas > statefulset_max_unavailable
	// - Monitor availability trends: statefulset_unavailable_replicas
	UnavailableReplicas = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      StatefulSetControllerSubsystem,
			Name:           "statefulset_unavailable_replicas",
			Help:           "Current number of unavailable pods in StatefulSet",
			StabilityLevel: metrics.ALPHA,
		}, []string{"statefulset_namespace", "statefulset_name", "pod_management_policy"},
	)
)

var registerMetrics sync.Once

func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(MaxUnavailable)
		legacyregistry.MustRegister(UnavailableReplicas)
	})
}
