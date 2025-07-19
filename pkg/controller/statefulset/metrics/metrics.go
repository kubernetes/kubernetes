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
	// MaxUnavailableViolations tracks the number of times that
	// .spec.replicas - .status.availableReplicas > .spec.updateStrategy.rollingUpdate.maxUnavailable.
	MaxUnavailableViolations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      StatefulSetControllerSubsystem,
			Name:           "statefulset_unavailability_violation",
			Help:           "Number of times maxunavailable has been violated",
			StabilityLevel: metrics.BETA,
		}, []string{"statefulset_namespace", "statefulset_name", "pod_management_policy"},
	)
)

var registerMetrics sync.Once

func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(MaxUnavailableViolations)
	})
}
