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

const pvcProtectionControllerSubsystem = "pvc_protection_controller"

var (
	UnusedConditionSyncsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      pvcProtectionControllerSubsystem,
			Name:           "unused_condition_syncs_total",
			Help:           "Number of PVC Unused condition status update attempts",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"},
	)
)

var registerMetrics sync.Once

// Register registers PVC protection controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(UnusedConditionSyncsTotal)
	})
}
