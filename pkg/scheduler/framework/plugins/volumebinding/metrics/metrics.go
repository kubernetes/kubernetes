/*
Copyright 2018 The Kubernetes Authors.

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
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// VolumeSchedulerSubsystem - subsystem name used by scheduler
const VolumeSchedulerSubsystem = "scheduler_volume"

var (
	// VolumeBindingRequestSchedulerBinderCache tracks the number of volume binder cache operations.
	VolumeBindingRequestSchedulerBinderCache = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      VolumeSchedulerSubsystem,
			Name:           "binder_cache_requests_total",
			Help:           "Total number for request volume binding cache",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation"},
	)
	// VolumeSchedulingStageFailed tracks the number of failed volume scheduling operations.
	VolumeSchedulingStageFailed = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      VolumeSchedulerSubsystem,
			Name:           "scheduling_stage_error_total",
			Help:           "Volume scheduling stage error count",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation"},
	)
)

// RegisterVolumeSchedulingMetrics is used for scheduler, because the volume binding cache is a library
// used by scheduler process.
func RegisterVolumeSchedulingMetrics() {
	legacyregistry.MustRegister(VolumeBindingRequestSchedulerBinderCache)
	legacyregistry.MustRegister(VolumeSchedulingStageFailed)
}
