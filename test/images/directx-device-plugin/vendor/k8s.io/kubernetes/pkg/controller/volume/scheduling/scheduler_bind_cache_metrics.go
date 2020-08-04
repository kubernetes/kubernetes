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

package scheduling

import (
	"github.com/prometheus/client_golang/prometheus"
)

// VolumeSchedulerSubsystem - subsystem name used by scheduler
const VolumeSchedulerSubsystem = "scheduler_volume"

var (
	// VolumeBindingRequestSchedulerBinderCache tracks the number of volume binder cache operations.
	VolumeBindingRequestSchedulerBinderCache = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: VolumeSchedulerSubsystem,
			Name:      "binder_cache_requests_total",
			Help:      "Total number for request volume binding cache",
		},
		[]string{"operation"},
	)
	// VolumeSchedulingStageLatency tracks the latency of volume scheduling operations.
	VolumeSchedulingStageLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: VolumeSchedulerSubsystem,
			Name:      "scheduling_duration_seconds",
			Help:      "Volume scheduling stage latency",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
		[]string{"operation"},
	)
	// VolumeSchedulingStageFailed tracks the number of failed volume scheduling operations.
	VolumeSchedulingStageFailed = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: VolumeSchedulerSubsystem,
			Name:      "scheduling_stage_error_total",
			Help:      "Volume scheduling stage error count",
		},
		[]string{"operation"},
	)
)

// RegisterVolumeSchedulingMetrics is used for scheduler, because the volume binding cache is a library
// used by scheduler process.
func RegisterVolumeSchedulingMetrics() {
	prometheus.MustRegister(VolumeBindingRequestSchedulerBinderCache)
	prometheus.MustRegister(VolumeSchedulingStageLatency)
	prometheus.MustRegister(VolumeSchedulingStageFailed)
}
