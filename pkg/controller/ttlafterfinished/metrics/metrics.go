/*
Copyright 2021 The Kubernetes Authors.

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

// TTLAfterFinishedSubsystem - subsystem name used for this controller.
const TTLAfterFinishedSubsystem = "ttl_after_finished_controller"

var (
	// JobDeletionDurationSeconds tracks the time it took to delete the job since it
	// became eligible for deletion.
	JobDeletionDurationSeconds = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      TTLAfterFinishedSubsystem,
			Name:           "job_deletion_duration_seconds",
			Help:           "The time it took to delete the job since it became eligible for deletion",
			StabilityLevel: metrics.ALPHA,
			// Start with 100ms with the last bucket being [~27m, Inf).
			Buckets: metrics.ExponentialBuckets(0.1, 2, 14),
		},
	)
)

var registerMetrics sync.Once

// Register registers TTL after finished controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(JobDeletionDurationSeconds)
	})
}
