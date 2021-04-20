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

// JobControllerSubsystem - subsystem name used for this controller.
const JobControllerSubsystem = "job_controller"

var (
	// JobSyncDurationSeconds tracks the latency of job syncs as
	// completion_mode = Indexed / NonIndexed and result = success / error.
	JobSyncDurationSeconds = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      JobControllerSubsystem,
			Name:           "job_sync_duration_seconds",
			Help:           "The time it took to sync a job",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
		},
		[]string{"completion_mode", "result"},
	)
	// JobSyncNum tracks the number of job syncs as
	// completion_mode = Indexed / NonIndexed and result = success / error.
	JobSyncNum = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      JobControllerSubsystem,
			Name:           "job_sync_total",
			Help:           "The number of job syncs",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"completion_mode", "result"},
	)
	// JobFinishedNum tracks the number of jobs that finish as
	// completion_mode = Indexed / NonIndexed and result = failed / succeeded.
	JobFinishedNum = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      JobControllerSubsystem,
			Name:           "job_finished_total",
			Help:           "The number of finished job",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"completion_mode", "result"},
	)
)

var registerMetrics sync.Once

// Register registers Job controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(JobSyncDurationSeconds)
		legacyregistry.MustRegister(JobSyncNum)
		legacyregistry.MustRegister(JobFinishedNum)
	})
}
