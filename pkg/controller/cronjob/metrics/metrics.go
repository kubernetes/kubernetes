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

const CronJobControllerSubsystem = "cronjob_controller"

var (
	CronJobCreationSkew = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      CronJobControllerSubsystem,
			Name:           "job_creation_skew_duration_seconds",
			Help:           "Time between when a cronjob is scheduled to be run, and when the corresponding job is created",
			StabilityLevel: metrics.STABLE,
			Buckets:        metrics.ExponentialBuckets(1, 2, 10),
		},
	)
)

var registerMetrics sync.Once

// Register registers CronjobController metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(CronJobCreationSkew)
	})
}
