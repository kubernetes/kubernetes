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
	"k8s.io/component-base/metrics"
)

const ReplicaSetControllerSubsystem = "replicaset_controller"

var SortingDeletionAgeRatio = metrics.NewHistogram(
	&metrics.HistogramOpts{
		Subsystem: ReplicaSetControllerSubsystem,
		Name:      "sorting_deletion_age_ratio",
		Help: "The ratio of chosen deleted pod's ages to the current youngest pod's age (at the time). Should be <2." +
			"The intent of this metric is to measure the rough efficacy of the LogarithmicScaleDown feature gate's effect on" +
			"the sorting (and deletion) of pods when a replicaset scales down. This only considers Ready pods when calculating and reporting.",
		Buckets:        metrics.ExponentialBuckets(0.25, 2, 6),
		StabilityLevel: metrics.ALPHA,
	},
)

// Register registers ReplicaSet controller metrics.
func Register(registrationFunc func(metrics.Registerable) error) error {
	return registrationFunc(SortingDeletionAgeRatio)
}
