/*
Copyright 2020 The Kubernetes Authors.

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

// This file contains helpers for metrics that are associated to a profile.

var (
	scheduledResult     = "scheduled"
	unschedulableResult = "unschedulable"
	errorResult         = "error"
)

// PodScheduled can records a successful scheduling attempt and the duration
// since `start`.
func PodScheduled(profile string, duration float64) {
	observeScheduleAttemptAndLatency(scheduledResult, profile, duration)
}

// PodUnschedulable can records a scheduling attempt for an unschedulable pod
// and the duration since `start`.
func PodUnschedulable(profile string, duration float64) {
	observeScheduleAttemptAndLatency(unschedulableResult, profile, duration)
}

// PodScheduleError can records a scheduling attempt that had an error and the
// duration since `start`.
func PodScheduleError(profile string, duration float64) {
	observeScheduleAttemptAndLatency(errorResult, profile, duration)
}

func observeScheduleAttemptAndLatency(result, profile string, duration float64) {
	e2eSchedulingLatency.WithLabelValues(result, profile).Observe(duration)
	scheduleAttempts.WithLabelValues(result, profile).Inc()
}
