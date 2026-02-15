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
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestPodFailuresHandledByFailurePolicy(t *testing.T) {
	// Reset the metric to ensure clean state
	PodFailuresHandledByFailurePolicy.Reset()

	// Register the metric
	Register()

	// Test with different action labels
	PodFailuresHandledByFailurePolicy.WithLabelValues("FailJob").Inc()
	PodFailuresHandledByFailurePolicy.WithLabelValues("Ignore").Inc()
	PodFailuresHandledByFailurePolicy.WithLabelValues("Count").Inc()

	want := `
		# HELP job_controller_pod_failures_handled_by_failure_policy_total [BETA] The number of failed Pods handled by failure policy with respect to the failure policy action applied based on the matched rule. Possible values of the action label correspond to the possible values for the failure policy rule action, which are: "FailJob", "Ignore" and "Count".
		# TYPE job_controller_pod_failures_handled_by_failure_policy_total counter
		job_controller_pod_failures_handled_by_failure_policy_total{action="Count"} 1
		job_controller_pod_failures_handled_by_failure_policy_total{action="FailJob"} 1
		job_controller_pod_failures_handled_by_failure_policy_total{action="Ignore"} 1
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "job_controller_pod_failures_handled_by_failure_policy_total"); err != nil {
		t.Fatal(err)
	}
}

func TestTerminatedPodsTrackingFinalizerTotal(t *testing.T) {
	// Reset the metric to ensure clean state
	TerminatedPodsTrackingFinalizerTotal.Reset()

	// Register the metric
	Register()

	// Test with add event
	TerminatedPodsTrackingFinalizerTotal.WithLabelValues("add").Inc()
	TerminatedPodsTrackingFinalizerTotal.WithLabelValues("add").Inc()

	// Test with delete event
	TerminatedPodsTrackingFinalizerTotal.WithLabelValues("delete").Inc()

	want := `
		# HELP job_controller_terminated_pods_tracking_finalizer_total [BETA] The number of terminated pods (phase=Failed|Succeeded) that have the finalizer batch.kubernetes.io/job-tracking. The event label can be "add" or "delete".
		# TYPE job_controller_terminated_pods_tracking_finalizer_total counter
		job_controller_terminated_pods_tracking_finalizer_total{event="add"} 2
		job_controller_terminated_pods_tracking_finalizer_total{event="delete"} 1
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "job_controller_terminated_pods_tracking_finalizer_total"); err != nil {
		t.Fatal(err)
	}
}
