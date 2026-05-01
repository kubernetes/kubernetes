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

package testutil

import (
	"fmt"
	"io"
	"strings"

	"github.com/prometheus/client_golang/prometheus/testutil/promlint"
)

// exceptionMetrics is an exception list of metrics which violates promlint rules.
//
// The original entries come from the existing metrics when we introduce promlint.
// We setup this list for allow and not fail on the current violations.
// Generally speaking, you need to fix the problem for a new metric rather than add it into the list.
var exceptionMetrics = []string{
	// k8s.io/apiserver/pkg/server/egressselector
	"apiserver_egress_dialer_dial_failure_count", // counter metrics should have "_total" suffix

	// k8s.io/apiserver/pkg/endpoints/filters
	"authenticated_user_requests", // counter metrics should have "_total" suffix
	"authentication_attempts",     // counter metrics should have "_total" suffix

	// kube-apiserver
	"aggregator_openapi_v2_regeneration_count",                    // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix
	"apiserver_admission_step_admission_duration_seconds_summary", // metric name should not include type 'summary'
	"apiserver_admission_webhook_fail_open_count",                 // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix
	"apiserver_admission_webhook_rejection_count",                 // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix
	"apiserver_flowcontrol_latest_s",                              // metric names should not contain abbreviated units
	"apiserver_flowcontrol_next_discounted_s_bounds",              // metric names should not contain abbreviated units
	"apiserver_flowcontrol_next_s_bounds",                         // metric names should not contain abbreviated units
	"authentication_token_cache_active_fetch_count",               // no help text; non-histogram and non-summary metrics should not have "_count" suffix
	"authentication_token_cache_fetch_total",                      // no help text
	"authentication_token_cache_request_duration_seconds",         // no help text
	"authentication_token_cache_request_total",                    // no help text
	"apiserver_watch_shards_total",                                // non-counter metrics should not have "_total" suffix

	// apiextensions-apiserver
	"apiextensions_openapi_v2_regeneration_count", // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix
	"apiextensions_openapi_v3_regeneration_count", // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix

	// attach-detach controller
	"attach_detach_controller_attachdetach_controller_forced_detaches", // counter metrics should have "_total" suffix
	"endpoint_slice_controller_changes",                                // counter metrics should have "_total" suffix
	"endpoint_slice_controller_syncs",                                  // counter metrics should have "_total" suffix
	"endpoint_slice_mirroring_controller_changes",                      // counter metrics should have "_total" suffix

	// kubelet
	"kubelet_container_aligned_compute_resources_count",         // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix
	"kubelet_container_aligned_compute_resources_failure_count", // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix
	"kubelet_cpu_manager_exclusive_cpu_allocation_count",        // non-histogram and non-summary metrics should not have "_count" suffix
	"kubelet_evented_pleg_connection_error_count",               // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix
	"kubelet_evented_pleg_connection_success_count",             // counter metrics should have "_total" suffix; non-histogram and non-summary metrics should not have "_count" suffix
	"kubelet_evictions",                                          // counter metrics should have "_total" suffix
	"kubelet_pleg_discard_events",                                // counter metrics should have "_total" suffix
	"kubelet_pod_resources_endpoint_errors_get",                  // counter metrics should have "_total" suffix
	"kubelet_pod_resources_endpoint_errors_get_allocatable",      // counter metrics should have "_total" suffix
	"kubelet_pod_resources_endpoint_errors_list",                 // counter metrics should have "_total" suffix
	"kubelet_pod_resources_endpoint_requests_get",                // counter metrics should have "_total" suffix
	"kubelet_pod_resources_endpoint_requests_get_allocatable",    // counter metrics should have "_total" suffix
	"kubelet_pod_resources_endpoint_requests_list",               // counter metrics should have "_total" suffix
	"kubelet_preemptions",                                        // counter metrics should have "_total" suffix
	"kubelet_server_expiration_renew_errors",                     // counter metrics should have "_total" suffix
	"kubelet_topology_manager_admission_duration_ms",             // metric names should not contain abbreviated units
	"kubelet_certificate_manager_client_expiration_renew_errors", // counter metrics should have "_total" suffix
	"kubelet_pod_resize_duration_milliseconds",                   // use base unit "seconds" instead of "milliseconds"
	"resource_manager_container_assignments",                     // counter metrics should have "_total" suffix

	// kube-proxy
	"kubeproxy_sync_proxy_rules_iptables_total",           // non-counter metrics should not have "_total" suffix
	"kubeproxy_sync_proxy_rules_no_local_endpoints_total", // non-counter metrics should not have "_total" suffix

	// volume-manager
	"volume_manager_selinux_container_errors_total",                 // non-counter metrics should not have "_total" suffix
	"volume_manager_selinux_container_warnings_total",               // non-counter metrics should not have "_total" suffix
	"volume_manager_selinux_pod_context_mismatch_errors_total",      // non-counter metrics should not have "_total" suffix
	"volume_manager_selinux_pod_context_mismatch_warnings_total",    // non-counter metrics should not have "_total" suffix
	"volume_manager_selinux_volume_context_mismatch_errors_total",   // non-counter metrics should not have "_total" suffix
	"volume_manager_selinux_volume_context_mismatch_warnings_total", // non-counter metrics should not have "_total" suffix
	"volume_manager_selinux_volumes_admitted_total",                 // non-counter metrics should not have "_total" suffix

	// persistent-volume-controller
	"volume_operation_total_errors", // counter metrics should have "_total" suffix
}

// A Problem is an issue detected by a Linter.
type Problem promlint.Problem

func (p *Problem) String() string {
	return fmt.Sprintf("%s:%s", p.Metric, p.Text)
}

// A Linter is a Prometheus metrics linter.  It identifies issues with metric
// names, types, and metadata, and reports them to the caller.
type Linter struct {
	promLinter *promlint.Linter
}

// Lint performs a linting pass, returning a slice of Problems indicating any
// issues found in the metrics stream.  The slice is sorted by metric name
// and issue description.
func (l *Linter) Lint() ([]Problem, error) {
	promProblems, err := l.promLinter.Lint()
	if err != nil {
		return nil, err
	}

	// Ignore problems those in exception list
	problems := make([]Problem, 0, len(promProblems))
	for i := range promProblems {
		if !l.shouldIgnore(promProblems[i].Metric) {
			problems = append(problems, Problem(promProblems[i]))
		}
	}

	return problems, nil
}

// shouldIgnore returns true if metric in the exception list, otherwise returns false.
func (l *Linter) shouldIgnore(metricName string) bool {
	for i := range exceptionMetrics {
		if metricName == exceptionMetrics[i] {
			return true
		}
	}

	return false
}

// NewPromLinter creates a new Linter that reads an input stream of Prometheus metrics.
// Only the text exposition format is supported.
func NewPromLinter(r io.Reader) *Linter {
	return &Linter{
		promLinter: promlint.New(r),
	}
}

func mergeProblems(problems []Problem) string {
	var problemsMsg []string

	for index := range problems {
		problemsMsg = append(problemsMsg, problems[index].String())
	}

	return strings.Join(problemsMsg, ",")
}

// shouldIgnore returns true if metric in the exception list, otherwise returns false.
func shouldIgnore(metricName string) bool {
	for i := range exceptionMetrics {
		if metricName == exceptionMetrics[i] {
			return true
		}
	}

	return false
}

// GetLintError will ignore the metrics in exception list and converts lint problem to error.
func GetLintError(problems []promlint.Problem) error {
	var filteredProblems []Problem
	for _, problem := range problems {
		if shouldIgnore(problem.Metric) {
			continue
		}

		filteredProblems = append(filteredProblems, Problem(problem))
	}

	if len(filteredProblems) == 0 {
		return nil
	}

	return fmt.Errorf("lint error: %s", mergeProblems(filteredProblems))
}

// CheckUnusedExceptions returns an error if any metric in exceptionMetrics did not have any lint problems.
func CheckUnusedExceptions(problems []promlint.Problem) error {
	usedExceptions := make(map[string]bool)
	for _, p := range problems {
		if shouldIgnore(p.Metric) {
			usedExceptions[p.Metric] = true
		}
	}

	var unused []string
	for _, exc := range exceptionMetrics {
		if !usedExceptions[exc] {
			unused = append(unused, exc)
		}
	}

	if len(unused) > 0 {
		return fmt.Errorf("metrics in exception list but have no violations: %s", strings.Join(unused, ", "))
	}
	return nil
}
