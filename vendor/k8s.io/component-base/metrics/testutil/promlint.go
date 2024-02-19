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
	// k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/server/egressselector
	"apiserver_egress_dialer_dial_failure_count", // counter metrics should have "_total" suffix

	// k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/server/healthz
	"apiserver_request_total", // label names should be written in 'snake_case' not 'camelCase'

	// k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/endpoints/filters
	"authenticated_user_requests", // counter metrics should have "_total" suffix
	"authentication_attempts",     // counter metrics should have "_total" suffix

	// kube-apiserver
	"aggregator_openapi_v2_regeneration_count",
	"apiserver_admission_step_admission_duration_seconds_summary",
	"apiserver_current_inflight_requests",
	"apiserver_longrunning_gauge",
	"get_token_count",
	"get_token_fail_count",
	"ssh_tunnel_open_count",
	"ssh_tunnel_open_fail_count",

	// kube-controller-manager
	"attachdetach_controller_forced_detaches",
	"authenticated_user_requests",
	"authentication_attempts",
	"get_token_count",
	"get_token_fail_count",
	"node_collector_evictions_number",
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

// getLintError will ignore the metrics in exception list and converts lint problem to error.
func getLintError(problems []promlint.Problem) error {
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
