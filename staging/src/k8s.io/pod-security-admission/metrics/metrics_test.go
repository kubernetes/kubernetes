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
	"bytes"
	"fmt"
	"sort"
	"strings"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/pod-security-admission/api"

	"github.com/stretchr/testify/assert"
)

var (
	decisions  = []Decision{DecisionAllow, DecisionDeny}
	modes      = []Mode{ModeEnforce, ModeAudit, ModeWarn}
	operations = []admissionv1.Operation{admissionv1.Create, admissionv1.Update}
	levels     = []api.Level{api.LevelPrivileged, api.LevelBaseline, api.LevelRestricted}

	// Map of resource types to test to expected label value.
	resourceExpectations = map[schema.GroupVersionResource]string{
		corev1.SchemeGroupVersion.WithResource("pods"):        "pod",
		appsv1.SchemeGroupVersion.WithResource("deployments"): "controller",
		batchv1.SchemeGroupVersion.WithResource("cronjobs"):   "controller",
	}

	// Map of versions to expected label value (compared against testVersion).
	versionExpectations = map[string]string{
		"latest": "latest",
		"v1.22":  "v1.22",
		"v1.23":  "v1.23",
		"v1.24":  "future",
	}
	testVersion = api.MajorMinorVersion(1, 23)
)

func TestRecordEvaluation(t *testing.T) {
	recorder := NewPrometheusRecorder(testVersion)
	registry := testutil.NewFakeKubeRegistry("1.23.0")
	recorder.MustRegister(registry.MustRegister)

	for _, decision := range decisions {
		for _, mode := range modes {
			for _, op := range operations {
				for _, level := range levels {
					for version, expectedVersion := range versionExpectations {
						for resource, expectedResource := range resourceExpectations {
							recorder.RecordEvaluation(decision, levelVersion(level, version), mode, &api.AttributesRecord{
								Resource:  resource,
								Operation: op,
								Namespace: "some-namespace",
							})

							if level == api.LevelPrivileged {
								expectedVersion = "latest"
							}

							expected := fmt.Sprintf(`
							# HELP pod_security_evaluations_total [ALPHA] Number of policy evaluations that occurred, not counting ignored or exempt requests.
        	            	# TYPE pod_security_evaluations_total counter
							pod_security_evaluations_total{decision="%s",mode="%s",ocp_namespace="",policy_level="%s",policy_version="%s",request_operation="%s",resource="%s",subresource=""} 1
							`, decision, mode, level, expectedVersion, strings.ToLower(string(op)), expectedResource)
							expected = expectCachedMetrics("pod_security_evaluations_total", expected)

							assert.NoError(t, testutil.GatherAndCompare(registry, bytes.NewBufferString(expected), "pod_security_evaluations_total"))

							recorder.Reset()
						}
					}
				}
			}
		}
	}
}

func TestRecordExemption(t *testing.T) {
	recorder := NewPrometheusRecorder(testVersion)
	registry := testutil.NewFakeKubeRegistry("1.23.0")
	recorder.MustRegister(registry.MustRegister)

	for _, op := range operations {
		for resource, expectedResource := range resourceExpectations {
			for _, subresource := range []string{"", "ephemeralcontainers"} {
				recorder.RecordExemption(&api.AttributesRecord{
					Resource:    resource,
					Operation:   op,
					Subresource: subresource,
				})

				expected := fmt.Sprintf(`
				# HELP pod_security_exemptions_total [ALPHA] Number of exempt requests, not counting ignored or out of scope requests.
				# TYPE pod_security_exemptions_total counter
				pod_security_exemptions_total{request_operation="%s",resource="%s",subresource="%s"} 1
				`, strings.ToLower(string(op)), expectedResource, subresource)
				expected = expectCachedMetrics("pod_security_exemptions_total", expected)

				assert.NoError(t, testutil.GatherAndCompare(registry, bytes.NewBufferString(expected), "pod_security_exemptions_total"))

				recorder.Reset()
			}
		}
	}
}

func TestRecordError(t *testing.T) {
	recorder := NewPrometheusRecorder(testVersion)
	registry := testutil.NewFakeKubeRegistry("1.23.0")
	recorder.MustRegister(registry.MustRegister)

	for _, fatal := range []bool{true, false} {
		for _, op := range operations {
			for resource, expectedResource := range resourceExpectations {
				recorder.RecordError(fatal, &api.AttributesRecord{
					Resource:  resource,
					Operation: op,
				})

				expected := bytes.NewBufferString(fmt.Sprintf(`
				# HELP pod_security_errors_total [ALPHA] Number of errors preventing normal evaluation. Non-fatal errors may result in the latest restricted profile being used for evaluation.
				# TYPE pod_security_errors_total counter
				pod_security_errors_total{fatal="%t",request_operation="%s",resource="%s",subresource=""} 1
				`, fatal, strings.ToLower(string(op)), expectedResource))

				assert.NoError(t, testutil.GatherAndCompare(registry, expected, "pod_security_errors_total"))

				recorder.Reset()
			}
		}
	}
}

func levelVersion(level api.Level, version string) api.LevelVersion {
	lv := api.LevelVersion{Level: level}
	var err error
	if lv.Version, err = api.ParseVersion(version); err != nil {
		panic(err)
	}
	return lv
}

// The cached metrics should always be present (value 0 if not counted).
var expectedCachedMetrics = map[string][]string{
	"pod_security_evaluations_total": {
		`pod_security_evaluations_total{decision="allow",mode="enforce",ocp_namespace="",policy_level="privileged",policy_version="latest",request_operation="create",resource="pod",subresource=""}`,
		`pod_security_evaluations_total{decision="allow",mode="enforce",ocp_namespace="",policy_level="privileged",policy_version="latest",request_operation="update",resource="pod",subresource=""}`,
	},
	"pod_security_exemptions_total": {
		`pod_security_exemptions_total{request_operation="create",resource="controller",subresource=""}`,
		`pod_security_exemptions_total{request_operation="create",resource="pod",subresource=""}`,
		`pod_security_exemptions_total{request_operation="update",resource="controller",subresource=""}`,
		`pod_security_exemptions_total{request_operation="update",resource="pod",subresource=""}`,
	},
}

func expectCachedMetrics(metricName, expected string) string {
	expectations := strings.Split(strings.TrimSpace(expected), "\n")
	for i, expectation := range expectations {
		expectations[i] = strings.TrimSpace(expectation) // Whitespace messes with sorting.
	}
	for _, cached := range expectedCachedMetrics[metricName] {
		expectations = addZeroExpectation(expectations, cached)
	}
	sort.Strings(expectations[:len(expectations)-1])
	return "\n" + strings.Join(expectations, "\n") + "\n"
}

// addZeroExpectation adds the mixin as an empty sample if not already present.
func addZeroExpectation(currentExpectations []string, mixin string) []string {
	for _, current := range currentExpectations {
		if strings.HasPrefix(current, mixin) {
			return currentExpectations // Mixin value already present.
		}
	}
	return append(currentExpectations, fmt.Sprintf("%s 0", mixin))
}
