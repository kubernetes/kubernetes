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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/pod-security-admission/api"
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
							})
							expectedLabels := map[string]string{
								"decision":          string(decision),
								"policy_level":      string(level),
								"policy_version":    expectedVersion,
								"mode":              string(mode),
								"request_operation": strings.ToLower(string(op)),
								"resource":          expectedResource,
								"subresource":       "",
							}
							val, err := testutil.GetCounterMetricValue(recorder.evaluationsCounter.With(expectedLabels))
							require.NoError(t, err, expectedLabels)

							if !assert.EqualValues(t, 1, val, expectedLabels) {
								findMetric(t, registry, "pod_security_evaluations_total")
							}

							recorder.Reset()
						}
					}
				}
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

// findMetric dumps non-zero metric samples for the metric with the given name, to help with debugging.
func findMetric(t *testing.T, gatherer metrics.Gatherer, metricName string) {
	t.Helper()
	m, _ := gatherer.Gather()
	for _, mFamily := range m {
		if mFamily.GetName() == metricName {
			for _, metric := range mFamily.GetMetric() {
				if metric.GetCounter().GetValue() > 0 {
					t.Logf("Found metric: %s", metric.String())
				}
			}
		}
	}
}
