/*
Copyright 2022 The Kubernetes Authors.

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

package slis

import (
	"context"
	"strings"
	"testing"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

var (
	testedMetrics = []string{"kubernetes_healthcheck", "kubernetes_healthchecks_total"}
)

func TestObserveHealthcheck(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	defer registry.Reset()
	defer ResetHealthMetrics()
	Register(registry)
	initialState := Error
	healthcheckName := "healthcheck-a"
	initialOutput := `
        # HELP kubernetes_healthcheck [STABLE] This metric records the result of a single healthcheck.
        # TYPE kubernetes_healthcheck gauge
        kubernetes_healthcheck{name="healthcheck-a",type="healthz"} 0
        # HELP kubernetes_healthchecks_total [STABLE] This metric records the results of all healthcheck.
        # TYPE kubernetes_healthchecks_total counter
        kubernetes_healthchecks_total{name="healthcheck-a",status="error",type="healthz"} 1
`
	testCases := []struct {
		desc     string
		name     string
		hcType   string
		hcStatus HealthcheckStatus
		want     string
	}{
		{
			desc:     "test success",
			name:     healthcheckName,
			hcType:   "healthz",
			hcStatus: Success,
			want: `
        # HELP kubernetes_healthcheck [STABLE] This metric records the result of a single healthcheck.
        # TYPE kubernetes_healthcheck gauge
        kubernetes_healthcheck{name="healthcheck-a",type="healthz"} 1
        # HELP kubernetes_healthchecks_total [STABLE] This metric records the results of all healthcheck.
        # TYPE kubernetes_healthchecks_total counter
        kubernetes_healthchecks_total{name="healthcheck-a",status="error",type="healthz"} 1
        kubernetes_healthchecks_total{name="healthcheck-a",status="success",type="healthz"} 1
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			defer ResetHealthMetrics()
			// let's first record an error as initial state
			err := ObserveHealthcheck(context.Background(), test.name, test.hcType, initialState)
			if err != nil {
				t.Errorf("unexpected err: %v", err)
			}
			if err := testutil.GatherAndCompare(registry, strings.NewReader(initialOutput), testedMetrics...); err != nil {
				t.Fatal(err)
			}
			// now record that we successfully purge state
			err = ObserveHealthcheck(context.Background(), test.name, test.hcType, test.hcStatus)
			if err != nil {
				t.Errorf("unexpected err: %v", err)
			}
			if err := testutil.GatherAndCompare(registry, strings.NewReader(test.want), testedMetrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
