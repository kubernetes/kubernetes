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

package health

import (
	"context"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

var (
	testedMetrics = []string{"k8s_healthcheck"}
)

func TestObserveHealthcheck(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetHealthMetrics()
	initialState := ERROR
	healthcheckName := "healthcheck-a"
	initialOutput := `
        # HELP k8s_healthcheck [ALPHA] This metric records the result of a single health check.
        # TYPE k8s_healthcheck gauge
        k8s_healthcheck{name="healthcheck-a",status="error",type="healthz"} 1
        k8s_healthcheck{name="healthcheck-a",status="pending",type="healthz"} 0
        k8s_healthcheck{name="healthcheck-a",status="success",type="healthz"} 0
`
	testCases := []struct {
		desc     string
		name     string
		hcType   HealthcheckType
		hcStatus HealthcheckStatus
		want     string
	}{
		{
			desc:     "test pending",
			name:     healthcheckName,
			hcType:   HEALTHZ,
			hcStatus: PENDING,
			want: `
        # HELP k8s_healthcheck [ALPHA] This metric records the result of a single health check.
        # TYPE k8s_healthcheck gauge
        k8s_healthcheck{name="healthcheck-a",status="error",type="healthz"} 0
        k8s_healthcheck{name="healthcheck-a",status="pending",type="healthz"} 1
        k8s_healthcheck{name="healthcheck-a",status="success",type="healthz"} 0
`,
		},
		{
			desc:     "test success",
			name:     healthcheckName,
			hcType:   HEALTHZ,
			hcStatus: SUCCESS,
			want: `
        # HELP k8s_healthcheck [ALPHA] This metric records the result of a single health check.
        # TYPE k8s_healthcheck gauge
        k8s_healthcheck{name="healthcheck-a",status="error",type="healthz"} 0
        k8s_healthcheck{name="healthcheck-a",status="pending",type="healthz"} 0
        k8s_healthcheck{name="healthcheck-a",status="success",type="healthz"} 1
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			// let's first record an error as initial state
			err := ObserveHealthcheck(context.Background(), test.name, test.hcType, initialState)
			if err != nil {
				t.Errorf("unexpected err: %v", err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(initialOutput), testedMetrics...); err != nil {
				t.Fatal(err)
			}
			// now record that we successfully purge state
			err = ObserveHealthcheck(context.Background(), test.name, test.hcType, test.hcStatus)
			if err != nil {
				t.Errorf("unexpected err: %v", err)
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), testedMetrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
