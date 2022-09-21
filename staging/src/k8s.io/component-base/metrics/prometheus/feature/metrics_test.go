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

package feature

import (
	"context"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

var (
	testedMetrics = []string{"k8s_feature_enabled"}
)

func TestObserveHealthcheck(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetFeatureEnabledMetric()

	testCases := []struct {
		desc    string
		name    string
		enabled bool
		want    string
	}{
		{
			desc:    "test enabled",
			name:    "feature-a",
			enabled: true,
			want: `
       	# HELP k8s_feature_enabled [ALPHA] This metric records the result of whether a feature is enabled.
        # TYPE k8s_feature_enabled gauge
        k8s_feature_enabled{enabled="true",name="feature-a"} 1
`,
		},
		{
			desc:    "test disabled",
			name:    "feature-b",
			enabled: false,
			want: `
       	# HELP k8s_feature_enabled [ALPHA] This metric records the result of whether a feature is enabled.
        # TYPE k8s_feature_enabled gauge
        k8s_feature_enabled{enabled="false",name="feature-b"} 1
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			defer ResetFeatureEnabledMetric()
			err := RecordFeatureEnabled(context.Background(), test.name, test.enabled)
			if err != nil {
				t.Errorf("unexpected err: %v", err)
			}

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), testedMetrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
