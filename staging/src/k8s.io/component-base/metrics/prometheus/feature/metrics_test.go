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
	testedMetrics = []string{"kubernetes_feature_enabled"}
)

func TestObserveHealthcheck(t *testing.T) {
	defer legacyregistry.Reset()
	defer ResetFeatureInfoMetric()

	testCases := []struct {
		desc    string
		name    string
		stage   string
		enabled bool
		want    string
	}{
		{
			desc:    "test enabled",
			name:    "feature-a",
			stage:   "ALPHA",
			enabled: true,
			want: `
       	# HELP kubernetes_feature_enabled [BETA] This metric records the data about the stage and enablement of a k8s feature.
        # TYPE kubernetes_feature_enabled gauge
        kubernetes_feature_enabled{name="feature-a",stage="ALPHA"} 1
`,
		},
		{
			desc:    "test disabled",
			name:    "feature-b",
			stage:   "BETA",
			enabled: false,
			want: `
       	# HELP kubernetes_feature_enabled [BETA] This metric records the data about the stage and enablement of a k8s feature.
        # TYPE kubernetes_feature_enabled gauge
        kubernetes_feature_enabled{name="feature-b",stage="BETA"} 0
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			defer ResetFeatureInfoMetric()
			RecordFeatureInfo(context.Background(), test.name, test.stage, test.enabled)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), testedMetrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
