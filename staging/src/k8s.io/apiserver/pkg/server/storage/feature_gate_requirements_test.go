/*
Copyright The Kubernetes Authors.

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

package storage

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/featuregate"
)

func TestFeatureGateAPIRequirementsValidate(t *testing.T) {
	const (
		featureA featuregate.Feature = "FeatureA"
		featureB featuregate.Feature = "FeatureB"
	)

	var (
		firstResource  = schema.GroupResource{Group: "one", Resource: "first"}
		secondResource = schema.GroupResource{Group: "one", Resource: "second"}
		legacyResource = schema.GroupResource{Resource: "legacy"}
	)

	tests := []struct {
		name         string
		enabled      []featuregate.Feature
		requirements FeatureGateAPIRequirements
		served       sets.Set[schema.GroupResource]
		wantErr      bool
		wantContains []string
	}{
		{
			name:         "no requirements declared",
			enabled:      []featuregate.Feature{featureA},
			requirements: FeatureGateAPIRequirements{},
			served:       sets.New[schema.GroupResource](),
		},
		{
			name:         "disabled feature with unserved requirement is ignored",
			requirements: FeatureGateAPIRequirements{featureA: {firstResource}},
			served:       sets.New[schema.GroupResource](),
		},
		{
			name:         "enabled feature with served requirement",
			enabled:      []featuregate.Feature{featureA},
			requirements: FeatureGateAPIRequirements{featureA: {firstResource}},
			served:       sets.New(firstResource),
		},
		{
			name:         "enabled feature with unserved requirement",
			enabled:      []featuregate.Feature{featureA},
			requirements: FeatureGateAPIRequirements{featureA: {firstResource}},
			served:       sets.New[schema.GroupResource](),
			wantErr:      true,
			wantContains: []string{"FeatureA is enabled", "first.one"},
		},
		{
			name:         "enabled feature reports only the unserved of several requirements",
			enabled:      []featuregate.Feature{featureA},
			requirements: FeatureGateAPIRequirements{featureA: {firstResource, secondResource}},
			served:       sets.New(firstResource),
			wantErr:      true,
			wantContains: []string{"second.one"},
		},
		{
			name:         "legacy group resource renders without a group suffix",
			enabled:      []featuregate.Feature{featureA},
			requirements: FeatureGateAPIRequirements{featureA: {legacyResource}},
			served:       sets.New[schema.GroupResource](),
			wantErr:      true,
			wantContains: []string{"not served: legacy;"},
		},
		{
			name:    "only enabled features are checked",
			enabled: []featuregate.Feature{featureB},
			requirements: FeatureGateAPIRequirements{
				featureA: {firstResource},
				featureB: {secondResource},
			},
			served: sets.New(secondResource),
		},
		{
			name:    "several failing features are all reported",
			enabled: []featuregate.Feature{featureA, featureB},
			requirements: FeatureGateAPIRequirements{
				featureA: {firstResource},
				featureB: {secondResource},
			},
			served:       sets.New[schema.GroupResource](),
			wantErr:      true,
			wantContains: []string{"FeatureA is enabled", "FeatureB is enabled"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gate := featuregate.NewFeatureGate()
			if err := gate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
				featureA: {Default: false, PreRelease: featuregate.Alpha},
				featureB: {Default: false, PreRelease: featuregate.Alpha},
			}); err != nil {
				t.Fatalf("failed to add features: %v", err)
			}
			for _, feature := range tc.enabled {
				if err := gate.Set(string(feature) + "=true"); err != nil {
					t.Fatalf("failed to enable %s: %v", feature, err)
				}
			}

			err := tc.requirements.Validate(gate, tc.served)
			if tc.wantErr != (err != nil) {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tc.wantErr)
			}
			if err == nil {
				return
			}
			for _, want := range tc.wantContains {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not contain %q", err.Error(), want)
				}
			}
		})
	}
}
