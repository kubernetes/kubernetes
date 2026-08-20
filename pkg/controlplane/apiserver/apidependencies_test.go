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

package apiserver

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/component-base/featuregate"
)

func TestValidateFeatureGateAPIDependencies(t *testing.T) {
	const (
		featureA featuregate.Feature = "FeatureA"
		featureB featuregate.Feature = "FeatureB"
	)

	// Synthetic group/versions/resources. These do not need to be registered in
	// any scheme: the validation only consults the passed-in APIResourceConfigSource.
	var (
		gvV1      = schema.GroupVersion{Group: "example.io", Version: "v1"}
		gvV1beta1 = schema.GroupVersion{Group: "example.io", Version: "v1beta1"}
		gvV1alpha = schema.GroupVersion{Group: "example.io", Version: "v1alpha1"}

		widgets = "widgets"
		gadgets = "gadgets"
	)

	// enabledConfig returns a resource config that serves exactly the given group/versions.
	enabledConfig := func(versions ...schema.GroupVersion) serverstorage.APIResourceConfigSource {
		rc := serverstorage.NewResourceConfig()
		rc.EnableVersions(versions...)
		return rc
	}

	tests := []struct {
		name        string
		enabled     []featuregate.Feature
		deps        map[featuregate.Feature][]schema.GroupVersionResource
		resources   serverstorage.APIResourceConfigSource
		wantErr     bool
		wantContain []string
	}{
		{
			name:      "no dependencies declared",
			enabled:   []featuregate.Feature{featureA},
			deps:      map[featuregate.Feature][]schema.GroupVersionResource{},
			resources: serverstorage.NewResourceConfig(),
			wantErr:   false,
		},
		{
			name:    "feature disabled, required API not served",
			enabled: nil,
			deps: map[featuregate.Feature][]schema.GroupVersionResource{
				featureA: {gvV1.WithResource(widgets)},
			},
			resources: serverstorage.NewResourceConfig(),
			wantErr:   false,
		},
		{
			name:    "feature enabled, single required resource served",
			enabled: []featuregate.Feature{featureA},
			deps: map[featuregate.Feature][]schema.GroupVersionResource{
				featureA: {gvV1.WithResource(widgets)},
			},
			resources: enabledConfig(gvV1),
			wantErr:   false,
		},
		{
			name:    "feature enabled, single required resource not served",
			enabled: []featuregate.Feature{featureA},
			deps: map[featuregate.Feature][]schema.GroupVersionResource{
				featureA: {gvV1.WithResource(widgets)},
			},
			resources:   serverstorage.NewResourceConfig(),
			wantErr:     true,
			wantContain: []string{"FeatureA is enabled", "example.io/widgets", "[v1]"},
		},
		{
			name:    "multi-version resource, at least one version served",
			enabled: []featuregate.Feature{featureA},
			deps: map[featuregate.Feature][]schema.GroupVersionResource{
				featureA: {
					gvV1beta1.WithResource(widgets),
					gvV1alpha.WithResource(widgets),
				},
			},
			// Only the alpha version is served; that is sufficient.
			resources: enabledConfig(gvV1alpha),
			wantErr:   false,
		},
		{
			name:    "multi-version resource, no version served",
			enabled: []featuregate.Feature{featureA},
			deps: map[featuregate.Feature][]schema.GroupVersionResource{
				featureA: {
					gvV1beta1.WithResource(widgets),
					gvV1alpha.WithResource(widgets),
				},
			},
			resources:   serverstorage.NewResourceConfig(),
			wantErr:     true,
			wantContain: []string{"FeatureA is enabled", "example.io/widgets", "v1beta1", "v1alpha1"},
		},
		{
			name:    "multiple resources, one served one missing",
			enabled: []featuregate.Feature{featureA},
			deps: map[featuregate.Feature][]schema.GroupVersionResource{
				featureA: {
					gvV1beta1.WithResource(widgets),
					gvV1alpha.WithResource(gadgets),
				},
			},
			// widgets served via v1beta1, gadgets(v1alpha1) not served.
			resources:   enabledConfig(gvV1beta1),
			wantErr:     true,
			wantContain: []string{"example.io/gadgets"},
		},
		{
			name:    "multiple features, only enabled ones checked",
			enabled: []featuregate.Feature{featureB},
			deps: map[featuregate.Feature][]schema.GroupVersionResource{
				// featureA is unsatisfied but disabled, so it must be ignored.
				featureA: {gvV1.WithResource(widgets)},
				featureB: {gvV1.WithResource(gadgets)},
			},
			resources: enabledConfig(gvV1),
			wantErr:   false,
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
			for _, f := range tc.enabled {
				if err := gate.Set(string(f) + "=true"); err != nil {
					t.Fatalf("failed to enable %s: %v", f, err)
				}
			}

			err := validateFeatureGateAPIDependencies(gate, tc.resources, tc.deps)
			if tc.wantErr != (err != nil) {
				t.Fatalf("validateFeatureGateAPIDependencies() error = %v, wantErr %v", err, tc.wantErr)
			}
			if err != nil {
				for _, want := range tc.wantContain {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error %q does not contain %q", err.Error(), want)
					}
				}
			}
		})
	}
}
