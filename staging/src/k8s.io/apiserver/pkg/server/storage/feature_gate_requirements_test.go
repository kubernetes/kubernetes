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
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/featuregate"
)

const (
	featureA         featuregate.Feature = "FeatureA"
	featureB         featuregate.Feature = "FeatureB"
	featureDefaultOn featuregate.Feature = "FeatureDefaultOn"
)

func newTestGate(t *testing.T, set ...string) featuregate.MutableVersionedFeatureGate {
	t.Helper()
	gate := featuregate.NewFeatureGate()
	if err := gate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		featureA:         {Default: false, PreRelease: featuregate.Alpha},
		featureB:         {Default: false, PreRelease: featuregate.Alpha},
		featureDefaultOn: {Default: true, PreRelease: featuregate.Beta},
	}); err != nil {
		t.Fatalf("failed to add features: %v", err)
	}
	for _, s := range set {
		if err := gate.Set(s); err != nil {
			t.Fatalf("failed to set %q: %v", s, err)
		}
	}
	return gate
}

// gateWithoutExplicitness hides ExplicitlySet, standing in for a FeatureGate implementation
// that cannot tell an explicit setting from a default.
type gateWithoutExplicitness struct {
	featuregate.FeatureGate
}

func TestFeatureGateAPIRequirementsValidate(t *testing.T) {
	var (
		firstResource  = schema.GroupResource{Group: "one", Resource: "first"}
		secondResource = schema.GroupResource{Group: "one", Resource: "second"}
		legacyResource = schema.GroupResource{Resource: "legacy"}
	)

	tests := []struct {
		name         string
		set          []string
		noExplicit   bool
		requirements FeatureGateAPIRequirements
		served       sets.Set[schema.GroupResource]
		wantErr      bool
		wantContains []string
		wantWarnings []string
	}{
		{
			name:         "no requirements declared",
			set:          []string{"FeatureA=true"},
			requirements: FeatureGateAPIRequirements{},
			served:       sets.New[schema.GroupResource](),
		},
		{
			name:         "disabled feature with unserved requirement is ignored",
			requirements: FeatureGateAPIRequirements{featureA: {firstResource}},
			served:       sets.New[schema.GroupResource](),
		},
		{
			name:         "explicitly enabled feature with served requirement",
			set:          []string{"FeatureA=true"},
			requirements: FeatureGateAPIRequirements{featureA: {firstResource}},
			served:       sets.New(firstResource),
		},
		{
			name:         "explicitly enabled feature with unserved requirement",
			set:          []string{"FeatureA=true"},
			requirements: FeatureGateAPIRequirements{featureA: {firstResource}},
			served:       sets.New[schema.GroupResource](),
			wantErr:      true,
			wantContains: []string{"FeatureA is enabled", "first.one"},
		},
		{
			name:         "explicitly enabled feature reports only the unserved of several requirements",
			set:          []string{"FeatureA=true"},
			requirements: FeatureGateAPIRequirements{featureA: {firstResource, secondResource}},
			served:       sets.New(firstResource),
			wantErr:      true,
			wantContains: []string{"second.one"},
		},
		{
			name:         "legacy group resource renders without a group suffix",
			set:          []string{"FeatureA=true"},
			requirements: FeatureGateAPIRequirements{featureA: {legacyResource}},
			served:       sets.New[schema.GroupResource](),
			wantErr:      true,
			wantContains: []string{"not served: legacy;"},
		},
		{
			name: "only enabled features are checked",
			set:  []string{"FeatureB=true"},
			requirements: FeatureGateAPIRequirements{
				featureA: {firstResource},
				featureB: {secondResource},
			},
			served: sets.New(secondResource),
		},
		{
			name: "several failing features are all reported",
			set:  []string{"FeatureA=true", "FeatureB=true"},
			requirements: FeatureGateAPIRequirements{
				featureA: {firstResource},
				featureB: {secondResource},
			},
			served:       sets.New[schema.GroupResource](),
			wantErr:      true,
			wantContains: []string{"FeatureA is enabled", "FeatureB is enabled"},
		},
		{
			name:         "feature enabled by default with served requirement",
			requirements: FeatureGateAPIRequirements{featureDefaultOn: {firstResource}},
			served:       sets.New(firstResource),
		},
		{
			name:         "feature enabled by default with unserved requirement only warns",
			requirements: FeatureGateAPIRequirements{featureDefaultOn: {firstResource}},
			served:       sets.New[schema.GroupResource](),
			wantWarnings: []string{"FeatureDefaultOn is enabled by default", "first.one", "inactive"},
		},
		{
			name:         "feature enabled by default and explicitly enabled with unserved requirement fails",
			set:          []string{"FeatureDefaultOn=true"},
			requirements: FeatureGateAPIRequirements{featureDefaultOn: {firstResource}},
			served:       sets.New[schema.GroupResource](),
			wantErr:      true,
			wantContains: []string{"FeatureDefaultOn is enabled", "first.one"},
		},
		{
			name:         "feature enabled by default but explicitly disabled is ignored",
			set:          []string{"FeatureDefaultOn=false"},
			requirements: FeatureGateAPIRequirements{featureDefaultOn: {firstResource}},
			served:       sets.New[schema.GroupResource](),
		},
		{
			name: "explicit failure is an error while default failure is a warning",
			set:  []string{"FeatureA=true"},
			requirements: FeatureGateAPIRequirements{
				featureA:         {firstResource},
				featureDefaultOn: {secondResource},
			},
			served:       sets.New[schema.GroupResource](),
			wantErr:      true,
			wantContains: []string{"FeatureA is enabled"},
			wantWarnings: []string{"FeatureDefaultOn is enabled by default"},
		},
		{
			name:         "gate that cannot report explicitness never fails",
			set:          []string{"FeatureA=true"},
			noExplicit:   true,
			requirements: FeatureGateAPIRequirements{featureA: {firstResource}},
			served:       sets.New[schema.GroupResource](),
			wantWarnings: []string{"FeatureA is enabled by default", "first.one"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gate featuregate.FeatureGate = newTestGate(t, tc.set...)
			if tc.noExplicit {
				gate = gateWithoutExplicitness{gate}
			}

			warnings, err := tc.requirements.Validate(gate, tc.served)
			if tc.wantErr != (err != nil) {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tc.wantErr)
			}
			if err != nil {
				for _, want := range tc.wantContains {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error %q does not contain %q", err.Error(), want)
					}
				}
			}

			if len(tc.wantWarnings) == 0 && len(warnings) > 0 {
				t.Errorf("unexpected warnings: %v", warnings)
			}
			joined := strings.Join(warnings, "\n")
			for _, want := range tc.wantWarnings {
				if !strings.Contains(joined, want) {
					t.Errorf("warnings %q do not contain %q", joined, want)
				}
			}
		})
	}
}

func TestFeatureGateAPIRequirementsGatesByResource(t *testing.T) {
	var (
		firstResource  = schema.GroupResource{Group: "one", Resource: "first"}
		secondResource = schema.GroupResource{Group: "one", Resource: "second"}
	)

	tests := []struct {
		name         string
		requirements FeatureGateAPIRequirements
		want         map[schema.GroupResource][]featuregate.Feature
	}{
		{
			name:         "no requirements",
			requirements: FeatureGateAPIRequirements{},
			want:         map[schema.GroupResource][]featuregate.Feature{},
		},
		{
			name:         "one gate with several resources",
			requirements: FeatureGateAPIRequirements{featureA: {firstResource, secondResource}},
			want: map[schema.GroupResource][]featuregate.Feature{
				firstResource:  {featureA},
				secondResource: {featureA},
			},
		},
		{
			name: "resource shared by several gates requires all of them in gate order",
			requirements: FeatureGateAPIRequirements{
				featureB: {firstResource},
				featureA: {firstResource, secondResource},
			},
			want: map[schema.GroupResource][]featuregate.Feature{
				firstResource:  {featureA, featureB},
				secondResource: {featureA},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.requirements.gatesByResource(); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("gatesByResource() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestFeatureGateAPIRequirementsValidateExplicitlyEnabledAPIs(t *testing.T) {
	var (
		gaVersion    = schema.GroupVersion{Group: "one", Version: "v1"}
		betaVersion  = schema.GroupVersion{Group: "one", Version: "v1beta1"}
		alphaVersion = schema.GroupVersion{Group: "one", Version: "v1alpha1"}

		// workloads and podgroups need featureA; composites need featureA and featureB;
		// plain is not gated.
		workloads  = schema.GroupResource{Group: "one", Resource: "workloads"}
		podgroups  = schema.GroupResource{Group: "one", Resource: "podgroups"}
		composites = schema.GroupResource{Group: "one", Resource: "composites"}
	)

	requirements := FeatureGateAPIRequirements{
		featureA: {workloads, podgroups, composites},
		featureB: {composites},
	}

	registered := sets.New(
		gaVersion.WithResource("plain"),
		betaVersion.WithResource("workloads"),
		betaVersion.WithResource("podgroups"),
		betaVersion.WithResource("plain"),
		alphaVersion.WithResource("workloads"),
		alphaVersion.WithResource("composites"),
	)

	tests := []struct {
		name         string
		set          []string
		config       func(*ResourceConfig)
		wantErr      bool
		wantContains []string
		wantMissing  []string
		wantOrdered  bool
	}{
		{
			name:   "nothing enabled",
			config: func(*ResourceConfig) {},
		},
		{
			name:         "resource explicitly enabled without its gate",
			config:       func(c *ResourceConfig) { c.ExplicitlyEnableResources(betaVersion.WithResource("workloads")) },
			wantErr:      true,
			wantContains: []string{"workloads.one", "one/v1beta1/workloads", "FeatureA"},
			wantMissing:  []string{"podgroups.one"},
		},
		{
			name:         "resource explicitly enabled with its gate explicitly disabled",
			set:          []string{"FeatureA=false"},
			config:       func(c *ResourceConfig) { c.ExplicitlyEnableResources(betaVersion.WithResource("workloads")) },
			wantErr:      true,
			wantContains: []string{"workloads.one", "FeatureA"},
		},
		{
			name:   "resource explicitly enabled with its gate enabled",
			set:    []string{"FeatureA=true"},
			config: func(c *ResourceConfig) { c.ExplicitlyEnableResources(betaVersion.WithResource("workloads")) },
		},
		{
			name:   "version explicitly enabled without its gate is not checked",
			config: func(c *ResourceConfig) { c.ExplicitlyEnableVersions(betaVersion) },
		},
		{
			name:   "version enabled by default without its gate is not checked",
			config: func(c *ResourceConfig) { c.EnableVersions(betaVersion) },
		},
		{
			name:   "resource enabled by default without its gate is not checked",
			config: func(c *ResourceConfig) { c.EnableResources(betaVersion.WithResource("workloads")) },
		},
		{
			// A later version-level preference drops the resource-level ones.
			name: "resource explicitly enabled but its version explicitly disabled is not checked",
			config: func(c *ResourceConfig) {
				c.ExplicitlyEnableResources(betaVersion.WithResource("workloads"))
				c.ExplicitlyDisableVersions(betaVersion)
			},
		},
		{
			name:         "resource requiring several gates reports only the disabled ones",
			set:          []string{"FeatureA=true"},
			config:       func(c *ResourceConfig) { c.ExplicitlyEnableResources(alphaVersion.WithResource("composites")) },
			wantErr:      true,
			wantContains: []string{"composites.one", "disabled: FeatureB;"},
		},
		{
			name:   "resource explicitly enabled in a version that does not carry it",
			config: func(c *ResourceConfig) { c.ExplicitlyEnableResources(gaVersion.WithResource("workloads")) },
		},
		{
			name: "errors are sorted by resource and list every version that asked",
			config: func(c *ResourceConfig) {
				c.ExplicitlyEnableResources(
					betaVersion.WithResource("workloads"),
					alphaVersion.WithResource("workloads"),
					betaVersion.WithResource("podgroups"),
					alphaVersion.WithResource("composites"),
				)
			},
			wantErr: true,
			wantContains: []string{
				"composites.one was explicitly enabled with --runtime-config (one/v1alpha1/composites), but requires feature gates that are disabled: FeatureA, FeatureB;",
				"podgroups.one was explicitly enabled with --runtime-config (one/v1beta1/podgroups)",
				"workloads.one was explicitly enabled with --runtime-config (one/v1alpha1/workloads, one/v1beta1/workloads)",
			},
			wantOrdered: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gate := newTestGate(t, tc.set...)
			cfg := NewResourceConfig()
			tc.config(cfg)

			err := requirements.ValidateExplicitlyEnabledAPIs(gate, cfg, registered)
			if tc.wantErr != (err != nil) {
				t.Fatalf("ValidateExplicitlyEnabledAPIs() error = %v, wantErr %v", err, tc.wantErr)
			}
			if err == nil {
				return
			}
			for _, want := range tc.wantContains {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not contain %q", err.Error(), want)
				}
			}
			for _, missing := range tc.wantMissing {
				if strings.Contains(err.Error(), missing) {
					t.Errorf("error %q unexpectedly contains %q", err.Error(), missing)
				}
			}

			if !tc.wantOrdered {
				return
			}
			last := -1
			for _, want := range tc.wantContains {
				if idx := strings.Index(err.Error(), want); idx < last {
					t.Errorf("error %q reports %q out of order", err.Error(), want)
				} else {
					last = idx
				}
			}
		})
	}
}
