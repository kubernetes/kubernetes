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

package workloadbuilder

import (
	"context"
	"fmt"
	"strings"
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

// defaultGangMinCount fills gang MinCount when Gang is selected but unset,
// mirroring the Job controller's defaultMinCountForJob from KEP-6089.
func defaultGangMinCount(n int32) WorkloadItemFunc {
	return func(item *WorkloadItem) {
		p := item.ResolvedConfig.Policy
		if p != nil && p.Gang != nil && p.Gang.MinCount == nil {
			p.Gang.MinCount = ptr.To(n)
		}
	}
}

// childrenRoot builds a root with n leaf children, each using Basic scheduling.
func childrenRoot(n int) *WorkloadItem {
	root := &WorkloadItem{Name: "root"}
	for i := 0; i < n; i++ {
		root.Children = append(root.Children, &WorkloadItem{
			Name:          fmt.Sprintf("child-%d", i),
			DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
		})
	}
	return root
}

func TestBuild(t *testing.T) {
	tests := []struct {
		name    string
		root    *WorkloadItem
		owner   *metav1.OwnerReference
		wantErr bool
		verify  func(t *testing.T, wl *schedulingv1alpha3.Workload, root *WorkloadItem)
	}{
		{
			name: "basic policy",
			root: &WorkloadItem{
				Name:          "test-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				if len(wl.Spec.PodGroupTemplates) != 1 {
					t.Fatalf("expected 1 template, got %d", len(wl.Spec.PodGroupTemplates))
				}
				tmpl := wl.Spec.PodGroupTemplates[0]
				if tmpl.SchedulingPolicy.Basic == nil {
					t.Error("expected Basic policy to be set")
				}
				if tmpl.SchedulingPolicy.Gang != nil {
					t.Error("Gang policy should not be set")
				}
			},
		},
		{
			name: "gang policy from default",
			root: &WorkloadItem{
				Name:          "gang-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: ptr.To[int32](4)}}},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				gang := wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang
				if gang == nil {
					t.Fatal("expected Gang policy")
				}
				if gang.MinCount != 4 {
					t.Errorf("expected MinCount=4, got %d", gang.MinCount)
				}
			},
		},
		{
			name: "gang minCount defaulted by callback",
			root: &WorkloadItem{
				Name:          "gang-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig:    &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
				Callbacks:     []WorkloadItemFunc{defaultGangMinCount(8)},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				gang := wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang
				if gang == nil {
					t.Fatal("expected Gang policy")
				}
				if gang.MinCount != 8 {
					t.Errorf("expected MinCount=8 (from callback), got %d", gang.MinCount)
				}
			},
		},
		{
			name: "user gang overrides default basic",
			root: &WorkloadItem{
				Name:          "override-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig:    &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: ptr.To[int32](2)}}},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				gang := wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang
				if gang == nil {
					t.Fatal("expected Gang policy from user override")
				}
				if gang.MinCount != 2 {
					t.Errorf("expected user MinCount=2, got %d", gang.MinCount)
				}
			},
		},
		{
			name: "field-by-field merge of default and user config",
			root: &WorkloadItem{
				Name: "merge-job",
				DefaultConfig: &SchedulingConfig{
					Policy:         &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}},
					DisruptionMode: &DisruptionMode{Single: &SingleDisruptionMode{}},
				},
				UserConfig: &SchedulingConfig{
					Constraints: &SchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
					},
				},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				tmpl := wl.Spec.PodGroupTemplates[0]
				if tmpl.SchedulingPolicy.Basic == nil {
					t.Error("expected default Basic policy to survive the merge")
				}
				if tmpl.DisruptionMode == nil || tmpl.DisruptionMode.Single == nil {
					t.Error("expected default Single disruption mode to survive the merge")
				}
				if tmpl.SchedulingConstraints == nil || len(tmpl.SchedulingConstraints.Topology) != 1 {
					t.Error("expected user-provided topology constraint")
				}
			},
		},
		{
			name: "topology constraints",
			root: &WorkloadItem{
				Name:          "topo-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					Constraints: &SchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
					},
				},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				tmpl := wl.Spec.PodGroupTemplates[0]
				if tmpl.SchedulingConstraints == nil || len(tmpl.SchedulingConstraints.Topology) != 1 {
					t.Fatal("expected 1 topology constraint")
				}
				if tmpl.SchedulingConstraints.Topology[0].Key != "topology.kubernetes.io/zone" {
					t.Errorf("unexpected topology key: %s", tmpl.SchedulingConstraints.Topology[0].Key)
				}
			},
		},
		{
			name: "disruption mode all",
			root: &WorkloadItem{
				Name:          "disruption-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig:    &SchedulingConfig{DisruptionMode: &DisruptionMode{All: &AllDisruptionMode{}}},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				tmpl := wl.Spec.PodGroupTemplates[0]
				if tmpl.DisruptionMode == nil || tmpl.DisruptionMode.All == nil {
					t.Error("expected All disruption mode")
				}
			},
		},
		{
			name: "resource claims",
			root: &WorkloadItem{
				Name:          "dra-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					ResourceClaims: []ResourceClaim{{Name: "gpu", ResourceClaimName: ptr.To("my-claim")}},
				},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				tmpl := wl.Spec.PodGroupTemplates[0]
				if len(tmpl.ResourceClaims) != 1 {
					t.Fatalf("expected 1 resource claim, got %d", len(tmpl.ResourceClaims))
				}
				if tmpl.ResourceClaims[0].Name != "gpu" {
					t.Errorf("expected claim name 'gpu', got %q", tmpl.ResourceClaims[0].Name)
				}
			},
		},
		{
			name: "resource claim missing both names fails",
			root: &WorkloadItem{
				Name:          "dra-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					ResourceClaims: []ResourceClaim{{Name: "gpu"}},
				},
			},
			wantErr: true,
		},
		{
			name: "resource claim with both names fails",
			root: &WorkloadItem{
				Name:          "dra-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					ResourceClaims: []ResourceClaim{{Name: "gpu", ResourceClaimName: ptr.To("a"), ResourceClaimTemplateName: ptr.To("b")}},
				},
			},
			wantErr: true,
		},
		{
			name: "resource claim empty name fails",
			root: &WorkloadItem{
				Name:          "dra-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					ResourceClaims: []ResourceClaim{{Name: "", ResourceClaimName: ptr.To("a")}},
				},
			},
			wantErr: true,
		},
		{
			name: "resource claim duplicate name fails",
			root: &WorkloadItem{
				Name:          "dra-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					ResourceClaims: []ResourceClaim{
						{Name: "dup", ResourceClaimName: ptr.To("a")},
						{Name: "dup", ResourceClaimName: ptr.To("b")},
					},
				},
			},
			wantErr: true,
		},
		{
			name: "owner reference and controllerRef parse APIGroup",
			root: &WorkloadItem{
				Name:          "owned-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
			},
			owner: &metav1.OwnerReference{
				APIVersion: "batch/v1",
				Kind:       "Job",
				Name:       "my-job",
				UID:        "abc-123",
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				if len(wl.OwnerReferences) != 1 {
					t.Fatalf("expected 1 owner reference, got %d", len(wl.OwnerReferences))
				}
				if wl.OwnerReferences[0].Name != "my-job" {
					t.Errorf("expected owner name 'my-job', got %q", wl.OwnerReferences[0].Name)
				}
				if wl.Spec.ControllerRef == nil || wl.Spec.ControllerRef.Name != "my-job" {
					t.Fatal("expected controllerRef to be set")
				}
				if wl.Spec.ControllerRef.APIGroup != "batch" || wl.Spec.ControllerRef.Kind != "Job" {
					t.Errorf("unexpected controllerRef: %+v", wl.Spec.ControllerRef)
				}
			},
		},
		{
			name: "empty leaf name fails",
			root: &WorkloadItem{
				Name:       "",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
			},
			wantErr: true,
		},
		{
			name: "multi-level tree flattens to leaf templates",
			root: &WorkloadItem{
				Name:          "jobset",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: ptr.To[int32](2)}}},
				Children: []*WorkloadItem{
					{
						Name:          "driver",
						DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
					},
					{
						Name:          "workers",
						DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: ptr.To[int32](4)}}},
					},
				},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				if len(wl.Spec.PodGroupTemplates) != 2 {
					t.Fatalf("expected 2 templates from children, got %d", len(wl.Spec.PodGroupTemplates))
				}
				if wl.Spec.PodGroupTemplates[0].Name != "driver" {
					t.Errorf("expected first template name 'driver', got %q", wl.Spec.PodGroupTemplates[0].Name)
				}
				if wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Basic == nil {
					t.Error("expected driver to use Basic policy")
				}
				if wl.Spec.PodGroupTemplates[1].Name != "workers" {
					t.Errorf("expected second template name 'workers', got %q", wl.Spec.PodGroupTemplates[1].Name)
				}
				gang := wl.Spec.PodGroupTemplates[1].SchedulingPolicy.Gang
				if gang == nil || gang.MinCount != 4 {
					t.Error("expected workers to use Gang policy with MinCount=4")
				}
			},
		},
		{
			name: "nil configs default to basic",
			root: &WorkloadItem{Name: "bare"},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				if wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Basic == nil {
					t.Error("expected default Basic policy when no config is provided")
				}
			},
		},
		{
			name: "callback performs arbitrary adjustment",
			root: &WorkloadItem{
				Name:          "adjusted",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: ptr.To[int32](4)}}},
				Callbacks: []WorkloadItemFunc{
					func(item *WorkloadItem) {
						if g := item.ResolvedConfig.Policy.Gang; g != nil && g.MinCount != nil {
							*g.MinCount *= 3
						}
					},
				},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				if got := wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount; got != 12 {
					t.Errorf("expected callback-adjusted MinCount=12, got %d", got)
				}
			},
		},
		{
			name: "resolution does not mutate caller inputs",
			root: &WorkloadItem{
				Name:       "gang-job",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
				Callbacks:  []WorkloadItemFunc{defaultGangMinCount(16)},
			},
			verify: func(t *testing.T, _ *schedulingv1alpha3.Workload, root *WorkloadItem) {
				if root.UserConfig.Policy.Gang.MinCount != nil {
					t.Errorf("callback mutated the caller's UserConfig; MinCount should remain nil, got %d", *root.UserConfig.Policy.Gang.MinCount)
				}
				if root.ResolvedConfig == nil || root.ResolvedConfig.Policy.Gang.MinCount == nil {
					t.Fatal("expected ResolvedConfig to carry the defaulted MinCount")
				}
				if *root.ResolvedConfig.Policy.Gang.MinCount != 16 {
					t.Errorf("expected ResolvedConfig MinCount=16, got %d", *root.ResolvedConfig.Policy.Gang.MinCount)
				}
			},
		},
		{
			name: "gang missing minCount fails",
			root: &WorkloadItem{
				Name:       "gang-job",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
			},
			wantErr: true,
		},
		{
			name: "duplicate template names fail",
			root: &WorkloadItem{
				Name: "jobset",
				Children: []*WorkloadItem{
					{Name: "dup", DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}}},
					{Name: "dup", DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}}},
				},
			},
			wantErr: true,
		},
		{
			name:    "too many templates fail",
			root:    childrenRoot(schedulingv1alpha3.WorkloadMaxPodGroupTemplates + 1),
			wantErr: true,
		},
		{
			name:    "nil root fails",
			root:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wl, err := NewBuilder(tt.root).Build(context.Background(), "wl", "ns", tt.owner)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.verify != nil {
				tt.verify(t, wl, tt.root)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name     string
		root     *WorkloadItem
		wantErrs bool
		wantType field.ErrorType
	}{
		{
			name:     "nil root is required",
			root:     nil,
			wantErrs: true,
			wantType: field.ErrorTypeRequired,
		},
		{
			name: "basic config is valid",
			root: &WorkloadItem{
				Name:          "job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
			},
		},
		{
			name: "gang minCount defaulted by callback is valid",
			root: &WorkloadItem{
				Name:       "job",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
				Callbacks:  []WorkloadItemFunc{defaultGangMinCount(4)},
			},
		},
		{
			name: "gang minCount unset without callback is invalid",
			root: &WorkloadItem{
				Name:       "job",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
			},
			wantErrs: true,
			wantType: field.ErrorTypeRequired,
		},
		{
			name: "resource claim missing both names is invalid",
			root: &WorkloadItem{
				Name:          "job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					ResourceClaims: []ResourceClaim{{Name: "gpu"}},
				},
			},
			wantErrs: true,
			wantType: field.ErrorTypeInvalid,
		},
		{
			name: "duplicate resource claim names is invalid",
			root: &WorkloadItem{
				Name:          "job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					ResourceClaims: []ResourceClaim{
						{Name: "dup", ResourceClaimName: ptr.To("a")},
						{Name: "dup", ResourceClaimName: ptr.To("b")},
					},
				},
			},
			wantErrs: true,
			wantType: field.ErrorTypeDuplicate,
		},
	}

	fldPath := field.NewPath("spec", "scheduling")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := NewBuilder(tt.root).Validate(fldPath)
			if tt.wantErrs {
				if len(errs) == 0 {
					t.Fatal("expected errors, got none")
				}
				if tt.wantType != "" && errs[0].Type != tt.wantType {
					t.Errorf("expected error type %s, got %s", tt.wantType, errs[0].Type)
				}
				for _, e := range errs {
					if !strings.HasPrefix(e.Field, "spec.scheduling") {
						t.Errorf("expected error rooted at spec.scheduling, got %q", e.Field)
					}
				}
				return
			}
			if len(errs) != 0 {
				t.Fatalf("expected no errors, got %v", errs)
			}
		})
	}
}

// TestValidateAgreesWithBuild guarantees the contract the API server relies on:
// a configuration Validate rejects is one Build also rejects, and vice versa.
func TestValidateAgreesWithBuild(t *testing.T) {
	roots := map[string]*WorkloadItem{
		"valid basic":      {Name: "job", DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}}},
		"gang no minCount": {Name: "job", UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}}},
		"bad claim":        {Name: "job", UserConfig: &SchedulingConfig{ResourceClaims: []ResourceClaim{{Name: "gpu"}}}},
		"empty name":       {UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}}},
	}

	for name, root := range roots {
		t.Run(name, func(t *testing.T) {
			validateErr := len(NewBuilder(root).Validate(field.NewPath("spec", "scheduling"))) > 0
			_, err := NewBuilder(root).Build(context.Background(), "wl", "ns", nil)
			buildErr := err != nil
			if validateErr != buildErr {
				t.Errorf("Validate/Build disagree: validateErr=%v buildErr=%v", validateErr, buildErr)
			}
		})
	}
}
