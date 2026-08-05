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
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestMapWorkloadInput(t *testing.T) {
	tests := []struct {
		name        string
		policy      *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
		constraints *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints
		disruption  *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
		claims      []schedulingv1alpha3.WorkloadPodGroupResourceClaim
		check       func(t *testing.T, cfg *SchedulingConfig)
	}{
		{
			name: "all nil leaves fields unset",
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy != nil || cfg.Constraints != nil ||
					cfg.DisruptionMode != nil ||
					cfg.ResourceClaims != nil {
					t.Errorf("expected all fields nil, got %+v", cfg)
				}
			},
		},
		{
			name:   "empty policy maps to nil so the default survives",
			policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy != nil {
					t.Errorf("expected nil policy for empty input, got %+v", cfg.Policy)
				}
			},
		},
		{
			name:        "empty constraints maps to nil so the default survives",
			constraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Constraints != nil {
					t.Errorf("expected nil constraints for empty input, got %+v", cfg.Constraints)
				}
			},
		},
		{
			name:       "empty disruption maps to nil so the default survives",
			disruption: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.DisruptionMode != nil {
					t.Errorf("expected nil disruption for empty input, got %+v", cfg.DisruptionMode)
				}
			},
		},
		{
			name:   "basic policy",
			policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy == nil || cfg.Policy.Basic == nil {
					t.Error("expected Basic policy")
				}
			},
		},
		{
			name:   "gang policy with minCount",
			policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](4)}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy == nil || cfg.Policy.Gang == nil {
					t.Fatal("expected Gang policy")
				}
				if cfg.Policy.Gang.MinCount == nil || *cfg.Policy.Gang.MinCount != 4 {
					t.Errorf("expected MinCount=4, got %v", cfg.Policy.Gang.MinCount)
				}
			},
		},
		{
			name:   "gang policy preserves nil minCount",
			policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy == nil || cfg.Policy.Gang == nil {
					t.Fatal("expected Gang policy")
				}
				if cfg.Policy.Gang.MinCount != nil {
					t.Errorf("expected nil MinCount to be preserved, got %d", *cfg.Policy.Gang.MinCount)
				}
			},
		},
		{
			name: "topology constraints",
			constraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
				Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/rack"}},
			},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Constraints == nil || len(cfg.Constraints.Topology) != 1 {
					t.Fatal("expected 1 topology constraint")
				}
				if cfg.Constraints.Topology[0].Key != "topology.kubernetes.io/rack" {
					t.Errorf("unexpected key: %s", cfg.Constraints.Topology[0].Key)
				}
			},
		},
		{
			name:       "disruption mode all",
			disruption: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.DisruptionMode == nil || cfg.DisruptionMode.All == nil {
					t.Error("expected All disruption mode")
				}
			},
		},
		{
			name:   "resource claims",
			claims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{{Name: "gpu", ResourceClaimName: new("shared-gpu")}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if len(cfg.ResourceClaims) != 1 {
					t.Fatalf("expected 1 claim, got %d", len(cfg.ResourceClaims))
				}
				if cfg.ResourceClaims[0].Name != "gpu" {
					t.Errorf("expected claim name 'gpu', got %q", cfg.ResourceClaims[0].Name)
				}
				if cfg.ResourceClaims[0].ResourceClaimName == nil || *cfg.ResourceClaims[0].ResourceClaimName != "shared-gpu" {
					t.Error("expected ResourceClaimName 'shared-gpu'")
				}
			},
		},
		{
			name:   "all fields populated",
			policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](2)}},
			constraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
				Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "zone"}},
			},
			disruption: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{Single: &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{}},
			claims:     []schedulingv1alpha3.WorkloadPodGroupResourceClaim{{Name: "net", ResourceClaimTemplateName: new("tpl")}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy == nil || cfg.Policy.Gang == nil {
					t.Error("expected Gang policy")
				}
				if cfg.Constraints == nil || len(cfg.Constraints.Topology) != 1 {
					t.Error("expected topology")
				}
				if cfg.DisruptionMode == nil || cfg.DisruptionMode.Single == nil {
					t.Error("expected Single disruption")
				}
				if len(cfg.ResourceClaims) != 1 || cfg.ResourceClaims[0].ResourceClaimTemplateName == nil {
					t.Error("expected resource claim with template name")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := mapWorkloadInput(WorkloadInput{
				Policy:         PolicyInput{PodGroupData: tt.policy},
				Constraints:    ConstraintsInput{PodGroupData: tt.constraints},
				DisruptionMode: DisruptionModeInput{PodGroupData: tt.disruption},
				ResourceClaims: ResourceClaimsInput{PodGroupData: tt.claims},
			})
			if cfg == nil {
				t.Fatal("expected non-nil config")
			}
			tt.check(t, cfg)
		})
	}
}

func TestMapCompositeGroupInput(t *testing.T) {
	inputPolicy := &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{
		Gang: &schedulingv1alpha3.WorkloadCompositePodGroupGangSchedulingPolicy{
			MinGroupCount: ptr.To[int32](2),
		},
	}

	tests := []struct {
		name        string
		policy      *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy
		constraints *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingConstraints
		disruption  *schedulingv1alpha3.WorkloadCompositePodGroupDisruptionMode
		check       func(t *testing.T, cfg *SchedulingConfig)
	}{
		{
			name: "nil policy leaves Policy unset",
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy != nil {
					t.Errorf("expected nil policy for nil input, got %+v", cfg.Policy)
				}
			},
		},
		{
			name:        "composite topology constraints",
			constraints: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingConstraints{Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "zone"}}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Constraints == nil || len(cfg.Constraints.Topology) != 1 || cfg.Constraints.Topology[0].Key != "zone" {
					t.Errorf("expected 1 composite topology constraint, got %+v", cfg.Constraints)
				}
			},
		},
		{
			name:       "composite disruption mode all",
			disruption: &schedulingv1alpha3.WorkloadCompositePodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadCompositePodGroupAllDisruptionMode{}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.DisruptionMode == nil || cfg.DisruptionMode.All == nil {
					t.Error("expected composite All disruption mode")
				}
			},
		},
		{
			name:   "empty policy maps to nil so the default survives",
			policy: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy != nil {
					t.Errorf("expected nil policy for empty input, got %+v", cfg.Policy)
				}
			},
		},
		{
			name:   "basic policy",
			policy: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadCompositePodGroupBasicSchedulingPolicy{}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy == nil || cfg.Policy.Basic == nil {
					t.Error("expected Basic policy")
				}
			},
		},
		{
			name:   "gang policy carries minGroupCount in the IR MinCount",
			policy: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadCompositePodGroupGangSchedulingPolicy{MinGroupCount: ptr.To[int32](3)}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy == nil || cfg.Policy.Gang == nil {
					t.Fatal("expected Gang policy")
				}
				if cfg.Policy.Gang.MinCount == nil || *cfg.Policy.Gang.MinCount != 3 {
					t.Errorf("expected MinCount=3 (from minGroupCount), got %v", cfg.Policy.Gang.MinCount)
				}
			},
		},
		{
			name:   "gang policy preserves nil minGroupCount",
			policy: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadCompositePodGroupGangSchedulingPolicy{}},
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy == nil || cfg.Policy.Gang == nil {
					t.Fatal("expected Gang policy")
				}
				if cfg.Policy.Gang.MinCount != nil {
					t.Errorf("expected nil MinCount to be preserved, got %d", *cfg.Policy.Gang.MinCount)
				}
			},
		},
		{
			name:   "gang policy copies minGroupCount instead of aliasing the input",
			policy: inputPolicy,
			check: func(t *testing.T, cfg *SchedulingConfig) {
				if cfg.Policy == nil || cfg.Policy.Gang == nil || cfg.Policy.Gang.MinCount == nil {
					t.Fatal("expected Gang policy with MinCount")
				}
				// Mutating the resolved value must not leak back into the caller's
				// building block.
				*cfg.Policy.Gang.MinCount = 99
				if *inputPolicy.Gang.MinGroupCount != 2 {
					t.Errorf("mapping aliased the input; minGroupCount mutated to %d", *inputPolicy.Gang.MinGroupCount)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := mapCompositeGroupInput(WorkloadInput{
				Policy:         PolicyInput{CompositePodGroupData: tt.policy},
				Constraints:    ConstraintsInput{CompositePodGroupData: tt.constraints},
				DisruptionMode: DisruptionModeInput{CompositePodGroupData: tt.disruption},
			})
			if cfg == nil {
				t.Fatal("expected non-nil config")
			}
			// resourceClaims is leaf-only; a composite never maps it.
			if cfg.ResourceClaims != nil {
				t.Errorf("expected no resourceClaims for a composite, got %+v", cfg.ResourceClaims)
			}
			tt.check(t, cfg)
		})
	}
}

// TestMapConfigEndToEnd exercises the documented controller integration path:
// map the public building blocks into the IR, then compile via the builder, for
// both the leaf and composite group shapes.
func TestMapConfigEndToEnd(t *testing.T) {
	tests := []struct {
		name   string
		root   *WorkloadItem
		verify func(t *testing.T, wl *schedulingv1beta1.Workload)
	}{
		{
			name: "leaf group maps and compiles every building block",
			root: &WorkloadItem{
				Name:          "job-root",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				Input: WorkloadInput{
					Policy: PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{},
					}},
					Constraints: ConstraintsInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
					}},
					DisruptionMode: DisruptionModeInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
						All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
					}},
				},
				Callbacks: []SchedulingConfigFunc{defaultGangMinCount(4)},
			},
			verify: func(t *testing.T, wl *schedulingv1beta1.Workload) {
				tmpl := wl.Spec.PodGroupTemplates[0]
				if tmpl.SchedulingPolicy.Gang == nil || tmpl.SchedulingPolicy.Gang.MinCount != 4 {
					t.Error("expected Gang policy with defaulted MinCount=4")
				}
				if tmpl.SchedulingConstraints == nil || len(tmpl.SchedulingConstraints.Topology) != 1 {
					t.Error("expected topology constraint to pass through")
				}
				if tmpl.DisruptionMode == nil || tmpl.DisruptionMode.All == nil {
					t.Error("expected All disruption mode to pass through")
				}
			},
		},
		{
			name: "composite group maps and compiles the group-of-groups policy",
			root: &WorkloadItem{
				Name: "cpg-root",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{
					Basic: &BasicSchedulingPolicy{},
				},
				},
				Input: WorkloadInput{
					Policy: PolicyInput{CompositePodGroupData: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha3.WorkloadCompositePodGroupGangSchedulingPolicy{},
					}},
				},
				Callbacks: []SchedulingConfigFunc{defaultGangMinCount(4)},
				Children: []*WorkloadItem{
					{Name: "workers", DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}}},
				},
			},
			verify: func(t *testing.T, wl *schedulingv1beta1.Workload) {
				cpg := wl.Spec.CompositePodGroupTemplates[0]
				if cpg.SchedulingPolicy.Gang == nil || cpg.SchedulingPolicy.Gang.MinGroupCount != 4 {
					t.Errorf("expected composite Gang policy with defaulted MinGroupCount=4, got %+v", cpg.SchedulingPolicy)
				}
				if len(cpg.PodGroupTemplates) != 1 || cpg.PodGroupTemplates[0].Name != "workers" {
					t.Errorf("expected the leaf child to compile, got %+v", cpg.PodGroupTemplates)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wl, err := NewBuilder(tt.root, BuildOptions{
				Name:      "job",
				Namespace: "ns",
				Owner: &metav1.OwnerReference{
					APIVersion: "batch/v1",
					Kind:       "Job",
					Name:       "test-job",
					UID:        "12345",
				},
			}).BuildWorkload()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tt.verify(t, wl)
		})
	}
}
