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
	"reflect"
	"strings"
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

// defaultGangMinCount fills gang MinCount when Gang is selected but unset,
// mirroring the Job controller's defaultMinCountForJob from KEP-6089.
func defaultGangMinCount(n int32) SchedulingConfigFunc {
	return func(cfg *SchedulingConfig) {
		p := cfg.Policy
		if p != nil && p.Gang != nil && p.Gang.MinCount == nil {
			p.Gang.MinCount = new(n)
		}
	}
}

func TestBuildWorkload(t *testing.T) {
	// Tracks whether the empty-leaf-name case's callback ran, proving the name
	// check short-circuits before callbacks.
	emptyNameCallbackRan := false
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
				Input:         WorkloadInput{Policy: PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}}}},
				Callbacks:     []SchedulingConfigFunc{defaultGangMinCount(8)},
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
				Input:         WorkloadInput{Policy: PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](2)}}}},
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
				Input: WorkloadInput{
					Constraints: ConstraintsInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
					}},
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
				Input: WorkloadInput{
					Constraints: ConstraintsInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
					}},
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
				Input:         WorkloadInput{DisruptionMode: DisruptionModeInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}}}},
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
				Input: WorkloadInput{
					ResourceClaims: ResourceClaimsInput{PodGroupData: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{{Name: "gpu", ResourceClaimName: new("my-claim")}}},
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
			name: "priorityClassName propagates to template",
			root: &WorkloadItem{
				Name: "prio-job",
				DefaultConfig: &SchedulingConfig{
					Policy:            &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}},
					PriorityClassName: "system-node-critical",
				},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, _ *WorkloadItem) {
				if got := wl.Spec.PodGroupTemplates[0].PriorityClassName; got != "system-node-critical" {
					t.Errorf("expected priorityClassName copied to template, got %q", got)
				}
			},
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
			name: "empty leaf name fails before resolution and callbacks",
			root: &WorkloadItem{
				Name:  "",
				Input: WorkloadInput{Policy: PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}}}},
				Callbacks: []SchedulingConfigFunc{
					func(*SchedulingConfig) {
						emptyNameCallbackRan = true
					},
				},
			},
			wantErr: true,
			verify: func(t *testing.T, _ *schedulingv1alpha3.Workload, root *WorkloadItem) {
				if emptyNameCallbackRan {
					t.Error("callback ran despite empty name; the name check should short-circuit before resolution and callbacks")
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
				Callbacks: []SchedulingConfigFunc{
					func(cfg *SchedulingConfig) {
						if g := cfg.Policy.Gang; g != nil && g.MinCount != nil {
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
				Name:      "gang-job",
				Input:     WorkloadInput{Policy: PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}}}},
				Callbacks: []SchedulingConfigFunc{defaultGangMinCount(16)},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, root *WorkloadItem) {
				if root.Input.Policy.PodGroupData.Gang.MinCount != nil {
					t.Errorf("callback mutated the caller's Input; MinCount should remain nil, got %d", *root.Input.Policy.PodGroupData.Gang.MinCount)
				}
				gang := wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang
				if gang == nil {
					t.Fatal("expected compiled Gang policy to carry the defaulted MinCount")
				}
				if gang.MinCount != 16 {
					t.Errorf("expected compiled MinCount=16, got %d", gang.MinCount)
				}
			},
		},
		{
			name: "gang missing minCount fails",
			root: &WorkloadItem{
				Name:  "gang-job",
				Input: WorkloadInput{Policy: PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}}}},
			},
			wantErr: true,
		},
		{
			name:    "nil root fails",
			root:    nil,
			wantErr: true,
		},
		{
			name: "missing owner returns error",
			root: &WorkloadItem{
				Name:          "job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
			},
			owner:   nil,
			wantErr: true,
		},
		{
			name: "owner apiVersion that fails to parse returns error",
			root: &WorkloadItem{
				Name:          "job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
			},
			owner: &metav1.OwnerReference{
				APIVersion: "batch/v1/extra",
				Kind:       "Job",
				Name:       "my-job",
				UID:        "abc-123",
			},
			wantErr: true,
		},
	}

	dummyOwner := &metav1.OwnerReference{
		APIVersion: "batch/v1",
		Kind:       "Job",
		Name:       "test-job",
		UID:        "12345",
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			owner := tt.owner
			if owner == nil && tt.name != "missing owner returns error" {
				owner = dummyOwner
			}
			wl, err := NewBuilder(tt.root, BuildOptions{Name: "wl", Namespace: "ns", Owner: owner}).BuildWorkload()
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.verify != nil {
				tt.verify(t, wl, tt.root)
			}
		})
	}
}

func TestResolveSchedulingConfigMergesEveryField(t *testing.T) {
	defaultConfig := &SchedulingConfig{
		Policy:            &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}},
		Constraints:       &SchedulingConstraints{Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}}},
		DisruptionMode:    &DisruptionMode{Single: &SingleDisruptionMode{}},
		ResourceClaims:    []ResourceClaim{{Name: "default-claim", ResourceClaimName: new("default")}},
		PriorityClassName: "default-priority",
	}
	userInput := WorkloadInput{
		Policy:         PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](4)}}},
		Constraints:    ConstraintsInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/region"}}}},
		DisruptionMode: DisruptionModeInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}}},
		ResourceClaims: ResourceClaimsInput{PodGroupData: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{{Name: "user-claim", ResourceClaimName: new("user")}}},
	}

	// WorkloadInput carries no priorityClassName, so the resolved config keeps
	// the default's value while every other field is overridden by the input.
	expectedConfig := &SchedulingConfig{
		Policy:            &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: ptr.To[int32](4)}},
		Constraints:       &SchedulingConstraints{Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/region"}}},
		DisruptionMode:    &DisruptionMode{All: &AllDisruptionMode{}},
		ResourceClaims:    []ResourceClaim{{Name: "user-claim", ResourceClaimName: new("user")}},
		PriorityClassName: "default-priority",
	}

	// Both fixtures must set every field to a non-zero value; otherwise the
	// override assertion below cannot distinguish a copied field from an ignored
	// one. A field added to SchedulingConfig without updating these fixtures trips
	// here first.
	cfgType := reflect.TypeFor[SchedulingConfig]()
	defVal := reflect.ValueOf(*defaultConfig)
	expectedVal := reflect.ValueOf(*expectedConfig)
	for i := 0; i < cfgType.NumField(); i++ {
		name := cfgType.Field(i).Name
		if defVal.Field(i).IsZero() {
			t.Fatalf("defaultConfig fixture leaves SchedulingConfig.%s unset; populate it so this test stays exhaustive", name)
		}
		if expectedVal.Field(i).IsZero() {
			t.Fatalf("expectedConfig fixture leaves SchedulingConfig.%s unset; populate it so this test stays exhaustive", name)
		}
	}

	item := &WorkloadItem{Name: "job", DefaultConfig: defaultConfig, Input: userInput}
	resolved := resolveSchedulingConfig(item)

	if !reflect.DeepEqual(resolved, expectedConfig) {
		t.Errorf("resolved config does not match expected; a field is likely missing from resolveSchedulingConfig's merge\n resolved: %+v\n expected: %+v", resolved, expectedConfig)
	}
}

// Every leaf is populated, including the mutually-exclusive union arms.
func TestSchedulingConfigDeepCopyCopiesEveryField(t *testing.T) {
	original := &SchedulingConfig{
		Policy: &SchedulingPolicy{
			Basic: &BasicSchedulingPolicy{},
			Gang:  &GangSchedulingPolicy{MinCount: ptr.To[int32](4)},
		},
		Constraints:       &SchedulingConstraints{Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}}},
		DisruptionMode:    &DisruptionMode{Single: &SingleDisruptionMode{}, All: &AllDisruptionMode{}},
		ResourceClaims:    []ResourceClaim{{Name: "gpu", ResourceClaimName: new("claim"), ResourceClaimTemplateName: new("tmpl")}},
		PriorityClassName: "high-priority",
	}

	// A field added anywhere under SchedulingConfig (top-level or nested)
	// without updating this fixture leaves a zero leaf and trips here, forcing
	// the fixture to grow so DeepCopy stays fully exercised.
	pkg := reflect.TypeFor[SchedulingConfig]().PkgPath()
	if leaf := firstZeroLeaf(reflect.ValueOf(*original), "SchedulingConfig", pkg); leaf != "" {
		t.Fatalf("fixture leaves %s unset; populate it so DeepCopy is fully exercised", leaf)
	}

	clone := original.DeepCopy()

	// A field DeepCopy forgets stays at its zero value in the clone.
	if !reflect.DeepEqual(original, clone) {
		t.Errorf("DeepCopy is not exhaustive; clone differs from original (a field is likely not copied)\n original: %+v\n clone:    %+v", original, clone)
	}

	// Mutating the clone must not affect the original; otherwise a field is
	// aliased rather than deep-copied.
	clone.Policy.Gang.MinCount = ptr.To[int32](99)
	clone.Constraints.Topology[0].Key = "mutated"
	clone.DisruptionMode.All = nil
	clone.ResourceClaims[0].Name = "mutated"
	*clone.ResourceClaims[0].ResourceClaimName = "mutated"

	if *original.Policy.Gang.MinCount != 4 ||
		original.Constraints.Topology[0].Key != "topology.kubernetes.io/zone" ||
		original.DisruptionMode.All == nil ||
		original.ResourceClaims[0].Name != "gpu" ||
		*original.ResourceClaims[0].ResourceClaimName != "claim" {
		t.Errorf("DeepCopy produced an aliased clone; mutating the clone changed the original: %+v", original)
	}
}

func TestNewBuilder(t *testing.T) {
	item := createGangWorkloadItem()
	opts := BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()}

	b := NewBuilder(item, opts)
	if b.root != item {
		t.Errorf("root not stored: got %p want %p", b.root, item)
	}
	if b.opts.Name != "wl" || b.opts.Namespace != "ns" || b.opts.Owner == nil {
		t.Errorf("opts not stored: %+v", b.opts)
	}
	if b.workload != nil {
		t.Error("workload should be nil before BuildWorkload")
	}
}

func TestBuilderBuildWorkload(t *testing.T) {
	t.Run("compiles identity and controllerRef", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		wl, err := b.BuildWorkload()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if wl.Name != "wl" || wl.Namespace != "ns" {
			t.Errorf("unexpected identity: %s/%s", wl.Namespace, wl.Name)
		}
		if len(wl.OwnerReferences) != 1 || wl.OwnerReferences[0].Name != "job" {
			t.Errorf("expected single job ownerReference, got %+v", wl.OwnerReferences)
		}
		if wl.Spec.ControllerRef == nil ||
			wl.Spec.ControllerRef.APIGroup != "batch" ||
			wl.Spec.ControllerRef.Kind != "Job" ||
			wl.Spec.ControllerRef.Name != "job" {
			t.Errorf("unexpected controllerRef: %+v", wl.Spec.ControllerRef)
		}
		if len(wl.Spec.PodGroupTemplates) != 1 || wl.Spec.PodGroupTemplates[0].Name != "pgt-0" {
			t.Errorf("expected single template pgt-0, got %+v", wl.Spec.PodGroupTemplates)
		}
	})

	t.Run("caches the compiled workload", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		first, err := b.BuildWorkload()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		second, err := b.BuildWorkload()
		if err != nil {
			t.Fatalf("unexpected error on second call: %v", err)
		}
		if first != second {
			t.Error("expected BuildWorkload to return the cached workload on repeat calls")
		}
	})

	t.Run("errors are returned and nothing is cached", func(t *testing.T) {
		tests := []struct {
			name string
			b    *Builder
		}{
			{
				name: "nil root",
				b:    NewBuilder(nil, BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()}),
			},
			{
				name: "missing owner",
				b:    NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns"}),
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				wl, err := tt.b.BuildWorkload()
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if wl != nil {
					t.Errorf("expected nil workload on error, got %+v", wl)
				}
				if tt.b.workload != nil {
					t.Error("failed build must not populate the cache")
				}
			})
		}
	})
}

func TestBuilderNewPodGroup(t *testing.T) {
	t.Run("requires a workload first", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		_, err := b.NewPodGroup("pg", "pgt-0")
		if err == nil {
			t.Fatal("expected error when neither BuildWorkload nor NewBuilderFromExistingWorkload has run")
		}
		if !strings.Contains(err.Error(), "call BuildWorkload or use NewBuilderFromExistingWorkload") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("materializes from the compiled template owned by the configured owner", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		if _, err := b.BuildWorkload(); err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}
		pg, err := b.NewPodGroup("pg", "pgt-0")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pg.Name != "pg" || pg.Namespace != "ns" {
			t.Errorf("unexpected identity: %s/%s", pg.Namespace, pg.Name)
		}
		if len(pg.OwnerReferences) != 1 || pg.OwnerReferences[0].Name != "job" {
			t.Errorf("expected single job ownerReference from opts.Owner, got %+v", pg.OwnerReferences)
		}
		if pg.Spec.WorkloadRef == nil ||
			pg.Spec.WorkloadRef.WorkloadName != "wl" ||
			pg.Spec.WorkloadRef.TemplateName != "pgt-0" {
			t.Errorf("unexpected workloadRef: %+v", pg.Spec.WorkloadRef)
		}
	})

	t.Run("omits ownerReferences when no owner is configured", func(t *testing.T) {
		// build() requires an owner, so compile with one, then clear it to
		// exercise the ownerless NewPodGroup path against the cached workload.
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		if _, err := b.BuildWorkload(); err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}
		b.opts.Owner = nil
		pg, err := b.NewPodGroup("pg", "pgt-0")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(pg.OwnerReferences) != 0 {
			t.Errorf("expected no ownerReferences, got %+v", pg.OwnerReferences)
		}
	})

	t.Run("errors on unknown template", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		if _, err := b.BuildWorkload(); err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}
		if _, err := b.NewPodGroup("pg", "missing"); err == nil {
			t.Error("expected error for unknown template name")
		}
	})
}

func TestNewBuilderFromExistingWorkload(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	t.Run("materializes from the supplied workload without BuildWorkload", func(t *testing.T) {
		b := NewBuilderFromExistingWorkload(existingWorkload(), BuildOptions{Owner: jobOwner()})
		pg, err := b.NewPodGroup("pg", "pgt-0")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pg.Name != "pg" || pg.Namespace != "ns" {
			t.Errorf("unexpected identity: %s/%s", pg.Namespace, pg.Name)
		}
		if len(pg.OwnerReferences) != 1 || pg.OwnerReferences[0].Name != "job" {
			t.Errorf("expected single job ownerReference, got %+v", pg.OwnerReferences)
		}
		// Fields the builder never sets must be copied from the supplied template.
		if pg.Spec.PriorityClassName != "high-priority" {
			t.Errorf("expected priorityClassName copied, got %q", pg.Spec.PriorityClassName)
		}
		if pg.Spec.Priority == nil || *pg.Spec.Priority != 1000 {
			t.Errorf("expected priority copied, got %v", pg.Spec.Priority)
		}
		if pg.Spec.WorkloadRef == nil || pg.Spec.WorkloadRef.WorkloadName != "parent-wl" {
			t.Errorf("unexpected workloadRef: %+v", pg.Spec.WorkloadRef)
		}
	})

	t.Run("BuildWorkload is refused in existing mode", func(t *testing.T) {
		b := NewBuilderFromExistingWorkload(existingWorkload(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		_, err := b.BuildWorkload()
		if err == nil {
			t.Fatal("expected BuildWorkload to be refused for a builder created from an existing Workload")
		}
		if !strings.Contains(err.Error(), "not available on a Builder created from an existing Workload") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("Validate is refused in existing mode", func(t *testing.T) {
		b := NewBuilderFromExistingWorkload(existingWorkload(), BuildOptions{Owner: jobOwner()})
		errs := b.Validate(ctx, field.NewPath("spec", "scheduling"), ValidationInput{})
		if len(errs) != 1 {
			t.Fatalf("expected exactly one validation error, got %v", errs)
		}
		if errs[0].Type != field.ErrorTypeInternal {
			t.Errorf("expected an internal error, got %s: %v", errs[0].Type, errs[0])
		}
	})

	t.Run("errors on unknown template", func(t *testing.T) {
		b := NewBuilderFromExistingWorkload(existingWorkload(), BuildOptions{Owner: jobOwner()})
		if _, err := b.NewPodGroup("pg", "missing"); err == nil {
			t.Error("expected error for unknown template name")
		}
	})
}

func jobOwner() *metav1.OwnerReference {
	return &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "job", UID: "job-uid"}
}

func createGangWorkloadItem() *WorkloadItem {
	return &WorkloadItem{
		Name: "pgt-0",
		Input: WorkloadInput{Policy: PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
			Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](3)},
		}}},
	}
}

func existingWorkload() *schedulingv1alpha3.Workload {
	return &schedulingv1alpha3.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: "parent-wl", Namespace: "ns", UID: "wl-uid"},
		Spec: schedulingv1alpha3.WorkloadSpec{
			PodGroupTemplates: []schedulingv1alpha3.PodGroupTemplate{{
				Name:              "pgt-0",
				SchedulingPolicy:  schedulingv1alpha3.PodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 4}},
				PriorityClassName: "high-priority",
				Priority:          ptr.To[int32](1000),
			}},
		},
	}
}

// firstZeroLeaf walks v and returns the dotted path of the first leaf still
// holding its zero value, or "" when every leaf is populated.
func firstZeroLeaf(v reflect.Value, path, pkg string) string {
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return path
		}
		return firstZeroLeaf(v.Elem(), path, pkg)
	case reflect.Slice:
		if v.Len() == 0 {
			return path
		}
		return firstZeroLeaf(v.Index(0), path+"[0]", pkg)
	case reflect.Struct:
		if v.Type().PkgPath() != pkg {
			if v.IsZero() {
				return path
			}
			return ""
		}
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			if leaf := firstZeroLeaf(v.Field(i), path+"."+t.Field(i).Name, pkg); leaf != "" {
				return leaf
			}
		}
		return ""
	default:
		if v.IsZero() {
			return path
		}
		return ""
	}
}
