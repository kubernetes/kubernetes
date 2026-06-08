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
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func TestBuild(t *testing.T) {
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
				UserConfig:    &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
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
			name: "more than one topology constraint fails",
			root: &WorkloadItem{
				Name:          "topo-job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					Constraints: &SchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{
							{Key: "topology.kubernetes.io/zone"},
							{Key: "topology.kubernetes.io/region"},
						},
					},
				},
			},
			wantErr: true,
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
					ResourceClaims: []ResourceClaim{{Name: "gpu", ResourceClaimName: new("my-claim")}},
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
					ResourceClaims: []ResourceClaim{{Name: "gpu", ResourceClaimName: new("a"), ResourceClaimTemplateName: new("b")}},
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
					ResourceClaims: []ResourceClaim{{Name: "", ResourceClaimName: new("a")}},
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
						{Name: "dup", ResourceClaimName: new("a")},
						{Name: "dup", ResourceClaimName: new("b")},
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
			name: "empty leaf name fails before resolution and callbacks",
			root: &WorkloadItem{
				Name:       "",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
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
				Name:       "gang-job",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
				Callbacks:  []SchedulingConfigFunc{defaultGangMinCount(16)},
			},
			verify: func(t *testing.T, wl *schedulingv1alpha3.Workload, root *WorkloadItem) {
				if root.UserConfig.Policy.Gang.MinCount != nil {
					t.Errorf("callback mutated the caller's UserConfig; MinCount should remain nil, got %d", *root.UserConfig.Policy.Gang.MinCount)
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
				Name:       "gang-job",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
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
			wl, err := Build(tt.root, BuildOptions{Name: "wl", Namespace: "ns", Owner: owner})
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

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		root    *WorkloadItem
		wantErr bool
	}{
		{
			name:    "nil root is required",
			root:    nil,
			wantErr: true,
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
				Callbacks:  []SchedulingConfigFunc{defaultGangMinCount(4)},
			},
		},
		{
			name: "gang minCount unset without callback is invalid",
			root: &WorkloadItem{
				Name:       "job",
				UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{}}},
			},
			wantErr: true,
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
			wantErr: true,
		},
		{
			name: "duplicate resource claim names is invalid",
			root: &WorkloadItem{
				Name:          "job",
				DefaultConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}}},
				UserConfig: &SchedulingConfig{
					ResourceClaims: []ResourceClaim{
						{Name: "dup", ResourceClaimName: new("a")},
						{Name: "dup", ResourceClaimName: new("b")},
					},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(tt.root)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("expected no error, got %v", err)
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
			validateErr := Validate(root) != nil
			_, buildErrs := Build(root, BuildOptions{
				Name:      "wl",
				Namespace: "ns",
				Owner: &metav1.OwnerReference{
					APIVersion: "batch/v1",
					Kind:       "Job",
					Name:       "test-job",
					UID:        "12345",
				},
			})
			buildErr := buildErrs != nil
			if validateErr != buildErr {
				t.Errorf("Validate/Build disagree: validateErr=%v buildErr=%v", validateErr, buildErr)
			}
		})
	}
}

func TestResolveSchedulingConfigMergesEveryField(t *testing.T) {
	defaultConfig := &SchedulingConfig{
		Policy:         &SchedulingPolicy{Basic: &BasicSchedulingPolicy{}},
		Constraints:    &SchedulingConstraints{Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}}},
		DisruptionMode: &DisruptionMode{Single: &SingleDisruptionMode{}},
		ResourceClaims: []ResourceClaim{{Name: "default-claim", ResourceClaimName: new("default")}},
	}
	userConfig := &SchedulingConfig{
		Policy:         &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: ptr.To[int32](4)}},
		Constraints:    &SchedulingConstraints{Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/region"}}},
		DisruptionMode: &DisruptionMode{All: &AllDisruptionMode{}},
		ResourceClaims: []ResourceClaim{{Name: "user-claim", ResourceClaimName: new("user")}},
	}

	// Both fixtures must set every field to a non-zero value; otherwise the
	// override assertion below cannot distinguish a copied field from an ignored
	// one. A field added to SchedulingConfig without updating these fixtures trips
	// here first.
	cfgType := reflect.TypeFor[SchedulingConfig]()
	defVal := reflect.ValueOf(*defaultConfig)
	userVal := reflect.ValueOf(*userConfig)
	for i := 0; i < cfgType.NumField(); i++ {
		name := cfgType.Field(i).Name
		if defVal.Field(i).IsZero() {
			t.Fatalf("defaultConfig fixture leaves SchedulingConfig.%s unset; populate it so this test stays exhaustive", name)
		}
		if userVal.Field(i).IsZero() {
			t.Fatalf("userConfig fixture leaves SchedulingConfig.%s unset; populate it so this test stays exhaustive", name)
		}
	}

	item := &WorkloadItem{Name: "job", DefaultConfig: defaultConfig, UserConfig: userConfig}
	resolved := resolveSchedulingConfig(item)

	if !reflect.DeepEqual(resolved, userConfig) {
		t.Errorf("resolved config does not match UserConfig; a field is likely missing from resolveSchedulingConfig's merge\n resolved:   %+v\n userConfig: %+v", resolved, userConfig)
	}
}

// Every leaf is populated, including the mutually-exclusive union arms.
func TestSchedulingConfigDeepCopyCopiesEveryField(t *testing.T) {
	original := &SchedulingConfig{
		Policy: &SchedulingPolicy{
			Basic: &BasicSchedulingPolicy{},
			Gang:  &GangSchedulingPolicy{MinCount: ptr.To[int32](4)},
		},
		Constraints:    &SchedulingConstraints{Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}}},
		DisruptionMode: &DisruptionMode{Single: &SingleDisruptionMode{}, All: &AllDisruptionMode{}},
		ResourceClaims: []ResourceClaim{{Name: "gpu", ResourceClaimName: new("claim"), ResourceClaimTemplateName: new("tmpl")}},
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
