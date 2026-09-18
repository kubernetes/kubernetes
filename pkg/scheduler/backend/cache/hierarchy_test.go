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

package cache

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	fwk "k8s.io/kube-scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// setupHierarchyWrapper builds a wrapper over snapshot states containing:
//   - pg1 -> cpg1 -> cpg2, where cpg2 is the root
//   - pg_root, a PodGroup with no parent
//   - pg_cycle -> cpg_cycle_1 -> cpg_cycle_2 -> cpg_cycle_1
//   - pg_missing_parent -> a CompositePodGroup that does not exist
func setupHierarchyWrapper(compositePodGroupEnabled bool) *hierarchyWrapper[*podGroupStateSnapshot, *compositePodGroupStateSnapshot] {
	podGroups := []*schedulingv1beta1.PodGroup{
		st.MakePodGroup().Name("pg1").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj(),
		st.MakePodGroup().Name("pg_root").Namespace("ns1").Obj(),
		st.MakePodGroup().Name("pg_cycle").Namespace("ns1").ParentCompositePodGroup("cpg_cycle_1").Obj(),
		st.MakePodGroup().Name("pg_missing_parent").Namespace("ns1").ParentCompositePodGroup("non-existent").Obj(),
	}
	compositePodGroups := []*schedulingv1alpha3.CompositePodGroup{
		st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").ParentCompositePodGroup("cpg2").Obj(),
		st.MakeCompositePodGroup().Name("cpg2").Namespace("ns1").Obj(),
		st.MakeCompositePodGroup().Name("cpg_cycle_1").Namespace("ns1").ParentCompositePodGroup("cpg_cycle_2").Obj(),
		st.MakeCompositePodGroup().Name("cpg_cycle_2").Namespace("ns1").ParentCompositePodGroup("cpg_cycle_1").Obj(),
	}

	pgStates := make(map[fwk.EntityKey]*podGroupStateSnapshot, len(podGroups))
	for _, pg := range podGroups {
		pgStates[fwk.PodGroupKey(pg.Namespace, pg.Name)] = &podGroupStateSnapshot{
			podGroupStateData: podGroupStateData{podGroup: pg},
		}
	}
	cpgStates := make(map[fwk.EntityKey]*compositePodGroupStateSnapshot, len(compositePodGroups))
	for _, cpg := range compositePodGroups {
		cpgStates[fwk.CompositePodGroupKey(cpg.Namespace, cpg.Name)] = &compositePodGroupStateSnapshot{
			compositePodGroupStateData: compositePodGroupStateData{compositePodGroup: cpg},
		}
	}

	return newHierarchyWrapper(pgStates, cpgStates, compositePodGroupEnabled)
}

func TestHierarchyWrapper_FindRootKeyForGroup(t *testing.T) {
	keyPtr := func(k fwk.EntityKey) *fwk.EntityKey { return &k }

	tests := []struct {
		name                     string
		compositePodGroupEnabled bool
		key                      fwk.EntityKey
		want                     *fwk.EntityKey
		wantErr                  bool
	}{
		{
			name:                     "from pg to root (CPG=true)",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "pg1"),
			want:                     keyPtr(fwk.CompositePodGroupKey("ns1", "cpg2")),
		},
		{
			name:                     "from cpg to root (CPG=true)",
			compositePodGroupEnabled: true,
			key:                      fwk.CompositePodGroupKey("ns1", "cpg1"),
			want:                     keyPtr(fwk.CompositePodGroupKey("ns1", "cpg2")),
		},
		{
			name:                     "from root cpg (CPG=true)",
			compositePodGroupEnabled: true,
			key:                      fwk.CompositePodGroupKey("ns1", "cpg2"),
			want:                     keyPtr(fwk.CompositePodGroupKey("ns1", "cpg2")),
		},
		{
			name:                     "from pg to root pg (CPG=true)",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "pg_root"),
			want:                     keyPtr(fwk.PodGroupKey("ns1", "pg_root")),
		},
		{
			name:                     "from pg (with parent set), compositePodGroup disabled",
			compositePodGroupEnabled: false,
			key:                      fwk.PodGroupKey("ns1", "pg1"),
			want:                     keyPtr(fwk.PodGroupKey("ns1", "pg1")),
		},
		{
			name:                     "from cpg (with parent set), compositePodGroup disabled",
			compositePodGroupEnabled: false,
			key:                      fwk.CompositePodGroupKey("ns1", "cpg1"),
			want:                     keyPtr(fwk.CompositePodGroupKey("ns1", "cpg1")),
		},
		{
			name:                     "missing intermediate",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "pg_missing_parent"),
			want:                     nil,
		},
		{
			name:                     "missing group",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "non-existent"),
			want:                     nil,
		},
		{
			name:                     "cycle detected",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "pg_cycle"),
			wantErr:                  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hw := setupHierarchyWrapper(tt.compositePodGroupEnabled)
			got, err := hw.FindRootKeyForGroup(tt.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("FindRootKeyForGroup() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmp.Equal(got, tt.want) {
				t.Errorf("FindRootKeyForGroup() diff (-got, +want): %s", cmp.Diff(got, tt.want))
			}
		})
	}
}

func TestHierarchyWrapper_FindRootGroup(t *testing.T) {
	tests := []struct {
		name                     string
		compositePodGroupEnabled bool
		key                      fwk.EntityKey
		wantKey                  fwk.EntityKey
		wantIsCPG                bool
		wantExists               bool
		wantErr                  bool
	}{
		{
			name:                     "from pg to cpg root (CPG=true)",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "pg1"),
			wantKey:                  fwk.CompositePodGroupKey("ns1", "cpg2"),
			wantIsCPG:                true,
			wantExists:               true,
		},
		{
			name:                     "from pg to pg root (CPG=true)",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "pg_root"),
			wantKey:                  fwk.PodGroupKey("ns1", "pg_root"),
			wantExists:               true,
		},
		{
			name:                     "from pg (with parent set), compositePodGroup disabled",
			compositePodGroupEnabled: false,
			key:                      fwk.PodGroupKey("ns1", "pg1"),
			wantKey:                  fwk.PodGroupKey("ns1", "pg1"),
			wantExists:               true,
		},
		{
			name:                     "from cpg (with parent set), compositePodGroup disabled",
			compositePodGroupEnabled: false,
			key:                      fwk.CompositePodGroupKey("ns1", "cpg1"),
			wantKey:                  fwk.CompositePodGroupKey("ns1", "cpg1"),
			wantIsCPG:                true,
			wantExists:               true,
		},
		{
			name:                     "missing intermediate",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "pg_missing_parent"),
			wantExists:               false,
		},
		{
			name:                     "cycle detected",
			compositePodGroupEnabled: true,
			key:                      fwk.PodGroupKey("ns1", "pg_cycle"),
			wantErr:                  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hw := setupHierarchyWrapper(tt.compositePodGroupEnabled)
			gotGroup, err := hw.FindRootGroup(tt.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("FindRootGroup() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			gotExists := gotGroup != nil
			if gotExists != tt.wantExists {
				t.Errorf("FindRootGroup() gotExists = %v, wantExists %v", gotExists, tt.wantExists)
				return
			}
			if !gotExists || tt.wantErr {
				return
			}
			if gotGroup.GetKey() != tt.wantKey {
				t.Errorf("FindRootGroup() gotKey = %v, wantKey %v", gotGroup.GetKey(), tt.wantKey)
			}
			if tt.wantIsCPG {
				if gotGroup.CompositePodGroup == nil || gotGroup.CompositePodGroupState == nil {
					t.Errorf("FindRootGroup() expected CPG and CPGState to be non-nil")
				}
				if gotGroup.PodGroup != nil || gotGroup.PodGroupState != nil {
					t.Errorf("FindRootGroup() expected PG and PGState to be nil when root is CPG")
				}
			} else {
				if gotGroup.PodGroup == nil || gotGroup.PodGroupState == nil {
					t.Errorf("FindRootGroup() expected PG and PGState to be non-nil")
				}
				if gotGroup.CompositePodGroup != nil || gotGroup.CompositePodGroupState != nil {
					t.Errorf("FindRootGroup() expected CPG and CPGState to be nil when root is PG")
				}
			}
		})
	}
}
