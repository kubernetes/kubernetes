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

package helper

import (
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestMatchingSchedulingGroup(t *testing.T) {
	testCases := []struct {
		name     string
		pod1     *v1.Pod
		pod2     *v1.Pod
		expected bool
	}{
		{
			name:     "same pod with schedulingGroup",
			pod1:     st.MakePod().Name("pod1").Namespace("test").PodGroupName("name").Obj(),
			pod2:     st.MakePod().Name("pod1").Namespace("test").PodGroupName("name").Obj(),
			expected: true,
		},
		{
			name:     "different pods, same schedulingGroup",
			pod1:     st.MakePod().Name("pod1").Namespace("test").PodGroupName("name").Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").PodGroupName("name").Obj(),
			expected: true,
		},
		{
			name:     "same pod but no schedulingGroup",
			pod1:     st.MakePod().Name("pod1").Namespace("test").Obj(),
			pod2:     st.MakePod().Name("pod1").Namespace("test").Obj(),
			expected: false,
		},
		{
			name:     "different pods, only one with schedulingGroup",
			pod1:     st.MakePod().Name("pod1").Namespace("test").PodGroupName("name").Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").Obj(),
			expected: false,
		},
		{
			name:     "same schedulingGroup but different namespaces",
			pod1:     st.MakePod().Name("pod1").Namespace("test1").PodGroupName("name").Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test2").PodGroupName("name").Obj(),
			expected: false,
		},
		{
			name:     "same namespace but different pod group",
			pod1:     st.MakePod().Name("pod1").Namespace("test").PodGroupName("name1").Obj(),
			pod2:     st.MakePod().Name("pod2").Namespace("test").PodGroupName("name2").Obj(),
			expected: false,
		},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchingSchedulingGroup(tt.pod1, tt.pod2); got != tt.expected {
				t.Errorf("MatchingSchedulingGroup() = %v, want %v", got, tt.expected)
			}
		})
	}
}

type fakeSharedLister struct {
	fwk.SharedLister
	faultyEntityKeys sets.Set[fwk.EntityKey]
}

func (f *fakeSharedLister) CompositePodGroupStates() fwk.CompositePodGroupStateLister {
	return &fakeCompositePodGroupStateLister{
		CompositePodGroupStateLister: f.SharedLister.CompositePodGroupStates(),
		faultyEntityKeys:             f.faultyEntityKeys,
	}
}

func (f *fakeSharedLister) PodGroupStates() fwk.PodGroupStateLister {
	return &fakePodGroupStateLister{
		PodGroupStateLister: f.SharedLister.PodGroupStates(),
		faultyEntityKeys:    f.faultyEntityKeys,
	}
}

type fakeCompositePodGroupStateLister struct {
	fwk.CompositePodGroupStateLister
	faultyEntityKeys sets.Set[fwk.EntityKey]
}

func (l *fakeCompositePodGroupStateLister) Get(namespace, name string) (fwk.CompositePodGroupState, error) {
	if l.faultyEntityKeys.Has(fwk.CompositePodGroupKey(namespace, name)) {
		return nil, fmt.Errorf("composite pod group %s/%s not found", namespace, name)
	}
	return l.CompositePodGroupStateLister.Get(namespace, name)
}

type fakePodGroupStateLister struct {
	fwk.PodGroupStateLister
	faultyEntityKeys sets.Set[fwk.EntityKey]
}

func (l *fakePodGroupStateLister) Get(namespace, name string) (fwk.PodGroupState, error) {
	if l.faultyEntityKeys.Has(fwk.PodGroupKey(namespace, name)) {
		return nil, fmt.Errorf("pod group %s/%s not found", namespace, name)
	}
	return l.PodGroupStateLister.Get(namespace, name)
}

func TestGetPodGroupStates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	testCases := []struct {
		name               string
		podGroups          []*schedulingv1beta1.PodGroup
		compositePodGroups []*schedulingv1alpha3.CompositePodGroup
		rootEntityKey      fwk.EntityKey
		faultyEntityKeys   []fwk.EntityKey
		wantErrCount       int
	}{
		{
			name: "root is PodGroup",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("test").Obj(),
			},
			rootEntityKey: fwk.PodGroupKey("test", "pg1"),
		},
		{
			name: "root is CompositePodGroup",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
				st.MakePodGroup().Name("pg2").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Namespace("test").Obj(),
			},
			rootEntityKey: fwk.CompositePodGroupKey("test", "cpg-root"),
		},
		{
			name: "root is CompositePodGroup with nested CompositePodGroup",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
				st.MakePodGroup().Name("pg2").Namespace("test").ParentCompositePodGroup("cpg1").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Namespace("test").Obj(),
				st.MakeCompositePodGroup().Name("cpg1").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
			},
			rootEntityKey: fwk.CompositePodGroupKey("test", "cpg-root"),
		},
		{
			name:      "root is CompositePodGroup and has no leaves",
			podGroups: []*schedulingv1beta1.PodGroup{},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Namespace("test").Obj(),
			},
			rootEntityKey: fwk.CompositePodGroupKey("test", "cpg-root"),
		},
		{
			name: "root is CompositePodGroup and one entity key is missing",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg-leaf").Namespace("test").ParentCompositePodGroup("cpg-mid2").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Namespace("test").Obj(),
				st.MakeCompositePodGroup().Name("cpg-mid1").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
				st.MakeCompositePodGroup().Name("cpg-mid2").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
			},
			rootEntityKey: fwk.CompositePodGroupKey("test", "cpg-root"),
			faultyEntityKeys: []fwk.EntityKey{
				fwk.CompositePodGroupKey("test", "cpg-mid1"),
			},
			wantErrCount: 1,
		},
		{
			name:      "root is CompositePodGroup and multiple entity keys are missing",
			podGroups: []*schedulingv1beta1.PodGroup{},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Namespace("test").Obj(),
				st.MakeCompositePodGroup().Name("cpg-mid1").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
				st.MakeCompositePodGroup().Name("cpg-mid2").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
			},
			rootEntityKey: fwk.CompositePodGroupKey("test", "cpg-root"),
			faultyEntityKeys: []fwk.EntityKey{
				fwk.CompositePodGroupKey("test", "cpg-mid1"),
				fwk.CompositePodGroupKey("test", "cpg-mid2"),
			},
			wantErrCount: 2,
		},
		{
			name:          "there's no root matching the entity key",
			rootEntityKey: fwk.PodGroupKey("test", "non-existent"),
			wantErrCount:  1,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			wantPodGroups := sets.New[string]()
			var pods []*v1.Pod
			for _, pg := range tt.podGroups {
				pods = append(pods, st.MakePod().Name(pg.Name).UID(pg.Name).Namespace(pg.Namespace).PodGroupName(pg.Name).Obj())
				wantPodGroups.Insert(pg.Name)
			}

			snapshot := internalcache.NewTestSnapshotWithPodGroups(pods, nil, tt.podGroups, tt.compositePodGroups)
			sharedLister := &fakeSharedLister{
				SharedLister:     snapshot,
				faultyEntityKeys: sets.New(tt.faultyEntityKeys...),
			}

			gotPodGroups := sets.New[string]()
			var errors []error
			for pgState, err := range GetPodGroupStates(sharedLister, tt.rootEntityKey) {
				if err != nil {
					errors = append(errors, err)
				} else {
					// Each PG has a single pod with the name matching the PG name.
					gotPodGroups.Insert(string(pgState.AllPods().UnsortedList()[0]))
				}
			}

			if diff := cmp.Diff(wantPodGroups, gotPodGroups); diff != "" {
				t.Errorf("GetPodGroupStates() returned unexpected pods (-want +got):\n%s", diff)
			}
			if tt.wantErrCount != len(errors) {
				t.Errorf("GetPodGroupStates() returned unexpected errors: want %d, got %d", tt.wantErrCount, len(errors))
			}
		})
	}
}

func TestGetPodGroupStates_EarlyBreak(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	podGroups := []*schedulingv1beta1.PodGroup{
		st.MakePodGroup().Name("pg1").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
		st.MakePodGroup().Name("pg2").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
		st.MakePodGroup().Name("pg3").Namespace("test").ParentCompositePodGroup("cpg-root").Obj(),
	}
	compositePodGroups := []*schedulingv1alpha3.CompositePodGroup{
		st.MakeCompositePodGroup().Name("cpg-root").Namespace("test").Obj(),
	}

	snapshot := internalcache.NewTestSnapshotWithPodGroups(nil, nil, podGroups, compositePodGroups)
	sharedLister := &fakeSharedLister{SharedLister: snapshot}
	rootEntityKey := fwk.CompositePodGroupKey("test", "cpg-root")

	var gotCount int
	for range GetPodGroupStates(sharedLister, rootEntityKey) {
		gotCount++
		break
	}

	if gotCount != 1 {
		t.Errorf("GetPodGroupStates() yielded %d times after early break, want 1", gotCount)
	}
}

func TestTraverseHierarchyUp(t *testing.T) {
	const ns = "test"
	testCases := []struct {
		name                      string
		disableCompositePodGroups bool
		podGroups                 []*schedulingv1beta1.PodGroup
		compositePodGroups        []*schedulingv1alpha3.CompositePodGroup
		startKey                  fwk.EntityKey
		// stopAt makes the consumer break out of the loop after this group, if non-empty.
		stopAt   string
		wantKeys []fwk.EntityKey
		// wantErr is a substring of the error the traversal is expected to end with; empty means no error.
		wantErr string
	}{
		{
			name: "PodGroup without parent",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace(ns).Obj(),
			},
			startKey: fwk.PodGroupKey(ns, "pg1"),
			wantKeys: []fwk.EntityKey{fwk.PodGroupKey(ns, "pg1")},
		},
		{
			name: "PodGroup under two levels of CompositePodGroups",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace(ns).ParentCompositePodGroup("cpg1").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace(ns).ParentCompositePodGroup("cpg2").Obj(),
				st.MakeCompositePodGroup().Name("cpg2").Namespace(ns).Obj(),
			},
			startKey: fwk.PodGroupKey(ns, "pg1"),
			wantKeys: []fwk.EntityKey{
				fwk.PodGroupKey(ns, "pg1"),
				fwk.CompositePodGroupKey(ns, "cpg1"),
				fwk.CompositePodGroupKey(ns, "cpg2"),
			},
		},
		{
			name: "start from a CompositePodGroup",
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace(ns).ParentCompositePodGroup("cpg2").Obj(),
				st.MakeCompositePodGroup().Name("cpg2").Namespace(ns).Obj(),
			},
			startKey: fwk.CompositePodGroupKey(ns, "cpg1"),
			wantKeys: []fwk.EntityKey{
				fwk.CompositePodGroupKey(ns, "cpg1"),
				fwk.CompositePodGroupKey(ns, "cpg2"),
			},
		},
		{
			name: "consumer stops early",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace(ns).ParentCompositePodGroup("cpg1").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace(ns).ParentCompositePodGroup("cpg2").Obj(),
				st.MakeCompositePodGroup().Name("cpg2").Namespace(ns).Obj(),
			},
			startKey: fwk.PodGroupKey(ns, "pg1"),
			stopAt:   "cpg1",
			wantKeys: []fwk.EntityKey{
				fwk.PodGroupKey(ns, "pg1"),
				fwk.CompositePodGroupKey(ns, "cpg1"),
			},
		},
		{
			name:     "missing PodGroup",
			startKey: fwk.PodGroupKey(ns, "missing"),
			wantErr:  "not found",
		},
		{
			name: "missing parent CompositePodGroup ends the traversal after the PodGroup",
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace(ns).ParentCompositePodGroup("missing").Obj(),
			},
			startKey: fwk.PodGroupKey(ns, "pg1"),
			wantKeys: []fwk.EntityKey{fwk.PodGroupKey(ns, "pg1")},
			wantErr:  "not found",
		},
		{
			name:                      "CompositePodGroup feature gate disabled ends the traversal after the PodGroup",
			disableCompositePodGroups: true,
			podGroups: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace(ns).ParentCompositePodGroup("cpg1").Obj(),
			},
			compositePodGroups: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg1").Namespace(ns).Obj(),
			},
			startKey: fwk.PodGroupKey(ns, "pg1"),
			wantKeys: []fwk.EntityKey{fwk.PodGroupKey(ns, "pg1")},
			wantErr:  "feature gate is disabled",
		},
		{
			name:     "unsupported key type",
			startKey: fwk.PodKey(ns, "p1"),
			wantErr:  "unsupported key type",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			// The snapshot's listers enforce the gates, which is what the gate-disabled case relies on.
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: true,
				features.CompositePodGroup:               !tt.disableCompositePodGroups,
			})
			snapshot := internalcache.NewTestSnapshotWithPodGroups(nil, nil, tt.podGroups, tt.compositePodGroups)

			var gotKeys []fwk.EntityKey
			var gotErr error
			for gpg, err := range TraverseHierarchyUp(snapshot, tt.startKey) {
				if err != nil {
					gotErr = err
					continue
				}
				gotKeys = append(gotKeys, gpg.GetKey())
				if tt.stopAt != "" && gpg.GetName() == tt.stopAt {
					break
				}
			}

			if diff := cmp.Diff(tt.wantKeys, gotKeys); diff != "" {
				t.Errorf("TraverseHierarchyUp() yielded unexpected keys (-want +got):\n%s", diff)
			}
			switch {
			case tt.wantErr == "" && gotErr != nil:
				t.Errorf("TraverseHierarchyUp() yielded unexpected error: %v", gotErr)
			case tt.wantErr != "" && gotErr == nil:
				t.Errorf("TraverseHierarchyUp() yielded no error, want one containing %q", tt.wantErr)
			case tt.wantErr != "" && !strings.Contains(gotErr.Error(), tt.wantErr):
				t.Errorf("TraverseHierarchyUp() yielded error %q, want one containing %q", gotErr, tt.wantErr)
			}
		})
	}
}
