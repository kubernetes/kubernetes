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

package queue

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestWorkloadForest_AddPodGroup(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()
	pg2 := st.MakePodGroup().Name("pg2").Namespace("ns1").UID("uid2").Obj()
	pg3WithParent := st.MakePodGroup().Name("pg3").Namespace("ns1").UID("uid3").ParentCompositePodGroup("cpg1").Obj()
	pg4WithParent := st.MakePodGroup().Name("pg4").Namespace("ns1").UID("uid4").ParentCompositePodGroup("cpg1").Obj()
	cpgChild := st.MakeCompositePodGroup().Name("cpgChild").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()

	tests := []struct {
		name                       string
		isCompositePodGroupEnabled bool
		initialCPGs                []*schedulingv1alpha3.CompositePodGroup
		podGroupsToAdd             []*schedulingv1beta1.PodGroup
		want                       map[fwk.EntityKey]*framework.AbstractPodGroup
		wantChildren               map[fwk.EntityKey]sets.Set[fwk.EntityKey]
	}{
		{
			name:                       "add single pod group",
			isCompositePodGroupEnabled: true,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg1},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg1"): framework.NewAbstractPodGroup(pg1),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "add multiple pod groups",
			isCompositePodGroupEnabled: true,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg1, pg2},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg1"): framework.NewAbstractPodGroup(pg1),
				fwk.PodGroupKey("ns1", "pg2"): framework.NewAbstractPodGroup(pg2),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "add pod group with parent, parent not in children",
			isCompositePodGroupEnabled: true,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg3WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg3"): framework.NewAbstractPodGroup(pg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.PodGroupKey("ns1", "pg3")),
			},
		},
		{
			name:                       "add pod group with parent, parent already in children",
			isCompositePodGroupEnabled: true,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg3WithParent, pg4WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg3"): framework.NewAbstractPodGroup(pg3WithParent),
				fwk.PodGroupKey("ns1", "pg4"): framework.NewAbstractPodGroup(pg4WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.PodGroupKey("ns1", "pg3"), fwk.PodGroupKey("ns1", "pg4")),
			},
		},
		{
			name:                       "add pod group with parent, parent already has composite pod group child",
			isCompositePodGroupEnabled: true,
			initialCPGs:                []*schedulingv1alpha3.CompositePodGroup{cpgChild},
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg3WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpgChild"): framework.NewAbstractCompositePodGroup(cpgChild),
				fwk.PodGroupKey("ns1", "pg3"):               framework.NewAbstractPodGroup(pg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.CompositePodGroupKey("ns1", "cpgChild"), fwk.PodGroupKey("ns1", "pg3")),
			},
		},
		{
			name:                       "add pod group with parent, feature disabled",
			isCompositePodGroupEnabled: false,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg3WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg3"): framework.NewAbstractPodGroup(pg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "add same pod group again, feature enabled",
			isCompositePodGroupEnabled: true,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg1, pg1},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg1"): framework.NewAbstractPodGroup(pg1),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "add same pod group again, feature disabled",
			isCompositePodGroupEnabled: false,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg1, pg1},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg1"): framework.NewAbstractPodGroup(pg1),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "add same pod group with parent again, feature enabled",
			isCompositePodGroupEnabled: true,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg3WithParent, pg3WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg3"): framework.NewAbstractPodGroup(pg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.PodGroupKey("ns1", "pg3")),
			},
		},
		{
			name:                       "add same pod group with parent again, feature disabled",
			isCompositePodGroupEnabled: false,
			podGroupsToAdd:             []*schedulingv1beta1.PodGroup{pg3WithParent, pg3WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg3"): framework.NewAbstractPodGroup(pg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(tt.isCompositePodGroupEnabled)
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}
			for _, pg := range tt.podGroupsToAdd {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected podGroups (-want,+got)\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantChildren, wf.children); diff != "" {
				t.Errorf("Unexpected children (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_UpdatePodGroup(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").MinCount(1).Obj()
	updatedPG1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").MinCount(2).Obj()
	pg2 := st.MakePodGroup().Name("pg2").Namespace("ns1").UID("uid2").MinCount(1).Obj()

	tests := []struct {
		name             string
		initialPodGroups []*schedulingv1beta1.PodGroup
		podGroupToUpdate *schedulingv1beta1.PodGroup
		want             map[fwk.EntityKey]*framework.AbstractPodGroup
	}{
		{
			name:             "update existing pod group",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			podGroupToUpdate: updatedPG1,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg1"): framework.NewAbstractPodGroup(updatedPG1),
			},
		},
		{
			name:             "update non-existent pod group adds it",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			podGroupToUpdate: pg2,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg1"): framework.NewAbstractPodGroup(pg1),
				fwk.PodGroupKey("ns1", "pg2"): framework.NewAbstractPodGroup(pg2),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(true)
			for _, pg := range tt.initialPodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}

			wf.updateAbstractPodGroup(framework.NewAbstractPodGroup(tt.podGroupToUpdate))

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected podGroups (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_DeletePodGroup(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()
	pg2 := st.MakePodGroup().Name("pg2").Namespace("ns1").UID("uid2").Obj()
	pg3WithParent := st.MakePodGroup().Name("pg3").Namespace("ns1").UID("uid3").ParentCompositePodGroup("cpg1").Obj()
	pg4WithParent := st.MakePodGroup().Name("pg4").Namespace("ns1").UID("uid4").ParentCompositePodGroup("cpg1").Obj()
	cpgChild := st.MakeCompositePodGroup().Name("cpgChild").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()

	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").Obj()

	tests := []struct {
		name                       string
		isCompositePodGroupEnabled bool
		initialPodGroups           []*schedulingv1beta1.PodGroup
		initialCPGs                []*schedulingv1alpha3.CompositePodGroup
		podGroupToDelete           *schedulingv1beta1.PodGroup
		want                       map[fwk.EntityKey]*framework.AbstractPodGroup
		wantChildren               map[fwk.EntityKey]sets.Set[fwk.EntityKey]
	}{
		{
			name:                       "delete existing pod group",
			isCompositePodGroupEnabled: true,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2},
			podGroupToDelete:           pg1,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg2"): framework.NewAbstractPodGroup(pg2),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "delete non-existent pod group is no-op",
			isCompositePodGroupEnabled: true,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1},
			podGroupToDelete:           pg2,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg1"): framework.NewAbstractPodGroup(pg1),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "delete pod group with parent, where parent CPG does not exist in compositePodGroups (e.g. was just deleted)",
			isCompositePodGroupEnabled: true,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg3WithParent},
			podGroupToDelete:           pg3WithParent,
			want:                       map[fwk.EntityKey]*framework.AbstractPodGroup{},
			wantChildren:               map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "delete pod group with parent, where parent CPG exists in compositePodGroups",
			isCompositePodGroupEnabled: true,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg3WithParent},
			initialCPGs:                []*schedulingv1alpha3.CompositePodGroup{cpg1},
			podGroupToDelete:           pg3WithParent,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg1"): framework.NewAbstractCompositePodGroup(cpg1),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:                       "delete pod group with parent, parent has other pod group children",
			isCompositePodGroupEnabled: true,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg3WithParent, pg4WithParent},
			podGroupToDelete:           pg3WithParent,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pg4"): framework.NewAbstractPodGroup(pg4WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.PodGroupKey("ns1", "pg4")),
			},
		},
		{
			name:                       "delete pod group with parent, parent has other composite pod group children",
			isCompositePodGroupEnabled: true,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg3WithParent},
			initialCPGs:                []*schedulingv1alpha3.CompositePodGroup{cpgChild},
			podGroupToDelete:           pg3WithParent,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpgChild"): framework.NewAbstractCompositePodGroup(cpgChild),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.CompositePodGroupKey("ns1", "cpgChild")),
			},
		},
		{
			name:                       "delete pod group with parent, feature disabled",
			isCompositePodGroupEnabled: false,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg3WithParent},
			podGroupToDelete:           pg3WithParent,
			want:                       map[fwk.EntityKey]*framework.AbstractPodGroup{},
			wantChildren:               map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(tt.isCompositePodGroupEnabled)
			for _, pg := range tt.initialPodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			wf.deleteAbstractPodGroup(framework.NewAbstractPodGroup(tt.podGroupToDelete))

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected podGroups (-want,+got)\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantChildren, wf.children); diff != "" {
				t.Errorf("Unexpected children (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_GetPodGroup(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()

	tests := []struct {
		name                       string
		initialPodGroups           []*schedulingv1beta1.PodGroup
		podGroupLookup             *schedulingv1beta1.PodGroup
		wantPodGroup               *schedulingv1beta1.PodGroup
		isCompositePodGroupEnabled bool
	}{
		{
			name:                       "get existing pod group",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1},
			podGroupLookup:             pg1,
			wantPodGroup:               pg1,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "get non-existent pod group",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{},
			podGroupLookup:             pg1,
			wantPodGroup:               nil,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "get existing pod group (CPG=false)",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1},
			podGroupLookup:             pg1,
			wantPodGroup:               pg1,
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "get non-existent pod group (CPG=false)",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{},
			podGroupLookup:             pg1,
			wantPodGroup:               nil,
			isCompositePodGroupEnabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(tt.isCompositePodGroupEnabled)
			for _, pg := range tt.initialPodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}

			gotAPG, gotFound := wf.podGroups[framework.NewAbstractPodGroup(tt.podGroupLookup).GetKey()]
			var gotPG *schedulingv1beta1.PodGroup
			if gotFound && gotAPG.PodGroup != nil {
				gotPG = gotAPG.PodGroup
			} else {
				gotFound = false
			}
			if wantFound := tt.wantPodGroup != nil; gotFound != wantFound {
				t.Errorf("Expected found: %v, got: %v", wantFound, gotFound)
			}
			if diff := cmp.Diff(tt.wantPodGroup, gotPG); diff != "" {
				t.Errorf("Unexpected pod group (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_GetCompositePodGroup(t *testing.T) {
	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").Obj()

	tests := []struct {
		name                      string
		initialCompositePodGroups []*schedulingv1alpha3.CompositePodGroup
		cpgLookup                 *schedulingv1alpha3.CompositePodGroup
		wantCompositePodGroup     *schedulingv1alpha3.CompositePodGroup
	}{
		{
			name:                      "get existing composite pod group",
			initialCompositePodGroups: []*schedulingv1alpha3.CompositePodGroup{cpg1},
			cpgLookup:                 cpg1,
			wantCompositePodGroup:     cpg1,
		},
		{
			name:                      "get non-existent composite pod group",
			initialCompositePodGroups: []*schedulingv1alpha3.CompositePodGroup{},
			cpgLookup:                 cpg1,
			wantCompositePodGroup:     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(true)
			for _, cpg := range tt.initialCompositePodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			gotAPG, gotFound := wf.podGroups[framework.NewAbstractCompositePodGroup(tt.cpgLookup).GetKey()]
			var gotCPG *schedulingv1alpha3.CompositePodGroup
			if gotFound && gotAPG.CompositePodGroup != nil {
				gotCPG = gotAPG.CompositePodGroup
			} else {
				gotFound = false
			}
			if wantFound := tt.wantCompositePodGroup != nil; gotFound != wantFound {
				t.Errorf("Expected found: %v, got: %v", wantFound, gotFound)
			}
			if diff := cmp.Diff(tt.wantCompositePodGroup, gotCPG); diff != "" {
				t.Errorf("Unexpected composite pod group (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_AddCompositePodGroup(t *testing.T) {
	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").Obj()
	cpg2 := st.MakeCompositePodGroup().Name("cpg2").Namespace("ns1").Obj()
	cpg3WithParent := st.MakeCompositePodGroup().Name("cpg3").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()
	cpg4WithParent := st.MakeCompositePodGroup().Name("cpg4").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()

	pgChild := st.MakePodGroup().Name("pgChild").Namespace("ns1").UID("uid1").ParentCompositePodGroup("cpg1").Obj()

	tests := []struct {
		name         string
		initialPGs   []*schedulingv1beta1.PodGroup
		cpgsToAdd    []*schedulingv1alpha3.CompositePodGroup
		want         map[fwk.EntityKey]*framework.AbstractPodGroup
		wantChildren map[fwk.EntityKey]sets.Set[fwk.EntityKey]
	}{
		{
			name:      "add single composite pod group",
			cpgsToAdd: []*schedulingv1alpha3.CompositePodGroup{cpg1},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg1"): framework.NewAbstractCompositePodGroup(cpg1),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:      "add multiple composite pod groups",
			cpgsToAdd: []*schedulingv1alpha3.CompositePodGroup{cpg1, cpg2},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg1"): framework.NewAbstractCompositePodGroup(cpg1),
				fwk.CompositePodGroupKey("ns1", "cpg2"): framework.NewAbstractCompositePodGroup(cpg2),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:      "add composite pod group with parent, parent not in children",
			cpgsToAdd: []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg3"): framework.NewAbstractCompositePodGroup(cpg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.CompositePodGroupKey("ns1", "cpg3")),
			},
		},
		{
			name:      "add composite pod group with parent, parent already has other composite pod group child",
			cpgsToAdd: []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent, cpg4WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg3"): framework.NewAbstractCompositePodGroup(cpg3WithParent),
				fwk.CompositePodGroupKey("ns1", "cpg4"): framework.NewAbstractCompositePodGroup(cpg4WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.CompositePodGroupKey("ns1", "cpg3"), fwk.CompositePodGroupKey("ns1", "cpg4")),
			},
		},
		{
			name:       "add composite pod group with parent, parent already has pod group child",
			initialPGs: []*schedulingv1beta1.PodGroup{pgChild},
			cpgsToAdd:  []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pgChild"):       framework.NewAbstractPodGroup(pgChild),
				fwk.CompositePodGroupKey("ns1", "cpg3"): framework.NewAbstractCompositePodGroup(cpg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.PodGroupKey("ns1", "pgChild"), fwk.CompositePodGroupKey("ns1", "cpg3")),
			},
		},
		{
			name:      "add same composite pod group again",
			cpgsToAdd: []*schedulingv1alpha3.CompositePodGroup{cpg1, cpg1},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg1"): framework.NewAbstractCompositePodGroup(cpg1),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:      "add same composite pod group with parent again",
			cpgsToAdd: []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent, cpg3WithParent},
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg3"): framework.NewAbstractCompositePodGroup(cpg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.CompositePodGroupKey("ns1", "cpg3")),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(true)
			for _, pg := range tt.initialPGs {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}
			for _, cpg := range tt.cpgsToAdd {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected compositePodGroups (-want,+got)\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantChildren, wf.children); diff != "" {
				t.Errorf("Unexpected children (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_UpdateCompositePodGroup(t *testing.T) {
	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").MinGroupCount(1).Obj()
	updatedCPG1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").MinGroupCount(2).Obj()
	cpg2 := st.MakeCompositePodGroup().Name("cpg2").Namespace("ns1").MinGroupCount(1).Obj()

	tests := []struct {
		name        string
		initialCPGs []*schedulingv1alpha3.CompositePodGroup
		cpgToUpdate *schedulingv1alpha3.CompositePodGroup
		want        map[fwk.EntityKey]*framework.AbstractPodGroup
	}{
		{
			name:        "update existing composite pod group",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg1},
			cpgToUpdate: updatedCPG1,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg1"): framework.NewAbstractCompositePodGroup(updatedCPG1),
			},
		},
		{
			name:        "update non-existent composite pod group adds it",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg1},
			cpgToUpdate: cpg2,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg1"): framework.NewAbstractCompositePodGroup(cpg1),
				fwk.CompositePodGroupKey("ns1", "cpg2"): framework.NewAbstractCompositePodGroup(cpg2),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(true)
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			wf.updateAbstractPodGroup(framework.NewAbstractCompositePodGroup(tt.cpgToUpdate))

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected compositePodGroups (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_DeleteCompositePodGroup(t *testing.T) {
	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").Obj()
	cpg3WithParent := st.MakeCompositePodGroup().Name("cpg3").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()
	cpg4WithParent := st.MakeCompositePodGroup().Name("cpg4").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()

	pgChild := st.MakePodGroup().Name("pgChild").Namespace("ns1").UID("uid1").ParentCompositePodGroup("cpg1").Obj()

	cpgMid := st.MakeCompositePodGroup().Name("cpgMid").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()
	pgLeaf := st.MakePodGroup().Name("pgLeaf").Namespace("ns1").UID("uidLeaf").ParentCompositePodGroup("cpgMid").Obj()

	tests := []struct {
		name         string
		initialPGs   []*schedulingv1beta1.PodGroup
		initialCPGs  []*schedulingv1alpha3.CompositePodGroup
		cpgToDelete  *schedulingv1alpha3.CompositePodGroup
		want         map[fwk.EntityKey]*framework.AbstractPodGroup
		wantChildren map[fwk.EntityKey]sets.Set[fwk.EntityKey]
	}{
		{
			name:         "delete existing composite pod group without parent",
			initialCPGs:  []*schedulingv1alpha3.CompositePodGroup{cpg1},
			cpgToDelete:  cpg1,
			want:         map[fwk.EntityKey]*framework.AbstractPodGroup{},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:         "delete composite pod group with parent, cleans up children map",
			initialCPGs:  []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent},
			cpgToDelete:  cpg3WithParent,
			want:         map[fwk.EntityKey]*framework.AbstractPodGroup{},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{},
		},
		{
			name:        "delete composite pod group with parent, parent has other pod group child",
			initialPGs:  []*schedulingv1beta1.PodGroup{pgChild},
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent},
			cpgToDelete: cpg3WithParent,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pgChild"): framework.NewAbstractPodGroup(pgChild),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.PodGroupKey("ns1", "pgChild")),
			},
		},
		{
			name:        "delete composite pod group with parent, parent has other composite pod group child",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent, cpg4WithParent},
			cpgToDelete: cpg3WithParent,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg4"): framework.NewAbstractCompositePodGroup(cpg4WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.CompositePodGroupKey("ns1", "cpg4")),
			},
		},
		{
			name:        "delete composite pod group with parent, parent has both other pg and cpg children",
			initialPGs:  []*schedulingv1beta1.PodGroup{pgChild},
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent, cpg4WithParent},
			cpgToDelete: cpg3WithParent,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.PodGroupKey("ns1", "pgChild"):       framework.NewAbstractPodGroup(pgChild),
				fwk.CompositePodGroupKey("ns1", "cpg4"): framework.NewAbstractCompositePodGroup(cpg4WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.PodGroupKey("ns1", "pgChild"), fwk.CompositePodGroupKey("ns1", "cpg4")),
			},
		},
		{
			name:        "delete mid cpg from root-mid-leaf hierarchy",
			initialPGs:  []*schedulingv1beta1.PodGroup{pgLeaf},
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg1, cpgMid},
			cpgToDelete: cpgMid,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg1"): framework.NewAbstractCompositePodGroup(cpg1),
				fwk.PodGroupKey("ns1", "pgLeaf"):        framework.NewAbstractPodGroup(pgLeaf),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpgMid"): sets.New(fwk.PodGroupKey("ns1", "pgLeaf")),
			},
		},
		{
			name:        "delete non-existent composite pod group",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg3WithParent},
			cpgToDelete: cpg1,
			want: map[fwk.EntityKey]*framework.AbstractPodGroup{
				fwk.CompositePodGroupKey("ns1", "cpg3"): framework.NewAbstractCompositePodGroup(cpg3WithParent),
			},
			wantChildren: map[fwk.EntityKey]sets.Set[fwk.EntityKey]{
				fwk.CompositePodGroupKey("ns1", "cpg1"): sets.New(fwk.CompositePodGroupKey("ns1", "cpg3")),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(true)
			for _, pg := range tt.initialPGs {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			wf.deleteAbstractPodGroup(framework.NewAbstractCompositePodGroup(tt.cpgToDelete))

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected compositePodGroups (-want,+got)\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantChildren, wf.children); diff != "" {
				t.Errorf("Unexpected children (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_GetRootLookupInfoForPod(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()
	pg2WithParent := st.MakePodGroup().Name("pg2").Namespace("ns1").UID("uid2").ParentCompositePodGroup("cpg1").Obj()
	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").Obj()

	podWithPG1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	podWithPG2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg2").Obj()
	podWithNonExistentPG := st.MakePod().Name("pod3").Namespace("ns1").PodGroupName("pg3").Obj()

	tests := []struct {
		name                       string
		initialPodGroups           []*schedulingv1beta1.PodGroup
		initialCPGs                []*schedulingv1alpha3.CompositePodGroup
		pod                        *v1.Pod
		wantInfo                   *framework.QueuedPodGroupInfo
		isCompositePodGroupEnabled bool
	}{
		{
			name:             "pod belongs to existing standalone pod group",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			pod:              podWithPG1,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "pg1",
					Type:      fwk.PodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "pod belongs to pod group with parent cpg",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg2WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1},
			pod:              podWithPG2,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "cpg1",
					Type:      fwk.CompositePodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "pod belongs to non-existent pod group",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1},
			pod:                        podWithNonExistentPG,
			wantInfo:                   nil,
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "pod belongs to existing standalone pod group (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			pod:              podWithPG1,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "pg1",
					Type:      fwk.PodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: false,
		},
		{
			name:             "pod belongs to pod group with parent cpg (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg2WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1},
			pod:              podWithPG2,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "pg2",
					Type:      fwk.PodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "pod belongs to non-existent pod group (CPG=false)",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1},
			pod:                        podWithNonExistentPG,
			wantInfo:                   nil,
			isCompositePodGroupEnabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(tt.isCompositePodGroupEnabled)
			for _, pg := range tt.initialPodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			gotInfo, gotFound := wf.getRootLookupInfoForPod(tt.pod)
			if wantFound := tt.wantInfo != nil; gotFound != wantFound {
				t.Errorf("Expected found: %v, got: %v", wantFound, gotFound)
			}
			if diff := cmp.Diff(tt.wantInfo, gotInfo); diff != "" {
				t.Errorf("Unexpected QueuedPodGroupInfo (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_GetRootLookupInfoForPodGroup(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()
	pg2WithParent := st.MakePodGroup().Name("pg2").Namespace("ns1").UID("uid2").ParentCompositePodGroup("cpg1").Obj()
	pg3WithNonExistentParent := st.MakePodGroup().Name("pg3").Namespace("ns1").UID("uid3").ParentCompositePodGroup("cpg-missing").Obj()

	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").Obj()

	tests := []struct {
		name                       string
		initialPodGroups           []*schedulingv1beta1.PodGroup
		initialCPGs                []*schedulingv1alpha3.CompositePodGroup
		podGroup                   *schedulingv1beta1.PodGroup
		wantInfo                   *framework.QueuedPodGroupInfo
		isCompositePodGroupEnabled bool
	}{
		{
			name:             "standalone pod group",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			podGroup:         pg1,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "pg1",
					Type:      fwk.PodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "pod group with existing parent cpg",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg2WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1},
			podGroup:         pg2WithParent,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "cpg1",
					Type:      fwk.CompositePodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "pod group with non-existent parent cpg",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg3WithNonExistentParent},
			podGroup:                   pg3WithNonExistentParent,
			wantInfo:                   nil,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "non-existent pod group",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{},
			podGroup:                   pg1,
			wantInfo:                   nil,
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "standalone pod group (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			podGroup:         pg1,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "pg1",
					Type:      fwk.PodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: false,
		},
		{
			name:             "pod group with existing parent cpg (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg2WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1},
			podGroup:         pg2WithParent,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "pg2",
					Type:      fwk.PodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: false,
		},
		{
			name:             "pod group with non-existent parent cpg (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg3WithNonExistentParent},
			podGroup:         pg3WithNonExistentParent,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "pg3",
					Type:      fwk.PodGroupKeyType,
				},
			},
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "non-existent pod group (CPG=false)",
			initialPodGroups:           []*schedulingv1beta1.PodGroup{},
			podGroup:                   pg1,
			wantInfo:                   nil,
			isCompositePodGroupEnabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(tt.isCompositePodGroupEnabled)
			for _, pg := range tt.initialPodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			gotInfo, gotFound := wf.getRootLookupInfo(framework.NewAbstractPodGroup(tt.podGroup))
			if wantFound := tt.wantInfo != nil; gotFound != wantFound {
				t.Errorf("Expected found: %v, got: %v", wantFound, gotFound)
			}
			if diff := cmp.Diff(tt.wantInfo, gotInfo); diff != "" {
				t.Errorf("Unexpected QueuedPodGroupInfo (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_GetRootLookupInfoForCPG(t *testing.T) {
	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").Obj()
	cpg2WithParent := st.MakeCompositePodGroup().Name("cpg2").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()
	cpg3WithCycle := st.MakeCompositePodGroup().Name("cpg3").Namespace("ns1").ParentCompositePodGroup("cpg4").Obj()
	cpg4WithCycle := st.MakeCompositePodGroup().Name("cpg4").Namespace("ns1").ParentCompositePodGroup("cpg3").Obj()

	tests := []struct {
		name        string
		initialCPGs []*schedulingv1alpha3.CompositePodGroup
		cpg         *schedulingv1alpha3.CompositePodGroup
		wantInfo    *framework.QueuedPodGroupInfo
	}{
		{
			name:        "standalone cpg",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg1},
			cpg:         cpg1,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "cpg1",
					Type:      fwk.CompositePodGroupKeyType,
				},
			},
		},
		{
			name:        "cpg with parent",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg1, cpg2WithParent},
			cpg:         cpg2WithParent,
			wantInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: "ns1",
					Name:      "cpg1",
					Type:      fwk.CompositePodGroupKeyType,
				},
			},
		},
		{
			name:        "cpg with parent missing",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg2WithParent},
			cpg:         cpg2WithParent,
			wantInfo:    nil,
		},
		{
			name:        "cpg cycle detection",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{cpg3WithCycle, cpg4WithCycle},
			cpg:         cpg3WithCycle,
			wantInfo:    nil, // cycle returns nil, false
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(true)
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			gotInfo, gotFound := wf.getRootLookupInfo(framework.NewAbstractCompositePodGroup(tt.cpg))
			if wantFound := tt.wantInfo != nil; gotFound != wantFound {
				t.Errorf("Expected found: %v, got: %v", wantFound, gotFound)
			}
			if diff := cmp.Diff(tt.wantInfo, gotInfo); diff != "" {
				t.Errorf("Unexpected QueuedPodGroupInfo (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_GetLeafPodGroups(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()
	pg2WithParent := st.MakePodGroup().Name("pg2").Namespace("ns1").UID("uid2").ParentCompositePodGroup("cpg1").Obj()
	pg3WithParent := st.MakePodGroup().Name("pg3").Namespace("ns1").UID("uid3").ParentCompositePodGroup("cpg2").Obj()

	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").Obj()
	cpg2WithParent := st.MakeCompositePodGroup().Name("cpg2").Namespace("ns1").ParentCompositePodGroup("cpg1").Obj()

	tests := []struct {
		name                       string
		initialPodGroups           []*schedulingv1beta1.PodGroup
		initialCPGs                []*schedulingv1alpha3.CompositePodGroup
		rootLookupInfo             *framework.QueuedPodGroupInfo
		wantLeaves                 []*schedulingv1beta1.PodGroup
		isCompositePodGroupEnabled bool
	}{
		{
			name:             "single pod group lookup",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			rootLookupInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{Namespace: "ns1", Name: "pg1", Type: fwk.PodGroupKeyType},
			},
			wantLeaves:                 []*schedulingv1beta1.PodGroup{pg1},
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "cpg with direct pod group child",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg2WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1},
			rootLookupInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{Namespace: "ns1", Name: "cpg1", Type: fwk.CompositePodGroupKeyType},
			},
			wantLeaves:                 []*schedulingv1beta1.PodGroup{pg2WithParent},
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "cpg with nested pod group child",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg2WithParent, pg3WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1, cpg2WithParent},
			rootLookupInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{Namespace: "ns1", Name: "cpg1", Type: fwk.CompositePodGroupKeyType},
			},
			wantLeaves:                 []*schedulingv1beta1.PodGroup{pg2WithParent, pg3WithParent},
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "missing root",
			initialPodGroups: []*schedulingv1beta1.PodGroup{},
			rootLookupInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{Namespace: "ns1", Name: "missing", Type: fwk.CompositePodGroupKeyType},
			},
			wantLeaves:                 nil,
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "single pod group lookup (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			rootLookupInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{Namespace: "ns1", Name: "pg1", Type: fwk.PodGroupKeyType},
			},
			wantLeaves:                 []*schedulingv1beta1.PodGroup{pg1},
			isCompositePodGroupEnabled: false,
		},
		{
			name:             "pg with direct pod group child lookup (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg2WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1},
			rootLookupInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{Namespace: "ns1", Name: "pg2", Type: fwk.PodGroupKeyType},
			},
			wantLeaves:                 []*schedulingv1beta1.PodGroup{pg2WithParent},
			isCompositePodGroupEnabled: false,
		},
		{
			name:             "missing root (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{},
			rootLookupInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{Namespace: "ns1", Name: "missing", Type: fwk.CompositePodGroupKeyType},
			},
			wantLeaves:                 nil,
			isCompositePodGroupEnabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(tt.isCompositePodGroupEnabled)
			for _, pg := range tt.initialPodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			logger, _ := ktesting.NewTestContext(t)
			gotLeaves := wf.getLeafPodGroups(logger, tt.rootLookupInfo)

			wantMap := make(map[string]*schedulingv1beta1.PodGroup)
			for _, l := range tt.wantLeaves {
				wantMap[string(l.UID)] = l
			}
			gotMap := make(map[string]*schedulingv1beta1.PodGroup)
			for _, l := range gotLeaves {
				gotMap[string(l.UID)] = l
			}

			if diff := cmp.Diff(wantMap, gotMap); diff != "" {
				t.Errorf("Unexpected leaf pod groups (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_BuildPodGroupInfoForPG(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()

	tests := []struct {
		name                       string
		initialPodGroups           []*schedulingv1beta1.PodGroup
		pg                         *schedulingv1beta1.PodGroup
		wantInfo                   *framework.PodGroupInfo
		isCompositePodGroupEnabled bool
	}{
		{
			name:             "build info for existing pod group",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			pg:               pg1,
			wantInfo: &framework.PodGroupInfo{
				Namespace: "ns1",
				Name:      "pg1",
				Type:      fwk.PodGroupKeyType,
				PodGroup:  pg1,
				Children:  []*framework.PodGroupInfo{},
			},
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "build info for existing pod group (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1},
			pg:               pg1,
			wantInfo: &framework.PodGroupInfo{
				Namespace: "ns1",
				Name:      "pg1",
				Type:      fwk.PodGroupKeyType,
				PodGroup:  pg1,
				Children:  []*framework.PodGroupInfo{},
			},
			isCompositePodGroupEnabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(tt.isCompositePodGroupEnabled)
			for _, pg := range tt.initialPodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}

			logger, _ := ktesting.NewTestContext(t)
			visited := sets.New[fwk.EntityKey]()
			gotInfo := wf.buildPodGroupInfo(logger, framework.NewAbstractPodGroup(tt.pg), visited)

			if diff := cmp.Diff(tt.wantInfo, gotInfo); diff != "" {
				t.Errorf("Unexpected PodGroupInfo (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_BuildPodGroupInfoForCPG(t *testing.T) {
	pg1WithParent := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").MinCount(2).ParentCompositePodGroup("cpg1").Obj()
	cpg1 := st.MakeCompositePodGroup().Name("cpg1").Namespace("ns1").MinGroupCount(1).Obj()

	tests := []struct {
		name                       string
		initialPodGroups           []*schedulingv1beta1.PodGroup
		initialCPGs                []*schedulingv1alpha3.CompositePodGroup
		cpg                        *schedulingv1alpha3.CompositePodGroup
		wantInfo                   *framework.PodGroupInfo
		isCompositePodGroupEnabled bool
	}{
		{
			name:             "build info for cpg with child pg",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1},
			cpg:              cpg1,
			wantInfo: &framework.PodGroupInfo{
				Namespace:         "ns1",
				Name:              "cpg1",
				Type:              fwk.CompositePodGroupKeyType,
				CompositePodGroup: cpg1,
				Children: []*framework.PodGroupInfo{
					{
						Namespace: "ns1",
						Name:      "pg1",
						Type:      fwk.PodGroupKeyType,
						PodGroup:  pg1WithParent,
						Children:  []*framework.PodGroupInfo{},
					},
				},
			},
			isCompositePodGroupEnabled: true,
		},
		{
			name:             "build info for cpg with child pg (CPG=false)",
			initialPodGroups: []*schedulingv1beta1.PodGroup{pg1WithParent},
			initialCPGs:      []*schedulingv1alpha3.CompositePodGroup{cpg1},
			cpg:              cpg1,
			wantInfo: &framework.PodGroupInfo{
				Namespace:         "ns1",
				Name:              "cpg1",
				Type:              fwk.CompositePodGroupKeyType,
				CompositePodGroup: cpg1,
				Children:          []*framework.PodGroupInfo{},
			},
			isCompositePodGroupEnabled: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest(tt.isCompositePodGroupEnabled)
			for _, pg := range tt.initialPodGroups {
				wf.addAbstractPodGroup(framework.NewAbstractPodGroup(pg))
			}
			for _, cpg := range tt.initialCPGs {
				wf.addAbstractPodGroup(framework.NewAbstractCompositePodGroup(cpg))
			}

			logger, _ := ktesting.NewTestContext(t)
			visited := sets.New[fwk.EntityKey]()
			gotInfo := wf.buildPodGroupInfo(logger, framework.NewAbstractCompositePodGroup(tt.cpg), visited)

			// Note: Children are sorted by name in buildPodGroupInfoForCPG, so it is deterministic.
			if diff := cmp.Diff(tt.wantInfo, gotInfo); diff != "" {
				t.Errorf("Unexpected PodGroupInfo (-want,+got)\n%s", diff)
			}
		})
	}
}
