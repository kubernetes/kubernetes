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
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestWorkloadForest_AddPodGroup(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()
	pg2 := st.MakePodGroup().Name("pg2").Namespace("ns1").UID("uid2").Obj()

	tests := []struct {
		name           string
		podGroupsToAdd []*schedulingv1alpha3.PodGroup
		want           map[string]*schedulingv1alpha3.PodGroup
	}{
		{
			name:           "add single pod group",
			podGroupsToAdd: []*schedulingv1alpha3.PodGroup{pg1},
			want: map[string]*schedulingv1alpha3.PodGroup{
				"pg/ns1/pg1": pg1,
			},
		},
		{
			name:           "add multiple pod groups",
			podGroupsToAdd: []*schedulingv1alpha3.PodGroup{pg1, pg2},
			want: map[string]*schedulingv1alpha3.PodGroup{
				"pg/ns1/pg1": pg1,
				"pg/ns1/pg2": pg2,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest()
			for _, pg := range tt.podGroupsToAdd {
				wf.addPodGroup(pg)
			}

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected podGroups (-want,+got)\n%s", diff)
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
		initialPodGroups []*schedulingv1alpha3.PodGroup
		podGroupToUpdate *schedulingv1alpha3.PodGroup
		want             map[string]*schedulingv1alpha3.PodGroup
	}{
		{
			name:             "update existing pod group",
			initialPodGroups: []*schedulingv1alpha3.PodGroup{pg1},
			podGroupToUpdate: updatedPG1,
			want: map[string]*schedulingv1alpha3.PodGroup{
				"pg/ns1/pg1": updatedPG1,
			},
		},
		{
			name:             "update non-existent pod group adds it",
			initialPodGroups: []*schedulingv1alpha3.PodGroup{pg1},
			podGroupToUpdate: pg2,
			want: map[string]*schedulingv1alpha3.PodGroup{
				"pg/ns1/pg1": pg1,
				"pg/ns1/pg2": pg2,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest()
			for _, pg := range tt.initialPodGroups {
				wf.addPodGroup(pg)
			}

			wf.updatePodGroup(tt.podGroupToUpdate)

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected podGroups (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_DeletePodGroup(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()
	pg2 := st.MakePodGroup().Name("pg2").Namespace("ns1").UID("uid2").Obj()

	tests := []struct {
		name             string
		initialPodGroups []*schedulingv1alpha3.PodGroup
		podGroupToDelete *schedulingv1alpha3.PodGroup
		want             map[string]*schedulingv1alpha3.PodGroup
	}{
		{
			name:             "delete existing pod group",
			initialPodGroups: []*schedulingv1alpha3.PodGroup{pg1, pg2},
			podGroupToDelete: pg1,
			want: map[string]*schedulingv1alpha3.PodGroup{
				"pg/ns1/pg2": pg2,
			},
		},
		{
			name:             "delete non-existent pod group is no-op",
			initialPodGroups: []*schedulingv1alpha3.PodGroup{pg1},
			podGroupToDelete: pg2,
			want: map[string]*schedulingv1alpha3.PodGroup{
				"pg/ns1/pg1": pg1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest()
			for _, pg := range tt.initialPodGroups {
				wf.addPodGroup(pg)
			}

			wf.deletePodGroup(tt.podGroupToDelete)

			if diff := cmp.Diff(tt.want, wf.podGroups); diff != "" {
				t.Errorf("Unexpected podGroups (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_GetRootForPod(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()
	podWithPG1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	podWithNonExistentPG := st.MakePod().Name("pod3").Namespace("ns1").PodGroupName("pg2").Obj()

	tests := []struct {
		name             string
		initialPodGroups []*schedulingv1alpha3.PodGroup
		pod              *v1.Pod
		wantPodGroup     *schedulingv1alpha3.PodGroup
	}{
		{
			name:             "pod belongs to existing pod group",
			initialPodGroups: []*schedulingv1alpha3.PodGroup{pg1},
			pod:              podWithPG1,
			wantPodGroup:     pg1,
		},
		{
			name:             "pod belongs to non-existent pod group",
			initialPodGroups: []*schedulingv1alpha3.PodGroup{pg1},
			pod:              podWithNonExistentPG,
			wantPodGroup:     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest()
			for _, pg := range tt.initialPodGroups {
				wf.addPodGroup(pg)
			}

			gotPG, gotFound := wf.getRootForPod(tt.pod)
			if wantFound := tt.wantPodGroup != nil; gotFound != wantFound {
				t.Errorf("Expected found: %v, got: %v", wantFound, gotFound)
			}
			if diff := cmp.Diff(tt.wantPodGroup, gotPG); diff != "" {
				t.Errorf("Unexpected pod group (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestWorkloadForest_GetPodGroup(t *testing.T) {
	pg1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("uid1").Obj()

	tests := []struct {
		name             string
		initialPodGroups []*schedulingv1alpha3.PodGroup
		podGroupLookup   *schedulingv1alpha3.PodGroup
		wantPodGroup     *schedulingv1alpha3.PodGroup
	}{
		{
			name:             "get existing pod group",
			initialPodGroups: []*schedulingv1alpha3.PodGroup{pg1},
			podGroupLookup:   pg1,
			wantPodGroup:     pg1,
		},
		{
			name:             "get non-existent pod group",
			initialPodGroups: []*schedulingv1alpha3.PodGroup{},
			podGroupLookup:   pg1,
			wantPodGroup:     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wf := newWorkloadForest()
			for _, pg := range tt.initialPodGroups {
				wf.addPodGroup(pg)
			}

			gotPG, gotFound := wf.getPodGroup(tt.podGroupLookup)
			if wantFound := tt.wantPodGroup != nil; gotFound != wantFound {
				t.Errorf("Expected found: %v, got: %v", wantFound, gotFound)
			}
			if diff := cmp.Diff(tt.wantPodGroup, gotPG); diff != "" {
				t.Errorf("Unexpected pod group (-want,+got)\n%s", diff)
			}
		})
	}
}
