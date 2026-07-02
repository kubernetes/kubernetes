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
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	"k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestIncompletePodGroupPods_Add(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("uid1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").UID("uid2").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").UID("uid3").PodGroupName("pg2").Obj()

	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}
	pInfo3 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod3)}

	tests := []struct {
		name      string
		podsToAdd []*framework.QueuedPodInfo
		want      map[string]map[string]*framework.QueuedPodInfo
	}{
		{
			name:      "pod group with multiple pods",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			want: map[string]map[string]*framework.QueuedPodInfo{
				"pg/ns1/pg1": {
					"ns1/pod1": pInfo1,
					"ns1/pod2": pInfo2,
				},
			},
		},
		{
			name:      "two pod groups",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo3},
			want: map[string]map[string]*framework.QueuedPodInfo{
				"pg/ns1/pg1": {
					"ns1/pod1": pInfo1,
				},
				"pg/ns1/pg2": {
					"ns1/pod3": pInfo3,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ie := newIncompletePodGroupPods()
			for _, pInfo := range tt.podsToAdd {
				ie.add(pInfo)
			}

			if diff := cmp.Diff(tt.want, ie.podGroupToPodInfos, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected podGroupToPodInfos (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestIncompletePodGroupPods_Get(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("uid1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").UID("uid2").PodGroupName("pg1").Obj()

	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}

	tests := []struct {
		name      string
		podsToAdd []*framework.QueuedPodInfo
		targetPod *v1.Pod
		want      *framework.QueuedPodInfo
	}{
		{
			name:      "get existing pod in pod group",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			targetPod: pod1,
			want:      pInfo1,
		},
		{
			name:      "get non-existent pod in existing pod group",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1},
			targetPod: pod2,
			want:      nil,
		},
		{
			name:      "get non-existent pod in non-existing pod group",
			podsToAdd: []*framework.QueuedPodInfo{},
			targetPod: pod2,
			want:      nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ie := newIncompletePodGroupPods()
			for _, pInfo := range tt.podsToAdd {
				ie.add(pInfo)
			}

			got := ie.get(tt.targetPod)
			if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected pod info (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestIncompletePodGroupPods_Has(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("uid1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").UID("uid2").PodGroupName("pg1").Obj()

	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}

	tests := []struct {
		name      string
		podsToAdd []*framework.QueuedPodInfo
		targetPod *v1.Pod
		want      bool
	}{
		{
			name:      "has existing pod in pod group",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			targetPod: pod1,
			want:      true,
		},
		{
			name:      "doesn't have non-existent pod in existing pod group",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1},
			targetPod: pod2,
			want:      false,
		},
		{
			name:      "doesn't have non-existent pod in non-existing pod group",
			podsToAdd: []*framework.QueuedPodInfo{},
			targetPod: pod2,
			want:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ie := newIncompletePodGroupPods()
			for _, pInfo := range tt.podsToAdd {
				ie.add(pInfo)
			}

			got := ie.has(tt.targetPod)
			if got != tt.want {
				t.Errorf("Expected has() to return %v, got %v", tt.want, got)
			}
		})
	}
}

func TestIncompletePodGroupPods_Update(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("uid1").PodGroupName("pg1").Obj()
	updatedPod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("uid1").Label("updated", "true").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").UID("uid2").PodGroupName("pg1").Obj()

	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	updatedPodInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, updatedPod1)}

	tests := []struct {
		name        string
		podsToAdd   []*framework.QueuedPodInfo
		updatePod   *v1.Pod
		wantUpdated *framework.QueuedPodInfo
	}{
		{
			name:        "update existing pod",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1},
			updatePod:   updatedPod1,
			wantUpdated: updatedPodInfo1,
		},
		{
			name:        "update non-existent pod",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1},
			updatePod:   pod2,
			wantUpdated: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ie := newIncompletePodGroupPods()
			for _, pInfo := range tt.podsToAdd {
				ie.add(pInfo.DeepCopy())
			}

			got := ie.update(tt.updatePod)
			if diff := cmp.Diff(tt.wantUpdated, got, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected updated pod info (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestIncompletePodGroupPods_Delete(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("uid1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").UID("uid2").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").UID("uid3").PodGroupName("pg1").Obj()

	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}

	tests := []struct {
		name        string
		podsToAdd   []*framework.QueuedPodInfo
		podToDelete *v1.Pod
		wantDeleted *framework.QueuedPodInfo
		want        map[string]*framework.QueuedPodInfo
	}{
		{
			name:        "delete non-existent pod from existing pod group",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1},
			podToDelete: pod3,
			wantDeleted: nil,
			want: map[string]*framework.QueuedPodInfo{
				"ns1/pod1": pInfo1,
			},
		},
		{
			name:        "delete pod from non-existent pod group",
			podsToAdd:   []*framework.QueuedPodInfo{},
			podToDelete: pod3,
			wantDeleted: nil,
			want:        nil,
		},
		{
			name:        "delete one pod from group with multiple pods",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1, pInfo2},
			podToDelete: pod1,
			wantDeleted: pInfo1,
			want: map[string]*framework.QueuedPodInfo{
				"ns1/pod2": pInfo2,
			},
		},
		{
			name:        "delete last pod in group cleans up group key",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1},
			podToDelete: pod1,
			wantDeleted: pInfo1,
			want:        nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ie := newIncompletePodGroupPods()
			for _, pInfo := range tt.podsToAdd {
				ie.add(pInfo.DeepCopy())
			}

			gotDeleted := ie.delete(tt.podToDelete)
			if diff := cmp.Diff(tt.wantDeleted, gotDeleted, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected deleted pod info (-want,+got)\n%s", diff)
			}

			gotRemaining := ie.podGroupToPodInfos[podGroupKeyForPod(tt.podToDelete)]
			if diff := cmp.Diff(tt.want, gotRemaining, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected remaining pod infos (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestIncompletePodGroupPods_Clear(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("uid1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").UID("uid2").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").UID("uid3").PodGroupName("pg2").Obj()

	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}
	pInfo3 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod3)}

	podGroup1 := st.MakePodGroup().Name("pg1").Namespace("ns1").UID("pg1").Obj()
	podGroup3 := st.MakePodGroup().Name("pg3").Namespace("ns1").UID("pg3").Obj()

	tests := []struct {
		name        string
		podsToAdd   []*framework.QueuedPodInfo
		clearGroup  *v1alpha3.PodGroup
		wantCleared sets.Set[*framework.QueuedPodInfo]
		want        map[string]map[string]*framework.QueuedPodInfo
	}{
		{
			name:        "clear existing pod group",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1, pInfo2, pInfo3},
			clearGroup:  podGroup1,
			wantCleared: sets.New(pInfo1, pInfo2),
			want: map[string]map[string]*framework.QueuedPodInfo{
				"pg/ns1/pg2": {
					"ns1/pod3": pInfo3,
				},
			},
		},
		{
			name:        "clear non-existent pod group",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1},
			clearGroup:  podGroup3,
			wantCleared: nil,
			want: map[string]map[string]*framework.QueuedPodInfo{
				"pg/ns1/pg1": {
					"ns1/pod1": pInfo1,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ie := newIncompletePodGroupPods()
			for _, pInfo := range tt.podsToAdd {
				ie.add(pInfo)
			}

			gotCleared := ie.clear(tt.clearGroup)
			gotClearedSet := sets.New(gotCleared...)
			if diff := cmp.Diff(tt.wantCleared, gotClearedSet, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected cleared pod infos (-want,+got)\n%s", diff)
			}

			if diff := cmp.Diff(tt.want, ie.podGroupToPodInfos, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected remaining podGroupToPodInfos (-want,+got)\n%s", diff)
			}
		})
	}
}
