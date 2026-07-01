/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPendingPodGroupMemberPods_Add(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}

	tests := []struct {
		name      string
		podsToAdd []*framework.QueuedPodInfo
		want      map[string]*framework.QueuedPodInfo
	}{
		{
			name:      "multiple pods",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			want: map[string]*framework.QueuedPodInfo{
				"ns1/pod1": pInfo1,
				"ns1/pod2": pInfo2,
			},
		},
		{
			name:      "add existing pod",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo1},
			want: map[string]*framework.QueuedPodInfo{
				"ns1/pod1": pInfo1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(pInfo)
			}

			for _, pInfo := range tt.podsToAdd {
				if !ppm.has(pInfo.Pod) {
					t.Errorf("Expected pod %s to be present", pInfo.Pod.Name)
				}
				gotPod := ppm.get(pInfo.Pod)
				if diff := cmp.Diff(pInfo, gotPod, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
					t.Errorf("Unexpected pod info for %s (-want,+got)\n%s", pInfo.Pod.Name, diff)
				}
			}
		})
	}
}

func TestPendingPodGroupMemberPods_Get(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").PodGroupName("pg3").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}

	tests := []struct {
		name      string
		podsToAdd []*framework.QueuedPodInfo
		targetPod *v1.Pod
		want      *framework.QueuedPodInfo
	}{
		{
			name:      "find existing pod",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			targetPod: pod1,
			want:      pInfo1,
		},
		{
			name:      "pod not found",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			targetPod: pod3,
			want:      nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(pInfo)
			}

			got := ppm.get(tt.targetPod)
			if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected pod info (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestPendingPodGroupMemberPods_Update(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	updatedPod1 := st.MakePod().Name("pod1").Namespace("ns1").Label("updated", "true").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()

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
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(pInfo.DeepCopy())
			}

			pInfo := ppm.update(tt.updatePod)
			if diff := cmp.Diff(tt.wantUpdated, pInfo, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected updated pod (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestPendingPodGroupMemberPods_Delete(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}

	tests := []struct {
		name          string
		podsToAdd     []*framework.QueuedPodInfo
		podToDelete   *v1.Pod
		wantDeleted   *framework.QueuedPodInfo
		wantRemaining sets.Set[*framework.QueuedPodInfo]
	}{
		{
			name:          "delete existing pod",
			podsToAdd:     []*framework.QueuedPodInfo{pInfo1, pInfo2},
			podToDelete:   pod1,
			wantDeleted:   pInfo1,
			wantRemaining: sets.New(pInfo2),
		},
		{
			name:          "delete non-existent pod",
			podsToAdd:     []*framework.QueuedPodInfo{pInfo1},
			podToDelete:   pod2,
			wantDeleted:   nil,
			wantRemaining: sets.New(pInfo1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(pInfo)
			}

			gotDeleted := ppm.delete(tt.podToDelete)
			if diff := cmp.Diff(tt.wantDeleted, gotDeleted, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected deleted pod (-want,+got)\n%s", diff)
			}

			gotRemaining := sets.New[*framework.QueuedPodInfo]()
			for _, pInfo := range ppm.keyToPod {
				gotRemaining.Insert(pInfo)
			}
			if diff := cmp.Diff(tt.wantRemaining, gotRemaining, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected remaining pods (-want,+got)\n%s", diff)
			}
		})
	}
}

func TestPendingPodGroupMemberPods_Clear(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}

	tests := []struct {
		name      string
		podsToAdd []*framework.QueuedPodInfo
		want      sets.Set[*framework.QueuedPodInfo]
	}{
		{
			name:      "clear all pods",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			want:      sets.New(pInfo1, pInfo2),
		},
		{
			name:      "clear empty",
			podsToAdd: []*framework.QueuedPodInfo{},
			want:      nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(pInfo)
			}

			gotCleared := ppm.clear()
			gotClearedSet := sets.New(gotCleared...)
			if diff := cmp.Diff(tt.want, gotClearedSet, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected cleared pods (-want,+got)\n%s", diff)
			}

			if ppm.len() != 0 {
				t.Errorf("Expected map to be empty, got length %d", ppm.len())
			}
		})
	}
}

func TestPendingPodGroupMemberPods_Len(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}

	tests := []struct {
		name      string
		podsToAdd []*framework.QueuedPodInfo
		wantLen   int
	}{
		{
			name:      "empty map",
			podsToAdd: nil,
			wantLen:   0,
		},
		{
			name:      "multiple pods",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			wantLen:   2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(pInfo)
			}

			if got := ppm.len(); got != tt.wantLen {
				t.Errorf("Expected length %v, got %v", tt.wantLen, got)
			}
		})
	}
}
