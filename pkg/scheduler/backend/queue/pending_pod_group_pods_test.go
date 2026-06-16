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
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPendingPodGroupMemberPods_Add(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").PodGroupName("pg3").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}
	pInfo3 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod3)}
	pgInfo1 := newQueuedPodGroupInfoForLookup(pod1)
	pgInfo3 := newQueuedPodGroupInfoForLookup(pod3)

	tests := []struct {
		name      string
		podsToAdd []*framework.QueuedPodInfo
		want      map[*framework.QueuedPodGroupInfo][]*framework.QueuedPodInfo
	}{
		{
			name:      "pod group with multiple pods",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			want: map[*framework.QueuedPodGroupInfo][]*framework.QueuedPodInfo{
				pgInfo1: {pInfo1, pInfo2},
			},
		},
		{
			name:      "two pod groups",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo3},
			want: map[*framework.QueuedPodGroupInfo][]*framework.QueuedPodInfo{
				pgInfo1: {pInfo1},
				pgInfo3: {pInfo3},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(newQueuedPodGroupInfoForLookup(pInfo.Pod), pInfo)
			}

			for pgInfo, wantPods := range tt.want {
				if !ppm.has(pgInfo) {
					t.Errorf("Expected pod group %s to be present", pgInfo.Name)
				}
				gotPods := ppm.get(pgInfo)
				if diff := cmp.Diff(wantPods, gotPods, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
					t.Errorf("Unexpected pods for pod group %s (-want,+got)\n%s", pgInfo.Name, diff)
				}
			}
		})
	}
}

func TestPendingPodGroupMemberPods_GetPod(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").PodGroupName("pg3").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}
	pgInfo1 := newQueuedPodGroupInfoForLookup(pod1)

	tests := []struct {
		name         string
		podsToAdd    []*framework.QueuedPodInfo
		pgInfoLookup *framework.QueuedPodGroupInfo
		targetPod    *v1.Pod
		want         *framework.QueuedPodInfo
	}{
		{
			name:         "find existing pod in pod group",
			podsToAdd:    []*framework.QueuedPodInfo{pInfo1, pInfo2},
			pgInfoLookup: pgInfo1,
			targetPod:    pod1,
			want:         pInfo1,
		},
		{
			name:         "pod not in pod group",
			podsToAdd:    []*framework.QueuedPodInfo{pInfo1, pInfo2},
			pgInfoLookup: pgInfo1,
			targetPod:    pod3,
			want:         nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(newQueuedPodGroupInfoForLookup(pInfo.Pod), pInfo)
			}

			got := ppm.getPod(tt.pgInfoLookup, tt.targetPod)
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
	pgInfo1 := newQueuedPodGroupInfoForLookup(pod1)

	tests := []struct {
		name         string
		podsToAdd    []*framework.QueuedPodInfo
		pgInfoLookup *framework.QueuedPodGroupInfo
		updatePod    *v1.Pod
		checkPod     *v1.Pod
		wantUpdated  bool
	}{
		{
			name:         "update existing pod",
			podsToAdd:    []*framework.QueuedPodInfo{pInfo1},
			pgInfoLookup: pgInfo1,
			updatePod:    updatedPod1,
			wantUpdated:  true,
		},
		{
			name:         "update non-existent pod",
			podsToAdd:    []*framework.QueuedPodInfo{pInfo1},
			pgInfoLookup: pgInfo1,
			updatePod:    pod2,
			wantUpdated:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(newQueuedPodGroupInfoForLookup(pInfo.Pod), pInfo.DeepCopy())
			}

			updated := ppm.update(tt.pgInfoLookup, tt.updatePod)
			if updated != tt.wantUpdated {
				t.Fatalf("Expected updated pod: %v, got: %v", tt.wantUpdated, updated)
			}

			if tt.wantUpdated {
				got := ppm.getPod(tt.pgInfoLookup, tt.updatePod)
				if diff := cmp.Diff(tt.updatePod, got.Pod); diff != "" {
					t.Errorf("Unexpected pod (-want,+got)\n%s", diff)
				}
			}
		})
	}
}

func TestPendingPodGroupMemberPods_Delete(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns3").PodGroupName("pg3").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}
	pgInfo1 := newQueuedPodGroupInfoForLookup(pod1)
	pgInfo3 := newQueuedPodGroupInfoForLookup(pod3)

	tests := []struct {
		name         string
		podsToAdd    []*framework.QueuedPodInfo
		pgInfoLookup *framework.QueuedPodGroupInfo
		podToDelete  *v1.Pod
		want         []*framework.QueuedPodInfo
	}{
		{
			name:         "delete non-existent pod from existing pod group",
			podsToAdd:    []*framework.QueuedPodInfo{pInfo1},
			pgInfoLookup: pgInfo1,
			podToDelete:  pod2,
			want:         []*framework.QueuedPodInfo{pInfo1},
		},
		{
			name:         "delete pod from non-existent pod group",
			podsToAdd:    []*framework.QueuedPodInfo{pInfo1, pInfo2},
			pgInfoLookup: pgInfo3,
			podToDelete:  pod3,
			want:         nil,
		},
		{
			name:         "delete one pod from pod group",
			podsToAdd:    []*framework.QueuedPodInfo{pInfo1, pInfo2},
			pgInfoLookup: pgInfo1,
			podToDelete:  pod1,
			want:         []*framework.QueuedPodInfo{pInfo2},
		},
		{
			name:         "delete all pods removes pod group",
			podsToAdd:    []*framework.QueuedPodInfo{pInfo1},
			pgInfoLookup: pgInfo1,
			podToDelete:  pod1,
			want:         nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(newQueuedPodGroupInfoForLookup(pInfo.Pod), pInfo.DeepCopy())
			}

			ppm.delete(tt.pgInfoLookup, tt.podToDelete)

			expectedPodGroupExists := len(tt.want) > 0
			if exists := ppm.has(tt.pgInfoLookup); exists != expectedPodGroupExists {
				t.Errorf("Expected pod group to exist: %v, got: %v", expectedPodGroupExists, exists)
			}

			gotPods := ppm.get(tt.pgInfoLookup)
			if diff := cmp.Diff(tt.want, gotPods, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected pods for pod group %s (-want,+got)\n%s", tt.pgInfoLookup.Name, diff)
			}
		})
	}
}

func TestPendingPodGroupMemberPods_Clear(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").PodGroupName("pg3").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}
	pInfo3 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod3)}
	pgInfo1 := newQueuedPodGroupInfoForLookup(pod1)
	pgInfo3 := newQueuedPodGroupInfoForLookup(pod3)

	tests := []struct {
		name        string
		podsToAdd   []*framework.QueuedPodInfo
		clearGroup  *framework.QueuedPodGroupInfo
		wantPresent *framework.QueuedPodGroupInfo
	}{
		{
			name:        "clear target pod group",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1, pInfo2, pInfo3},
			clearGroup:  pgInfo1,
			wantPresent: pgInfo3,
		},
		{
			name:        "clear non-existent group",
			podsToAdd:   []*framework.QueuedPodInfo{pInfo1, pInfo2},
			clearGroup:  pgInfo3,
			wantPresent: pgInfo1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(newQueuedPodGroupInfoForLookup(pInfo.Pod), pInfo)
			}

			ppm.clear(tt.clearGroup)

			if ppm.has(tt.clearGroup) {
				t.Errorf("Expected pod group %s to be cleared", tt.clearGroup.Name)
			}
			if !ppm.has(tt.wantPresent) {
				t.Errorf("Expected pod group %s to be still present", tt.wantPresent.Name)
			}
		})
	}
}

func TestPendingPodGroupMemberPods_Len(t *testing.T) {
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Name("pod3").Namespace("ns1").PodGroupName("pg3").Obj()
	pInfo1 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod1)}
	pInfo2 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod2)}
	pInfo3 := &framework.QueuedPodInfo{PodInfo: mustNewTestPodInfo(t, pod3)}

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
			name:      "single pod group with multiple pods",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2},
			wantLen:   2,
		},
		{
			name:      "multiple pod groups",
			podsToAdd: []*framework.QueuedPodInfo{pInfo1, pInfo2, pInfo3},
			wantLen:   3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ppm := newPendingPodGroupMemberPods()
			for _, pInfo := range tt.podsToAdd {
				ppm.add(newQueuedPodGroupInfoForLookup(pInfo.Pod), pInfo)
			}

			if got := ppm.len(); got != tt.wantLen {
				t.Errorf("Expected length %v, got %v", tt.wantLen, got)
			}
		})
	}
}
