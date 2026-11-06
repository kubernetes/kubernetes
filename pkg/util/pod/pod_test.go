/*
Copyright 2018 The Kubernetes Authors.

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

package pod

import (
	"context"
	"testing"

	"reflect"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
)

func TestPatchPodStatus(t *testing.T) {
	ns := "ns"
	name := "name"
	uid := types.UID("myuid")
	client := &fake.Clientset{}
	client.CoreV1().Pods(ns).Create(context.TODO(), &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
	}, metav1.CreateOptions{})

	testCases := []struct {
		description        string
		mutate             func(input v1.PodStatus) v1.PodStatus
		expectUnchanged    bool
		expectedPatchBytes []byte
	}{
		{
			"no change",
			func(input v1.PodStatus) v1.PodStatus { return input },
			true,
			[]byte(`{"metadata":{"uid":"myuid"}}`),
		},
		{
			"message change",
			func(input v1.PodStatus) v1.PodStatus {
				input.Message = "random message"
				return input
			},
			false,
			[]byte(`{"metadata":{"uid":"myuid"},"status":{"message":"random message"}}`),
		},
		{
			"pod condition change",
			func(input v1.PodStatus) v1.PodStatus {
				input.Conditions[0].Status = v1.ConditionFalse
				return input
			},
			false,
			[]byte(`{"metadata":{"uid":"myuid"},"status":{"$setElementOrder/conditions":[{"type":"Ready"},{"type":"PodScheduled"}],"conditions":[{"status":"False","type":"Ready"}]}}`),
		},
		{
			"additional init container condition",
			func(input v1.PodStatus) v1.PodStatus {
				input.InitContainerStatuses = []v1.ContainerStatus{
					{
						Name:  "init-container",
						Ready: true,
					},
				}
				return input
			},
			false,
			[]byte(`{"metadata":{"uid":"myuid"},"status":{"initContainerStatuses":[{"image":"","imageID":"","lastState":{},"name":"init-container","ready":true,"restartCount":0,"state":{}}]}}`),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			_, patchBytes, unchanged, err := PatchPodStatus(context.TODO(), client, ns, name, uid, getPodStatus(), tc.mutate(getPodStatus()))
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if unchanged != tc.expectUnchanged {
				t.Errorf("unexpected change: %t", unchanged)
			}
			if !reflect.DeepEqual(patchBytes, tc.expectedPatchBytes) {
				t.Errorf("expect patchBytes: %q, got: %q\n", tc.expectedPatchBytes, patchBytes)
			}
		})
	}
}

// TestPatchPodStatusVolumeHealth applies the generated patch, because a patch
// that looks reasonable can still merge conditions sharing a status.
func TestPatchPodStatusVolumeHealth(t *testing.T) {
	ns := "ns"
	name := "name"
	uid := types.UID("myuid")
	slowIO := v1.VolumeHealthCondition{Status: v1.VolumeHealthDegraded, Reason: "SlowIO", Message: "io latency high"}
	pathDegraded := v1.VolumeHealthCondition{Status: v1.VolumeHealthDegraded, Reason: "PathDegraded", Message: "one path down"}
	statusWith := func(conditions ...v1.VolumeHealthCondition) v1.PodStatus {
		status := getPodStatus()
		status.VolumeStatuses = []v1.PodVolumeStatus{{
			Name:         "vol1",
			HealthStatus: &v1.VolumeHealthStatus{HealthConditions: conditions},
		}}
		return status
	}

	testCases := []struct {
		description string
		oldStatus   v1.PodStatus
		newStatus   v1.PodStatus
	}{
		{
			description: "add condition with same status and different reason",
			oldStatus:   statusWith(slowIO),
			newStatus:   statusWith(slowIO, pathDegraded),
		},
		{
			description: "remove one of two conditions with same status",
			oldStatus:   statusWith(slowIO, pathDegraded),
			newStatus:   statusWith(pathDegraded),
		},
		{
			description: "replace reason for same status",
			oldStatus:   statusWith(slowIO),
			newStatus:   statusWith(pathDegraded),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			client := fake.NewClientset(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Namespace: ns, Name: name, UID: uid},
				Status:     tc.oldStatus,
			})
			newPod, _, _, err := PatchPodStatus(context.TODO(), client, ns, name, uid, tc.oldStatus, tc.newStatus)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.newStatus.VolumeStatuses, newPod.Status.VolumeStatuses); diff != "" {
				t.Errorf("unexpected volume statuses after patch (-want +got):\n%s", diff)
			}
		})
	}
}

func getPodStatus() v1.PodStatus {
	return v1.PodStatus{
		Phase: v1.PodRunning,
		Conditions: []v1.PodCondition{
			{
				Type:   v1.PodReady,
				Status: v1.ConditionTrue,
			},
			{
				Type:   v1.PodScheduled,
				Status: v1.ConditionTrue,
			},
		},
		ContainerStatuses: []v1.ContainerStatus{
			{
				Name:  "container1",
				Ready: true,
			},
			{
				Name:  "container2",
				Ready: true,
			},
		},
		Message: "Message",
	}
}

func TestReplaceOrAppendPodCondition(t *testing.T) {
	cType := v1.PodConditionType("ExampleType")
	testCases := []struct {
		description    string
		conditions     []v1.PodCondition
		condition      v1.PodCondition
		wantConditions []v1.PodCondition
	}{
		{
			description: "append",
			conditions:  []v1.PodCondition{},
			condition: v1.PodCondition{
				Type:   cType,
				Status: v1.ConditionTrue,
			},
			wantConditions: []v1.PodCondition{
				{
					Type:   cType,
					Status: v1.ConditionTrue,
				},
			},
		},
		{
			description: "replace",
			conditions: []v1.PodCondition{
				{
					Type:   cType,
					Status: v1.ConditionTrue,
				},
			},
			condition: v1.PodCondition{
				Type:   cType,
				Status: v1.ConditionFalse,
			},
			wantConditions: []v1.PodCondition{
				{
					Type:   cType,
					Status: v1.ConditionFalse,
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			gotConditions := ReplaceOrAppendPodCondition(tc.conditions, &tc.condition)
			if diff := cmp.Diff(tc.wantConditions, gotConditions); diff != "" {
				t.Errorf("Unexpected conditions: %s", diff)
			}
		})
	}
}
