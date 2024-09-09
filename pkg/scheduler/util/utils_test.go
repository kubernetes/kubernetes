/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"context"
	"errors"
	"fmt"
	"syscall"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/net"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
)

func TestGetPodFullName(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "pod",
		},
	}
	got := GetPodFullName(pod)
	expected := fmt.Sprintf("%s_%s", pod.Name, pod.Namespace)
	if got != expected {
		t.Errorf("Got wrong full name, got: %s, expected: %s", got, expected)
	}
}

func newPriorityPodWithStartTime(name string, priority int32, startTime time.Time) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Priority: &priority,
		},
		Status: v1.PodStatus{
			StartTime: &metav1.Time{Time: startTime},
		},
	}
}

func TestGetEarliestPodStartTime(t *testing.T) {
	var priority int32 = 1
	currentTime := time.Now()
	tests := []struct {
		name              string
		pods              []*v1.Pod
		expectedStartTime *metav1.Time
	}{
		{
			name:              "Pods length is 0",
			pods:              []*v1.Pod{},
			expectedStartTime: nil,
		},
		{
			name: "generate new startTime",
			pods: []*v1.Pod{
				newPriorityPodWithStartTime("pod1", 1, currentTime.Add(-time.Second)),
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod2",
					},
					Spec: v1.PodSpec{
						Priority: &priority,
					},
				},
			},
			expectedStartTime: &metav1.Time{Time: currentTime.Add(-time.Second)},
		},
		{
			name: "Pod with earliest start time last in the list",
			pods: []*v1.Pod{
				newPriorityPodWithStartTime("pod1", 1, currentTime.Add(time.Second)),
				newPriorityPodWithStartTime("pod2", 2, currentTime.Add(time.Second)),
				newPriorityPodWithStartTime("pod3", 2, currentTime),
			},
			expectedStartTime: &metav1.Time{Time: currentTime},
		},
		{
			name: "Pod with earliest start time first in the list",
			pods: []*v1.Pod{
				newPriorityPodWithStartTime("pod1", 2, currentTime),
				newPriorityPodWithStartTime("pod2", 2, currentTime.Add(time.Second)),
				newPriorityPodWithStartTime("pod3", 2, currentTime.Add(2*time.Second)),
			},
			expectedStartTime: &metav1.Time{Time: currentTime},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			startTime := GetEarliestPodStartTime(&extenderv1.Victims{Pods: test.pods})
			if !startTime.Equal(test.expectedStartTime) {
				t.Errorf("startTime is not the expected result,got %v, expected %v", startTime, test.expectedStartTime)
			}
		})
	}
}

func TestMoreImportantPod(t *testing.T) {
	currentTime := time.Now()
	pod1 := newPriorityPodWithStartTime("pod1", 1, currentTime)
	pod2 := newPriorityPodWithStartTime("pod2", 2, currentTime.Add(time.Second))
	pod3 := newPriorityPodWithStartTime("pod3", 2, currentTime)

	tests := map[string]struct {
		p1       *v1.Pod
		p2       *v1.Pod
		expected bool
	}{
		"Pod with higher priority": {
			p1:       pod1,
			p2:       pod2,
			expected: false,
		},
		"Pod with older created time": {
			p1:       pod2,
			p2:       pod3,
			expected: false,
		},
		"Pods with same start time": {
			p1:       pod3,
			p2:       pod1,
			expected: true,
		},
	}

	for k, v := range tests {
		t.Run(k, func(t *testing.T) {
			got := MoreImportantPod(v.p1, v.p2)
			if got != v.expected {
				t.Errorf("expected %t but got %t", v.expected, got)
			}
		})
	}
}

func TestRemoveNominatedNodeName(t *testing.T) {
	tests := []struct {
		name                     string
		currentNominatedNodeName string
		newNominatedNodeName     string
		expectedPatchRequests    int
		expectedPatchData        string
	}{
		{
			name:                     "Should make patch request to clear node name",
			currentNominatedNodeName: "node1",
			expectedPatchRequests:    1,
			expectedPatchData:        `{"status":{"nominatedNodeName":null}}`,
		},
		{
			name:                     "Should not make patch request if nominated node is already cleared",
			currentNominatedNodeName: "",
			expectedPatchRequests:    0,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			actualPatchRequests := 0
			var actualPatchData string
			cs := &clientsetfake.Clientset{}
			cs.AddReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				actualPatchRequests++
				patch := action.(clienttesting.PatchAction)
				actualPatchData = string(patch.GetPatch())
				// For this test, we don't care about the result of the patched pod, just that we got the expected
				// patch request, so just returning &v1.Pod{} here is OK because scheduler doesn't use the response.
				return true, &v1.Pod{}, nil
			})

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Status:     v1.PodStatus{NominatedNodeName: test.currentNominatedNodeName},
			}

			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			if err := ClearNominatedNodeName(ctx, cs, pod); err != nil {
				t.Fatalf("Error calling removeNominatedNodeName: %v", err)
			}

			if actualPatchRequests != test.expectedPatchRequests {
				t.Fatalf("Actual patch requests (%d) dos not equal expected patch requests (%d)", actualPatchRequests, test.expectedPatchRequests)
			}

			if test.expectedPatchRequests > 0 && actualPatchData != test.expectedPatchData {
				t.Fatalf("Patch data mismatch: Actual was %v, but expected %v", actualPatchData, test.expectedPatchData)
			}
		})
	}
}

func TestPatchPodStatus(t *testing.T) {
	tests := []struct {
		name   string
		pod    v1.Pod
		client *clientsetfake.Clientset
		// validateErr checks if error returned from PatchPodStatus is expected one or not.
		// (true means error is expected one.)
		validateErr    func(goterr error) bool
		statusToUpdate v1.PodStatus
	}{
		{
			name:   "Should update pod conditions successfully",
			client: clientsetfake.NewClientset(),
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "ns",
					Name:      "pod1",
				},
				Spec: v1.PodSpec{
					ImagePullSecrets: []v1.LocalObjectReference{{Name: "foo"}},
				},
			},
			statusToUpdate: v1.PodStatus{
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
		},
		{
			// ref: #101697, #94626 - ImagePullSecrets are allowed to have empty secret names
			// which would fail the 2-way merge patch generation on Pod patches
			// due to the mergeKey being the name field
			name:   "Should update pod conditions successfully on a pod Spec with secrets with empty name",
			client: clientsetfake.NewClientset(),
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "ns",
					Name:      "pod1",
				},
				Spec: v1.PodSpec{
					// this will serialize to imagePullSecrets:[{}]
					ImagePullSecrets: make([]v1.LocalObjectReference, 1),
				},
			},
			statusToUpdate: v1.PodStatus{
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
		},
		{
			name: "retry patch request when an 'connection refused' error is returned",
			client: func() *clientsetfake.Clientset {
				client := clientsetfake.NewClientset()

				reqcount := 0
				client.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
					defer func() { reqcount++ }()
					if reqcount == 0 {
						// return an connection refused error for the first patch request.
						return true, &v1.Pod{}, fmt.Errorf("connection refused: %w", syscall.ECONNREFUSED)
					}
					if reqcount == 1 {
						// not return error for the second patch request.
						return false, &v1.Pod{}, nil
					}

					// return error if requests comes in more than three times.
					return true, nil, errors.New("requests comes in more than three times.")
				})

				return client
			}(),
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "ns",
					Name:      "pod1",
				},
				Spec: v1.PodSpec{
					ImagePullSecrets: []v1.LocalObjectReference{{Name: "foo"}},
				},
			},
			statusToUpdate: v1.PodStatus{
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
		},
		{
			name: "only 4 retries at most",
			client: func() *clientsetfake.Clientset {
				client := clientsetfake.NewClientset()

				reqcount := 0
				client.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
					defer func() { reqcount++ }()
					if reqcount >= 4 {
						// return error if requests comes in more than four times.
						return true, nil, errors.New("requests comes in more than four times.")
					}

					// return an connection refused error for the first patch request.
					return true, &v1.Pod{}, fmt.Errorf("connection refused: %w", syscall.ECONNREFUSED)
				})

				return client
			}(),
			pod: v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "ns",
					Name:      "pod1",
				},
				Spec: v1.PodSpec{
					ImagePullSecrets: []v1.LocalObjectReference{{Name: "foo"}},
				},
			},
			validateErr: net.IsConnectionRefused,
			statusToUpdate: v1.PodStatus{
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := tc.client
			_, err := client.CoreV1().Pods(tc.pod.Namespace).Create(context.TODO(), &tc.pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			err = PatchPodStatus(ctx, client, &tc.pod, &tc.statusToUpdate)
			if err != nil && tc.validateErr == nil {
				// shouldn't be error
				t.Fatal(err)
			}
			if tc.validateErr != nil {
				if !tc.validateErr(err) {
					t.Fatalf("Returned unexpected error: %v", err)
				}
				return
			}

			retrievedPod, err := client.CoreV1().Pods(tc.pod.Namespace).Get(ctx, tc.pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(tc.statusToUpdate, retrievedPod.Status); diff != "" {
				t.Errorf("unexpected pod status (-want,+got):\n%s", diff)
			}
		})
	}
}

// Test_As tests the As function with Pod.
func Test_As_Pod(t *testing.T) {
	tests := []struct {
		name       string
		oldObj     interface{}
		newObj     interface{}
		wantOldObj *v1.Pod
		wantNewObj *v1.Pod
		wantErr    bool
	}{
		{
			name:       "nil old Pod",
			oldObj:     nil,
			newObj:     &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			wantOldObj: nil,
			wantNewObj: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			name:       "nil new Pod",
			oldObj:     &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			newObj:     nil,
			wantOldObj: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			wantNewObj: nil,
		},
		{
			name:    "two different kinds of objects",
			oldObj:  &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			newObj:  &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotOld, gotNew, err := As[*v1.Pod](tc.oldObj, tc.newObj)
			if err != nil && !tc.wantErr {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, but got nil")
				}
				return
			}

			if diff := cmp.Diff(tc.wantOldObj, gotOld); diff != "" {
				t.Errorf("unexpected old object (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantNewObj, gotNew); diff != "" {
				t.Errorf("unexpected new object (-want,+got):\n%s", diff)
			}
		})
	}
}

// Test_As_Node tests the As function with Node.
func Test_As_Node(t *testing.T) {
	tests := []struct {
		name       string
		oldObj     interface{}
		newObj     interface{}
		wantOldObj *v1.Node
		wantNewObj *v1.Node
		wantErr    bool
	}{
		{
			name:       "nil old Node",
			oldObj:     nil,
			newObj:     &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			wantOldObj: nil,
			wantNewObj: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
		},
		{
			name:       "nil new Node",
			oldObj:     &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			newObj:     nil,
			wantOldObj: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			wantNewObj: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotOld, gotNew, err := As[*v1.Node](tc.oldObj, tc.newObj)
			if err != nil && !tc.wantErr {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, but got nil")
				}
				return
			}

			if diff := cmp.Diff(tc.wantOldObj, gotOld); diff != "" {
				t.Errorf("unexpected old object (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantNewObj, gotNew); diff != "" {
				t.Errorf("unexpected new object (-want,+got):\n%s", diff)
			}
		})
	}
}

// Test_As_KMetadata tests the As function with Pod.
func Test_As_KMetadata(t *testing.T) {
	tests := []struct {
		name    string
		oldObj  interface{}
		newObj  interface{}
		wantErr bool
	}{
		{
			name:    "nil old Pod",
			oldObj:  nil,
			newObj:  &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			wantErr: false,
		},
		{
			name:    "nil new Pod",
			oldObj:  &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			newObj:  nil,
			wantErr: false,
		},
		{
			name:    "two different kinds of objects",
			oldObj:  &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			newObj:  &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			wantErr: false,
		},
		{
			name:    "unknown old type",
			oldObj:  "unknown type",
			wantErr: true,
		},
		{
			name:    "unknown new type",
			newObj:  "unknown type",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := As[klog.KMetadata](tc.oldObj, tc.newObj)
			if err != nil && !tc.wantErr {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, but got nil")
				}
				return
			}
		})
	}
}
