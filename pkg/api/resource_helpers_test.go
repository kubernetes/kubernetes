/*
Copyright 2015 The Kubernetes Authors.

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

package api

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestResourceHelpers(t *testing.T) {
	cpuLimit := resource.MustParse("10")
	memoryLimit := resource.MustParse("10G")
	resourceSpec := ResourceRequirements{
		Limits: ResourceList{
			"cpu":             cpuLimit,
			"memory":          memoryLimit,
			"kube.io/storage": memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Cmp(cpuLimit) != 0 {
		t.Errorf("expected cpulimit %v, got %v", cpuLimit, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
	resourceSpec = ResourceRequirements{
		Limits: ResourceList{
			"memory":          memoryLimit,
			"kube.io/storage": memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Value() != 0 {
		t.Errorf("expected cpulimit %v, got %v", 0, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
}

func TestDefaultResourceHelpers(t *testing.T) {
	resourceList := ResourceList{}
	if resourceList.Cpu().Format != resource.DecimalSI {
		t.Errorf("expected %v, actual %v", resource.DecimalSI, resourceList.Cpu().Format)
	}
	if resourceList.Memory().Format != resource.BinarySI {
		t.Errorf("expected %v, actual %v", resource.BinarySI, resourceList.Memory().Format)
	}
}

func newPod(now metav1.Time, ready bool, beforeSec int) *Pod {
	conditionStatus := ConditionFalse
	if ready {
		conditionStatus = ConditionTrue
	}
	return &Pod{
		Status: PodStatus{
			Conditions: []PodCondition{
				{
					Type:               PodReady,
					LastTransitionTime: metav1.NewTime(now.Time.Add(-1 * time.Duration(beforeSec) * time.Second)),
					Status:             conditionStatus,
				},
			},
		},
	}
}

func TestIsPodAvailable(t *testing.T) {
	now := metav1.Now()
	tests := []struct {
		pod             *Pod
		minReadySeconds int32
		expected        bool
	}{
		{
			pod:             newPod(now, false, 0),
			minReadySeconds: 0,
			expected:        false,
		},
		{
			pod:             newPod(now, true, 0),
			minReadySeconds: 1,
			expected:        false,
		},
		{
			pod:             newPod(now, true, 0),
			minReadySeconds: 0,
			expected:        true,
		},
		{
			pod:             newPod(now, true, 51),
			minReadySeconds: 50,
			expected:        true,
		},
	}

	for i, test := range tests {
		isAvailable := IsPodAvailable(test.pod, test.minReadySeconds, now)
		if isAvailable != test.expected {
			t.Errorf("[tc #%d] expected available pod: %t, got: %t", i, test.expected, isAvailable)
		}
	}
}

func TestUpdatePodCondition(t *testing.T) {
	now := metav1.Now()
	tests := []struct {
		podStatus      *PodStatus
		podCondition   *PodCondition
		expected       *PodStatus
		expectedResult bool
		test           string
	}{
		{
			podStatus: &PodStatus{
				Conditions: []PodCondition{
					{
						Type:   PodInitialized,
						Status: ConditionFalse,
					},
				},
			},
			podCondition: &PodCondition{
				Type:          PodReady,
				Status:        ConditionTrue,
				LastProbeTime: now,
			},
			expected: &PodStatus{
				Conditions: []PodCondition{
					{
						Type:   PodInitialized,
						Status: ConditionFalse,
					},
					{
						Type:          PodReady,
						Status:        ConditionTrue,
						LastProbeTime: now,
					},
				},
			},
			expectedResult: true,
			test:           "pod condition has been added",
		},
		{
			podStatus: &PodStatus{
				Conditions: []PodCondition{
					{
						Type:   PodReady,
						Status: ConditionFalse,
					},
				},
			},
			podCondition: &PodCondition{
				Type:          PodReady,
				Status:        ConditionTrue,
				LastProbeTime: now,
			},
			expected: &PodStatus{
				Conditions: []PodCondition{
					{
						Type:          PodReady,
						Status:        ConditionTrue,
						LastProbeTime: now,
					},
				},
			},
			expectedResult: true,
			test:           "pod condition has changed",
		},
		{
			podStatus: &PodStatus{
				Conditions: []PodCondition{
					{
						Type:   PodReady,
						Status: ConditionTrue,
					},
				},
			},
			podCondition: &PodCondition{
				Status: ConditionTrue,
				Type:   PodReady,
			},
			expected: &PodStatus{
				Conditions: []PodCondition{
					{
						Type:   PodReady,
						Status: ConditionTrue,
					},
				},
			},
			expectedResult: false,
			test:           "pod condition has not changed or not been added",
		},
	}

	for _, test := range tests {
		addOrChange := UpdatePodCondition(test.podStatus, test.podCondition)
		if addOrChange == false {
			if addOrChange != test.expectedResult {
				t.Errorf("%s: expected result: %t, got: %t", test.test, test.expected, test.podStatus)
			}
		} else {
			for i := range test.podStatus.Conditions {
				if test.podStatus.Conditions[i].Type == test.podCondition.Type {
					if test.podStatus.Conditions[i].LastTransitionTime.Equal(test.expected.Conditions[i].LastTransitionTime) {
						t.Errorf("%s: expected result: %t, got: %t", test.test, test.expected, test.podStatus)
					}
				}
				test.podStatus.Conditions[i].LastTransitionTime = metav1.Time{}
				test.expected.Conditions[i].LastTransitionTime = metav1.Time{}
			}
			if !reflect.DeepEqual(test.podStatus, test.expected) {
				t.Errorf("%s: expected result: %t, got: %t", test.test, test.expected, test.podStatus)
			}
		}
	}
}

func TestIsNodeReady(t *testing.T) {
	tests := []struct {
		node     *Node
		expected bool
	}{
		{
			node: &Node{
				Status: NodeStatus{
					Conditions: []NodeCondition{
						{Type: NodeReady, Status: ConditionTrue},
					},
				},
			},
			expected: true,
		},
		{
			node: &Node{
				Status: NodeStatus{
					Conditions: []NodeCondition{
						{Type: NodeReady, Status: ConditionFalse},
					},
				},
			},
			expected: false,
		},
		{
			node: &Node{
				Status: NodeStatus{
					Conditions: []NodeCondition{
						{Type: NodeOutOfDisk, Status: ConditionTrue},
						{Type: NodeMemoryPressure, Status: ConditionTrue},
					},
				},
			},
			expected: false,
		},
	}

	for i, test := range tests {
		ready := IsNodeReady(test.node)
		if ready != test.expected {
			t.Errorf("[tc #%d] expected result: %t, got: %t", i, test.expected, ready)
		}
	}

}
