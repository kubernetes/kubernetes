/*
Copyright 2014 The Kubernetes Authors.

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

package status

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
)

func TestGeneratePodReadyCondition(t *testing.T) {
	tests := []struct {
		spec              *v1.PodSpec
		containerStatuses []v1.ContainerStatus
		podPhase          v1.PodPhase
		expected          v1.PodCondition
	}{
		{
			spec:              nil,
			containerStatuses: nil,
			podPhase:          v1.PodRunning,
			expected:          getReadyCondition(false, "UnknownContainerStatuses", ""),
		},
		{
			spec:              &v1.PodSpec{},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expected:          getReadyCondition(true, "", ""),
		},
		{
			spec: &v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
				},
			},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expected:          getReadyCondition(false, "ContainersNotReady", "containers with unknown status: [1234]"),
		},
		{
			spec: &v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
				getReadyStatus("5678"),
			},
			podPhase: v1.PodRunning,
			expected: getReadyCondition(true, "", ""),
		},
		{
			spec: &v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
			},
			podPhase: v1.PodRunning,
			expected: getReadyCondition(false, "ContainersNotReady", "containers with unknown status: [5678]"),
		},
		{
			spec: &v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
				getNotReadyStatus("5678"),
			},
			podPhase: v1.PodRunning,
			expected: getReadyCondition(false, "ContainersNotReady", "containers with unready status: [5678]"),
		},
		{
			spec: &v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getNotReadyStatus("1234"),
			},
			podPhase: v1.PodSucceeded,
			expected: getReadyCondition(false, "PodCompleted", ""),
		},
	}

	for i, test := range tests {
		condition := GeneratePodReadyCondition(test.spec, test.containerStatuses, test.podPhase)
		if !reflect.DeepEqual(condition, test.expected) {
			t.Errorf("On test case %v, expected:\n%+v\ngot\n%+v\n", i, test.expected, condition)
		}
	}
}

func getReadyCondition(ready bool, reason, message string) v1.PodCondition {
	status := v1.ConditionFalse
	if ready {
		status = v1.ConditionTrue
	}
	return v1.PodCondition{
		Type:    v1.PodReady,
		Status:  status,
		Reason:  reason,
		Message: message,
	}
}

func getReadyStatus(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name:  cName,
		Ready: true,
	}
}

func getNotReadyStatus(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name:  cName,
		Ready: false,
	}
}
