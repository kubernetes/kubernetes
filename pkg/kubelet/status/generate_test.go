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

	"k8s.io/kubernetes/pkg/api"
)

func TestGeneratePodReadyCondition(t *testing.T) {
	tests := []struct {
		spec              *api.PodSpec
		containerStatuses []api.ContainerStatus
		podPhase          api.PodPhase
		expected          api.PodCondition
	}{
		{
			spec:              nil,
			containerStatuses: nil,
			podPhase:          api.PodRunning,
			expected:          getReadyCondition(false, "UnknownContainerStatuses", ""),
		},
		{
			spec:              &api.PodSpec{},
			containerStatuses: []api.ContainerStatus{},
			podPhase:          api.PodRunning,
			expected:          getReadyCondition(true, "", ""),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
				},
			},
			containerStatuses: []api.ContainerStatus{},
			podPhase:          api.PodRunning,
			expected:          getReadyCondition(false, "ContainersNotReady", "containers with unknown status: [1234]"),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []api.ContainerStatus{
				getReadyStatus("1234"),
				getReadyStatus("5678"),
			},
			podPhase: api.PodRunning,
			expected: getReadyCondition(true, "", ""),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []api.ContainerStatus{
				getReadyStatus("1234"),
			},
			podPhase: api.PodRunning,
			expected: getReadyCondition(false, "ContainersNotReady", "containers with unknown status: [5678]"),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []api.ContainerStatus{
				getReadyStatus("1234"),
				getNotReadyStatus("5678"),
			},
			podPhase: api.PodRunning,
			expected: getReadyCondition(false, "ContainersNotReady", "containers with unready status: [5678]"),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
				},
			},
			containerStatuses: []api.ContainerStatus{
				getNotReadyStatus("1234"),
			},
			podPhase: api.PodSucceeded,
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

func getReadyCondition(ready bool, reason, message string) api.PodCondition {
	status := api.ConditionFalse
	if ready {
		status = api.ConditionTrue
	}
	return api.PodCondition{
		Type:    api.PodReady,
		Status:  status,
		Reason:  reason,
		Message: message,
	}
}

func getReadyStatus(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name:  cName,
		Ready: true,
	}
}

func getNotReadyStatus(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name:  cName,
		Ready: false,
	}
}
