/*
Copyright 2021 The Kubernetes Authors.

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

package container

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseString(t *testing.T) {
	dataWithQuotes := "\"docker://fc1c84c5fdc0\""
	dataWithoutQuotes := "docker://fc1c84c5fdc0"
	dataError := "fc1c84c5fdc0"
	testCases := []struct {
		data        string
		containerID *ContainerID
		expected    *ContainerID
	}{
		{
			data:        dataWithQuotes,
			containerID: &ContainerID{},
			expected: &ContainerID{
				Type: "docker",
				ID:   "fc1c84c5fdc0",
			},
		},
		{
			data:        dataWithoutQuotes,
			containerID: &ContainerID{},
			expected: &ContainerID{
				Type: "docker",
				ID:   "fc1c84c5fdc0",
			},
		},
		{
			data:        dataError,
			containerID: &ContainerID{},
		},
	}

	var err error
	for _, test := range testCases {
		err = test.containerID.ParseString(test.data)
		if err == nil {
			assert.Equal(t, test.containerID.Type, test.expected.Type)
			assert.Equal(t, test.containerID.ID, test.expected.ID)
		} else {
			expectErr := fmt.Errorf("invalid container ID: %q", test.data)
			assert.EqualError(t, err, expectErr.Error())
		}
	}

}

func TestFindContainerStatusByName(t *testing.T) {
	status := &PodStatus{
		ID: "test-pod",
		ContainerStatuses: []*Status{
			{
				ID: ContainerID{
					Type: "fooRuntime",
					ID:   "foo",
				},
				Name: "foo",
			},
		},
	}

	expect := &Status{
		ID: ContainerID{
			Type: "fooRuntime",
			ID:   "foo",
		},
		Name: "foo",
	}

	actualFoo := status.FindContainerStatusByName("foo")
	assert.Equal(t, actualFoo, expect)
	actualGoo := status.FindContainerStatusByName("goo")
	assert.Nil(t, actualGoo)

}

func TestGetRunningContainerStatuses(t *testing.T) {
	status := &PodStatus{
		ID: "test-pod",
		ContainerStatuses: []*Status{
			{
				ID: ContainerID{
					Type: "fooRuntime",
					ID:   "foo",
				},
				Name:  "foo",
				State: ContainerStateRunning,
			},
			{
				ID: ContainerID{
					Type: "gooRuntime",
					ID:   "goo",
				},
				Name:  "goo",
				State: ContainerStateRunning,
			},
			{
				ID: ContainerID{
					Type: "hooRuntime",
					ID:   "hoo",
				},
				Name:  "hoo",
				State: ContainerStateCreated,
			},
		},
	}

	expect := []*Status{
		{
			ID: ContainerID{
				Type: "fooRuntime",
				ID:   "foo",
			},
			Name:  "foo",
			State: ContainerStateRunning,
		},
		{
			ID: ContainerID{
				Type: "gooRuntime",
				ID:   "goo",
			},
			Name:  "goo",
			State: ContainerStateRunning,
		},
	}

	actualFoo := status.GetRunningContainerStatuses()
	assert.Equal(t, actualFoo, expect)

}

func TestGetRuntimeCondition(t *testing.T) {
	runtimeReadyCondition := &RuntimeCondition{
		Type:   RuntimeReady,
		Status: false,
	}
	networkReadyCondition := &RuntimeCondition{
		Type:   NetworkReady,
		Status: true,
	}
	status := &RuntimeStatus{
		Conditions: []RuntimeCondition{
			{
				Type:   RuntimeReady,
				Status: false,
			},
			{
				Type:   NetworkReady,
				Status: true,
			},
		},
	}

	actualForRuntimeReady := status.GetRuntimeCondition(RuntimeReady)
	assert.Equal(t, runtimeReadyCondition, actualForRuntimeReady)
	actualForNetworkReady := status.GetRuntimeCondition(NetworkReady)
	assert.Equal(t, networkReadyCondition, actualForNetworkReady)
}

func TestFindPodByID(t *testing.T) {
	expect := Pod{
		ID:   "id-pod2",
		Name: "pod2",
	}

	expectNoResult := Pod{}

	pods := Pods{
		{
			ID:   "id-pod1",
			Name: "pod1",
		},
		{
			ID:   "id-pod2",
			Name: "pod2",
		},
	}

	actual := pods.FindPodByID(expect.ID)
	assert.Equal(t, expect, actual)

	actualNoResult := pods.FindPodByID("id-pod3")
	assert.Equal(t, actualNoResult, expectNoResult)

}

func TestFindPodByFullName(t *testing.T) {
	expect := Pod{
		Namespace: "ns2",
		Name:      "pod2",
	}

	expectNoResult := Pod{}

	pods := Pods{
		{
			Namespace: "ns1",
			Name:      "pod1",
		},
		{
			Namespace: "ns2",
			Name:      "pod2",
		},
	}

	actual := pods.FindPodByFullName("pod2_ns2")
	assert.Equal(t, expect, actual)

	actualNoResult := pods.FindPodByFullName("pod3_ns3")
	assert.Equal(t, actualNoResult, expectNoResult)
}

func TestFindContainerByName(t *testing.T) {
	expect := &Container{
		Name: "container1",
	}

	pod := &Pod{
		Name: "pod1",
		Containers: []*Container{
			{
				Name: "container1",
			},
			{
				Name: "container2",
			},
		},
	}

	actual := pod.FindContainerByName("container1")
	assert.Equal(t, expect, actual)

	actualNoResult := pod.FindContainerByName("container3")
	assert.Nil(t, actualNoResult)
}

func TestFindContainerByID(t *testing.T) {
	expect := &Container{
		Name: "container1",
		ID: ContainerID{
			Type: "docker",
			ID:   "1111",
		},
	}

	pod := &Pod{
		Name: "pod1",
		Containers: []*Container{
			{
				Name: "container1",
				ID: ContainerID{
					Type: "docker",
					ID:   "1111",
				},
			},
			{
				Name: "container2",
				ID: ContainerID{
					Type: "docker",
					ID:   "2222",
				},
			},
		},
	}

	actual := pod.FindContainerByID(ContainerID{
		Type: "docker",
		ID:   "1111",
	})
	assert.Equal(t, expect, actual)

	actualNoResult := pod.FindContainerByID(ContainerID{
		Type: "docker",
		ID:   "3333",
	})
	assert.Nil(t, actualNoResult)
}

func TestFindSandboxByID(t *testing.T) {
	expect := &Container{
		Name: "container1",
		ID: ContainerID{
			Type: "docker",
			ID:   "1111",
		},
	}

	pod := &Pod{
		Name: "pod1",
		Sandboxes: []*Container{
			{
				Name: "container1",
				ID: ContainerID{
					Type: "docker",
					ID:   "1111",
				},
			},
			{
				Name: "container2",
				ID: ContainerID{
					Type: "docker",
					ID:   "2222",
				},
			},
		},
	}

	actual := pod.FindSandboxByID(ContainerID{
		Type: "docker",
		ID:   "1111",
	})
	assert.Equal(t, expect, actual)

	actualNoResult := pod.FindSandboxByID(ContainerID{
		Type: "docker",
		ID:   "3333",
	})
	assert.Nil(t, actualNoResult)
}

func TestParsePodFullName(t *testing.T) {
	fullNameWithoutLine := "name3ns3"
	testCases := []struct {
		expectName      string
		expectNamespace string
		fullName        string
		err             error
	}{
		{
			expectName:      "name1",
			expectNamespace: "ns1",
			fullName:        "name1_ns1",
		},
		{
			expectName:      "name2",
			expectNamespace: "ns2",
			fullName:        "name2_ns2",
		},
		{
			err:      fmt.Errorf("failed to parse the pod full name %q", fullNameWithoutLine),
			fullName: fullNameWithoutLine,
		},
	}

	for _, test := range testCases {
		actualName, actualNamespace, err := ParsePodFullName(test.fullName)
		if err == nil {
			assert.Equal(t, test.expectName, actualName)
			assert.Equal(t, test.expectNamespace, actualNamespace)
		} else {
			assert.EqualError(t, err, test.err.Error())
		}

	}
}
