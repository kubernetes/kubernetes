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

package container

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestParseContainerID(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected ContainerID
	}{
		{
			name:     "valid docker container id",
			input:    `"docker://abc123"`,
			expected: ContainerID{Type: "docker", ID: "abc123"},
		},
		{
			name:     "valid containerd container id",
			input:    `"containerd://def456"`,
			expected: ContainerID{Type: "containerd", ID: "def456"},
		},
		{
			name:     "valid format - no quotes",
			input:    "docker://abc123",
			expected: ContainerID{Type: "docker", ID: "abc123"},
		},
		{
			name:     "invalid format - missing separator",
			input:    `"dockerabc123"`,
			expected: ContainerID{},
		},
		{
			name:     "empty string",
			input:    `""`,
			expected: ContainerID{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ParseContainerID(tt.input)
			assert.Equal(t, tt.expected, result, "ParseContainerID(%q)", tt.input)
		})
	}
}

func TestContainerIDString(t *testing.T) {
	tests := []struct {
		name     string
		cid      ContainerID
		expected string
	}{
		{
			name:     "docker container",
			cid:      ContainerID{Type: "docker", ID: "abc123"},
			expected: "docker://abc123",
		},
		{
			name:     "containerd container",
			cid:      ContainerID{Type: "containerd", ID: "def456"},
			expected: "containerd://def456",
		},
		{
			name:     "empty container id",
			cid:      ContainerID{},
			expected: "://",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.cid.String()
			assert.Equal(t, tt.expected, result, "ContainerID.String()")
		})
	}
}

func TestPodStatusFindContainerStatusByName(t *testing.T) {
	podStatus := &PodStatus{
		ContainerStatuses: []*Status{
			{Name: "container1", State: ContainerStateRunning},
			{Name: "container2", State: ContainerStateExited},
			{Name: "container1", State: ContainerStateCreated}, // duplicate name
		},
	}

	tests := []struct {
		name           string
		containerName  string
		expectedStatus *Status
	}{
		{
			name:           "find existing container",
			containerName:  "container1",
			expectedStatus: podStatus.ContainerStatuses[0], // should return first match
		},
		{
			name:           "find another existing container",
			containerName:  "container2",
			expectedStatus: podStatus.ContainerStatuses[1],
		},
		{
			name:           "find non-existing container",
			containerName:  "nonexistent",
			expectedStatus: nil,
		},
		{
			name:           "empty container name",
			containerName:  "",
			expectedStatus: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := podStatus.FindContainerStatusByName(tt.containerName)
			assert.Equal(t, tt.expectedStatus, result, "FindContainerStatusByName(%q)", tt.containerName)
		})
	}
}

func TestPodStatusGetRunningContainerStatuses(t *testing.T) {
	podStatus := &PodStatus{
		ContainerStatuses: []*Status{
			{Name: "container1", State: ContainerStateRunning},
			{Name: "container2", State: ContainerStateExited},
			{Name: "container3", State: ContainerStateRunning},
			{Name: "container4", State: ContainerStateCreated},
		},
	}

	expected := []*Status{
		podStatus.ContainerStatuses[0], // container1 - running
		podStatus.ContainerStatuses[2], // container3 - running
	}

	result := podStatus.GetRunningContainerStatuses()
	assert.Equal(t, expected, result, "GetRunningContainerStatuses()")
}

func TestGetPodFullName(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.Pod
		expected string
	}{
		{
			name: "normal pod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
				},
			},
			expected: "test-pod_test-namespace",
		},
		{
			name: "pod with empty name and namespace",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "",
					Namespace: "",
				},
			},
			expected: "_",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetPodFullName(tt.pod)
			assert.Equal(t, tt.expected, result, "GetPodFullName()")
		})
	}
}

func TestBuildPodFullName(t *testing.T) {
	tests := []struct {
		name      string
		podName   string
		namespace string
		expected  string
	}{
		{
			name:      "normal pod",
			podName:   "test-pod",
			namespace: "test-namespace",
			expected:  "test-pod_test-namespace",
		},
		{
			name:      "empty name and namespace",
			podName:   "",
			namespace: "",
			expected:  "_",
		},
		{
			name:      "empty name only",
			podName:   "",
			namespace: "test-namespace",
			expected:  "_test-namespace",
		},
		{
			name:      "empty namespace only",
			podName:   "test-pod",
			namespace: "",
			expected:  "test-pod_",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := BuildPodFullName(tt.podName, tt.namespace)
			assert.Equal(t, tt.expected, result, "BuildPodFullName(%q, %q)", tt.podName, tt.namespace)
		})
	}
}

func TestParsePodFullName(t *testing.T) {
	tests := []struct {
		name              string
		podFullName       string
		expectedName      string
		expectedNamespace string
		expectError       bool
	}{
		{
			name:              "valid pod full name",
			podFullName:       "test-pod_test-namespace",
			expectedName:      "test-pod",
			expectedNamespace: "test-namespace",
			expectError:       false,
		},
		{
			name:              "invalid format - no underscore",
			podFullName:       "test-pod",
			expectedName:      "",
			expectedNamespace: "",
			expectError:       true,
		},
		{
			name:              "invalid format - multiple underscores",
			podFullName:       "test_pod_namespace",
			expectedName:      "",
			expectedNamespace: "",
			expectError:       true,
		},
		{
			name:              "invalid format - empty parts",
			podFullName:       "_",
			expectedName:      "",
			expectedNamespace: "",
			expectError:       true,
		},
		{
			name:              "invalid format - empty name",
			podFullName:       "_namespace",
			expectedName:      "",
			expectedNamespace: "",
			expectError:       true,
		},
		{
			name:              "invalid format - empty namespace",
			podFullName:       "pod_",
			expectedName:      "",
			expectedNamespace: "",
			expectError:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name, namespace, err := ParsePodFullName(tt.podFullName)

			if tt.expectError {
				require.Error(t, err, "ParsePodFullName(%q) expected error", tt.podFullName)
			} else {
				require.NoError(t, err, "ParsePodFullName(%q) unexpected error", tt.podFullName)
			}
			assert.Equal(t, tt.expectedName, name, "ParsePodFullName(%q) name", tt.podFullName)
			assert.Equal(t, tt.expectedNamespace, namespace, "ParsePodFullName(%q) namespace", tt.podFullName)
		})
	}
}

func TestPodFindContainerByName(t *testing.T) {
	pod := &Pod{
		Containers: []*Container{
			{Name: "container1", ID: ContainerID{Type: "docker", ID: "abc123"}},
			{Name: "container2", ID: ContainerID{Type: "docker", ID: "def456"}},
			{Name: "container1", ID: ContainerID{Type: "docker", ID: "ghi789"}}, // duplicate name
		},
	}

	tests := []struct {
		name              string
		containerName     string
		expectedContainer *Container
	}{
		{
			name:              "find existing container",
			containerName:     "container1",
			expectedContainer: pod.Containers[0], // should return first match
		},
		{
			name:              "find another existing container",
			containerName:     "container2",
			expectedContainer: pod.Containers[1],
		},
		{
			name:              "find non-existing container",
			containerName:     "nonexistent",
			expectedContainer: nil,
		},
		{
			name:              "empty container name",
			containerName:     "",
			expectedContainer: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := pod.FindContainerByName(tt.containerName)
			assert.Equal(t, tt.expectedContainer, result, "FindContainerByName(%q)", tt.containerName)
		})
	}
}

func TestPodFindContainerByID(t *testing.T) {
	pod := &Pod{
		Containers: []*Container{
			{Name: "container1", ID: ContainerID{Type: "docker", ID: "abc123"}},
			{Name: "container2", ID: ContainerID{Type: "containerd", ID: "def456"}},
		},
	}

	tests := []struct {
		name              string
		containerID       ContainerID
		expectedContainer *Container
	}{
		{
			name:              "find existing container",
			containerID:       ContainerID{Type: "docker", ID: "abc123"},
			expectedContainer: pod.Containers[0],
		},
		{
			name:              "find another existing container",
			containerID:       ContainerID{Type: "containerd", ID: "def456"},
			expectedContainer: pod.Containers[1],
		},
		{
			name:              "find non-existing container",
			containerID:       ContainerID{Type: "docker", ID: "nonexistent"},
			expectedContainer: nil,
		},
		{
			name:              "empty container id",
			containerID:       ContainerID{},
			expectedContainer: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := pod.FindContainerByID(tt.containerID)
			assert.Equal(t, tt.expectedContainer, result, "FindContainerByID(%v)", tt.containerID)
		})
	}
}

func TestPodFindSandboxByID(t *testing.T) {
	pod := &Pod{
		Sandboxes: []*Container{
			{Name: "sandbox1", ID: ContainerID{Type: "docker", ID: "abc123"}},
			{Name: "sandbox2", ID: ContainerID{Type: "containerd", ID: "def456"}},
		},
	}

	tests := []struct {
		name            string
		sandboxID       ContainerID
		expectedSandbox *Container
	}{
		{
			name:            "find existing sandbox",
			sandboxID:       ContainerID{Type: "docker", ID: "abc123"},
			expectedSandbox: pod.Sandboxes[0],
		},
		{
			name:            "find another existing sandbox",
			sandboxID:       ContainerID{Type: "containerd", ID: "def456"},
			expectedSandbox: pod.Sandboxes[1],
		},
		{
			name:            "find non-existing sandbox",
			sandboxID:       ContainerID{Type: "docker", ID: "nonexistent"},
			expectedSandbox: nil,
		},
		{
			name:            "empty sandbox id",
			sandboxID:       ContainerID{},
			expectedSandbox: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := pod.FindSandboxByID(tt.sandboxID)
			assert.Equal(t, tt.expectedSandbox, result, "FindSandboxByID(%v)", tt.sandboxID)
		})
	}
}

func TestPodToAPIPod(t *testing.T) {
	pod := &Pod{
		ID:        "test-uid",
		Name:      "test-pod",
		Namespace: "test-namespace",
		Containers: []*Container{
			{Name: "container1", Image: "nginx:latest"},
			{Name: "container2", Image: "redis:latest"},
		},
	}

	expected := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "test-uid",
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "container1", Image: "nginx:latest"},
				{Name: "container2", Image: "redis:latest"},
			},
		},
	}

	result := pod.ToAPIPod()
	assert.Equal(t, expected, result, "ToAPIPod()")
}

func TestPodIsEmpty(t *testing.T) {
	tests := []struct {
		name     string
		pod      *Pod
		expected bool
	}{
		{
			name:     "empty pod",
			pod:      &Pod{},
			expected: true,
		},
		{
			name: "non-empty pod",
			pod: &Pod{
				ID:        "test-uid",
				Name:      "test-pod",
				Namespace: "test-namespace",
			},
			expected: false,
		},
		{
			name: "pod with containers",
			pod: &Pod{
				Containers: []*Container{{Name: "container1"}},
			},
			expected: false,
		},
		{
			name: "pod with sandboxes",
			pod: &Pod{
				Sandboxes: []*Container{{Name: "sandbox1"}},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.pod.IsEmpty()
			assert.Equal(t, tt.expected, result, "IsEmpty()")
		})
	}
}

func TestPodsFindPod(t *testing.T) {
	pods := Pods{
		{ID: "uid1", Name: "pod1", Namespace: "ns1"},
		{ID: "uid2", Name: "pod2", Namespace: "ns2"},
	}

	tests := []struct {
		name        string
		podFullName string
		podUID      types.UID
		expected    Pod
	}{
		{
			name:        "find by full name",
			podFullName: "pod1_ns1",
			podUID:      "",
			expected:    *pods[0],
		},
		{
			name:        "find by uid when full name is empty",
			podFullName: "",
			podUID:      "uid2",
			expected:    *pods[1],
		},
		{
			name:        "find by full name takes precedence",
			podFullName: "pod1_ns1",
			podUID:      "uid2",
			expected:    *pods[0],
		},
		{
			name:        "find non-existing pod",
			podFullName: "pod3_ns3",
			podUID:      "uid3",
			expected:    Pod{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := pods.FindPod(tt.podFullName, tt.podUID)
			assert.Equal(t, tt.expected, result, "FindPod(%q, %q)", tt.podFullName, tt.podUID)
		})
	}
}

func TestRuntimeStatusGetRuntimeCondition(t *testing.T) {
	status := &RuntimeStatus{
		Conditions: []RuntimeCondition{
			{Type: RuntimeReady, Status: true, Reason: "ready", Message: "runtime is ready"},
			{Type: NetworkReady, Status: false, Reason: "not ready", Message: "network is not ready"},
		},
	}

	tests := []struct {
		name          string
		conditionType RuntimeConditionType
		expected      *RuntimeCondition
	}{
		{
			name:          "find existing condition",
			conditionType: RuntimeReady,
			expected:      &status.Conditions[0],
		},
		{
			name:          "find another existing condition",
			conditionType: NetworkReady,
			expected:      &status.Conditions[1],
		},
		{
			name:          "find non-existing condition",
			conditionType: "NonExistent",
			expected:      nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := status.GetRuntimeCondition(tt.conditionType)
			assert.Equal(t, tt.expected, result, "GetRuntimeCondition(%q)", tt.conditionType)
		})
	}
}

func TestRuntimeStatusString(t *testing.T) {
	status := &RuntimeStatus{
		Conditions: []RuntimeCondition{
			{Type: RuntimeReady, Status: true, Reason: "ready", Message: "runtime is ready"},
			{Type: NetworkReady, Status: false, Reason: "not ready", Message: "network is not ready"},
		},
		Handlers: []RuntimeHandler{
			{Name: "handler1", SupportsRecursiveReadOnlyMounts: true, SupportsUserNamespaces: false},
			{Name: "handler2", SupportsRecursiveReadOnlyMounts: false, SupportsUserNamespaces: true},
		},
		Features: &RuntimeFeatures{SupplementalGroupsPolicy: true},
	}

	result := status.String()
	expected := "Runtime Conditions: RuntimeReady=true reason:ready message:runtime is ready, NetworkReady=false reason:not ready message:network is not ready; Handlers: Name=handler1 SupportsRecursiveReadOnlyMounts: true SupportsUserNamespaces: false, Name=handler2 SupportsRecursiveReadOnlyMounts: false SupportsUserNamespaces: true, Features: SupplementalGroupsPolicy: true"
	assert.Equal(t, expected, result, "String()")
}

func TestRuntimeHandlerString(t *testing.T) {
	tests := []struct {
		name     string
		handler  RuntimeHandler
		expected string
	}{
		{
			name: "handler with all features",
			handler: RuntimeHandler{
				Name:                            "test-handler",
				SupportsRecursiveReadOnlyMounts: true,
				SupportsUserNamespaces:          true,
			},
			expected: "Name=test-handler SupportsRecursiveReadOnlyMounts: true SupportsUserNamespaces: true",
		},
		{
			name: "handler with no features",
			handler: RuntimeHandler{
				Name:                            "test-handler",
				SupportsRecursiveReadOnlyMounts: false,
				SupportsUserNamespaces:          false,
			},
			expected: "Name=test-handler SupportsRecursiveReadOnlyMounts: false SupportsUserNamespaces: false",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.handler.String()
			assert.Equal(t, tt.expected, result, "String()")
		})
	}
}

func TestRuntimeConditionString(t *testing.T) {
	tests := []struct {
		name      string
		condition RuntimeCondition
		expected  string
	}{
		{
			name: "true condition",
			condition: RuntimeCondition{
				Type:    RuntimeReady,
				Status:  true,
				Reason:  "ready",
				Message: "runtime is ready",
			},
			expected: "RuntimeReady=true reason:ready message:runtime is ready",
		},
		{
			name: "false condition",
			condition: RuntimeCondition{
				Type:    NetworkReady,
				Status:  false,
				Reason:  "not ready",
				Message: "network is not ready",
			},
			expected: "NetworkReady=false reason:not ready message:network is not ready",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.condition.String()
			assert.Equal(t, tt.expected, result, "String()")
		})
	}
}

func TestRuntimeFeaturesString(t *testing.T) {
	tests := []struct {
		name     string
		features *RuntimeFeatures
		expected string
	}{
		{
			name: "features with SupplementalGroupsPolicy true",
			features: &RuntimeFeatures{
				SupplementalGroupsPolicy: true,
			},
			expected: "SupplementalGroupsPolicy: true",
		},
		{
			name: "features with SupplementalGroupsPolicy false",
			features: &RuntimeFeatures{
				SupplementalGroupsPolicy: false,
			},
			expected: "SupplementalGroupsPolicy: false",
		},
		{
			name:     "nil features",
			features: nil,
			expected: "nil",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.features.String()
			assert.Equal(t, tt.expected, result, "String()")
		})
	}
}

func TestSortContainerStatusesByCreationTime(t *testing.T) {
	now := time.Now()
	statuses := SortContainerStatusesByCreationTime{
		{CreatedAt: now.Add(2 * time.Hour)},
		{CreatedAt: now},
		{CreatedAt: now.Add(1 * time.Hour)},
	}

	// Test Len
	assert.Len(t, statuses, 3, "Len()")

	// Test Swap
	original1 := statuses[1]
	original2 := statuses[2]
	statuses.Swap(1, 2)
	assert.Equal(t, original1, statuses[2], "Swap(0, 1) did not work correctly (0)")
	assert.Equal(t, original2, statuses[1], "Swap(0, 1) did not work correctly (1)")

	// Test Less
	assert.True(t, statuses.Less(1, 0), "Less(1, 0) should be true")
	assert.False(t, statuses.Less(0, 1), "Less(0, 1) should be false")
}
