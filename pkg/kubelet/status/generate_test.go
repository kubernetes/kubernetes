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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/utils/ptr"
)

var (
	containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways
)

func TestGenerateContainersReadyCondition(t *testing.T) {
	tests := []struct {
		spec              v1.PodSpec
		containerStatuses []v1.ContainerStatus
		podPhase          v1.PodPhase
		expectReady       v1.PodCondition
	}{
		{
			spec:              v1.PodSpec{},
			containerStatuses: nil,
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.ContainersReady, v1.ConditionFalse, UnknownContainerStatuses, ""),
		},
		{
			spec:              v1.PodSpec{},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.ContainersReady, v1.ConditionTrue, "", ""),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
				},
			},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.ContainersReady, v1.ConditionFalse, ContainersNotReady, "containers with unknown status: [1234]"),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
				getReadyStatus("5678"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.ContainersReady, v1.ConditionTrue, "", ""),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.ContainersReady, v1.ConditionFalse, ContainersNotReady, "containers with unknown status: [5678]"),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
				getNotReadyStatus("5678"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.ContainersReady, v1.ConditionFalse, ContainersNotReady, "containers with unready status: [5678]"),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getNotReadyStatus("1234"),
			},
			podPhase:    v1.PodSucceeded,
			expectReady: getPodCondition(v1.ContainersReady, v1.ConditionFalse, PodCompleted, ""),
		},
		{
			spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{Name: "restartable-init-1", RestartPolicy: &containerRestartPolicyAlways},
				},
				Containers: []v1.Container{
					{Name: "regular-1"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("regular-1"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.ContainersReady, v1.ConditionFalse, ContainersNotReady, "containers with unknown status: [restartable-init-1]"),
		},
		{
			spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{Name: "restartable-init-1", RestartPolicy: &containerRestartPolicyAlways},
					{Name: "restartable-init-2", RestartPolicy: &containerRestartPolicyAlways},
				},
				Containers: []v1.Container{
					{Name: "regular-1"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("restartable-init-1"),
				getReadyStatus("restartable-init-2"),
				getReadyStatus("regular-1"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.ContainersReady, v1.ConditionTrue, "", ""),
		},
		{
			spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{Name: "restartable-init-1", RestartPolicy: &containerRestartPolicyAlways},
					{Name: "restartable-init-2", RestartPolicy: &containerRestartPolicyAlways},
				},
				Containers: []v1.Container{
					{Name: "regular-1"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("restartable-init-1"),
				getReadyStatus("regular-1"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.ContainersReady, v1.ConditionFalse, ContainersNotReady, "containers with unknown status: [restartable-init-2]"),
		},
		{
			spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{Name: "restartable-init-1", RestartPolicy: &containerRestartPolicyAlways},
					{Name: "restartable-init-2", RestartPolicy: &containerRestartPolicyAlways},
				},
				Containers: []v1.Container{
					{Name: "regular-1"},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("restartable-init-1"),
				getNotReadyStatus("restartable-init-2"),
				getReadyStatus("regular-1"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.ContainersReady, v1.ConditionFalse, ContainersNotReady, "containers with unready status: [restartable-init-2]"),
		},
	}

	for i, test := range tests {
		pod := &v1.Pod{Spec: test.spec}
		ready := GenerateContainersReadyCondition(pod, &v1.PodStatus{}, test.containerStatuses, test.podPhase)
		if !reflect.DeepEqual(ready, test.expectReady) {
			t.Errorf("On test case %v, expectReady:\n%+v\ngot\n%+v\n", i, test.expectReady, ready)
		}
	}
}

func TestGeneratePodReadyCondition(t *testing.T) {
	tests := []struct {
		spec              v1.PodSpec
		conditions        []v1.PodCondition
		containerStatuses []v1.ContainerStatus
		podPhase          v1.PodPhase
		expectReady       v1.PodCondition
	}{
		{
			spec:              v1.PodSpec{},
			conditions:        nil,
			containerStatuses: nil,
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionFalse, UnknownContainerStatuses, ""),
		},
		{
			spec:              v1.PodSpec{},
			conditions:        nil,
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionTrue, "", ""),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
				},
			},
			conditions:        nil,
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionFalse, ContainersNotReady, "containers with unknown status: [1234]"),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			conditions: nil,
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
				getReadyStatus("5678"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.PodReady, v1.ConditionTrue, "", ""),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			conditions: nil,
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.PodReady, v1.ConditionFalse, ContainersNotReady, "containers with unknown status: [5678]"),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			conditions: nil,
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
				getNotReadyStatus("5678"),
			},
			podPhase:    v1.PodRunning,
			expectReady: getPodCondition(v1.PodReady, v1.ConditionFalse, ContainersNotReady, "containers with unready status: [5678]"),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
				},
			},
			conditions: nil,
			containerStatuses: []v1.ContainerStatus{
				getNotReadyStatus("1234"),
			},
			podPhase:    v1.PodSucceeded,
			expectReady: getPodCondition(v1.PodReady, v1.ConditionFalse, PodCompleted, ""),
		},
		{
			spec: v1.PodSpec{
				ReadinessGates: []v1.PodReadinessGate{
					{ConditionType: v1.PodConditionType("gate1")},
				},
			},
			conditions:        nil,
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionFalse, ReadinessGatesNotReady, `corresponding condition of pod readiness gate "gate1" does not exist.`),
		},
		{
			spec: v1.PodSpec{
				ReadinessGates: []v1.PodReadinessGate{
					{ConditionType: v1.PodConditionType("gate1")},
				},
			},
			conditions: []v1.PodCondition{
				getPodCondition("gate1", v1.ConditionFalse, "", ""),
			},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionFalse, ReadinessGatesNotReady, `the status of pod readiness gate "gate1" is not "True", but False`),
		},
		{
			spec: v1.PodSpec{
				ReadinessGates: []v1.PodReadinessGate{
					{ConditionType: v1.PodConditionType("gate1")},
				},
			},
			conditions: []v1.PodCondition{
				getPodCondition("gate1", v1.ConditionTrue, "", ""),
			},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionTrue, "", ""),
		},
		{
			spec: v1.PodSpec{
				ReadinessGates: []v1.PodReadinessGate{
					{ConditionType: v1.PodConditionType("gate1")},
					{ConditionType: v1.PodConditionType("gate2")},
				},
			},
			conditions: []v1.PodCondition{
				getPodCondition("gate1", v1.ConditionTrue, "", ""),
			},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionFalse, ReadinessGatesNotReady, `corresponding condition of pod readiness gate "gate2" does not exist.`),
		},
		{
			spec: v1.PodSpec{
				ReadinessGates: []v1.PodReadinessGate{
					{ConditionType: v1.PodConditionType("gate1")},
					{ConditionType: v1.PodConditionType("gate2")},
				},
			},
			conditions: []v1.PodCondition{
				getPodCondition("gate1", v1.ConditionTrue, "", ""),
				getPodCondition("gate2", v1.ConditionFalse, "", ""),
			},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionFalse, ReadinessGatesNotReady, `the status of pod readiness gate "gate2" is not "True", but False`),
		},
		{
			spec: v1.PodSpec{
				ReadinessGates: []v1.PodReadinessGate{
					{ConditionType: v1.PodConditionType("gate1")},
					{ConditionType: v1.PodConditionType("gate2")},
				},
			},
			conditions: []v1.PodCondition{
				getPodCondition("gate1", v1.ConditionTrue, "", ""),
				getPodCondition("gate2", v1.ConditionTrue, "", ""),
			},
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionTrue, "", ""),
		},
		{
			spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "1234"},
				},
				ReadinessGates: []v1.PodReadinessGate{
					{ConditionType: v1.PodConditionType("gate1")},
				},
			},
			conditions: []v1.PodCondition{
				getPodCondition("gate1", v1.ConditionTrue, "", ""),
			},
			containerStatuses: []v1.ContainerStatus{getNotReadyStatus("1234")},
			podPhase:          v1.PodRunning,
			expectReady:       getPodCondition(v1.PodReady, v1.ConditionFalse, ContainersNotReady, "containers with unready status: [1234]"),
		},
	}

	for i, test := range tests {
		pod := &v1.Pod{Spec: test.spec}
		ready := GeneratePodReadyCondition(pod, &v1.PodStatus{}, test.conditions, test.containerStatuses, test.podPhase)
		if !reflect.DeepEqual(ready, test.expectReady) {
			t.Errorf("On test case %v, expectReady:\n%+v\ngot\n%+v\n", i, test.expectReady, ready)
		}
	}
}

func TestGeneratePodInitializedCondition(t *testing.T) {
	noInitContainer := &v1.PodSpec{}
	oneInitContainer := &v1.PodSpec{
		InitContainers: []v1.Container{
			{Name: "1234"},
		},
		Containers: []v1.Container{
			{Name: "regular"},
		},
	}
	twoInitContainer := &v1.PodSpec{
		InitContainers: []v1.Container{
			{Name: "1234"},
			{Name: "5678"},
		},
		Containers: []v1.Container{
			{Name: "regular"},
		},
	}
	oneRestartableInitContainer := &v1.PodSpec{
		InitContainers: []v1.Container{
			{
				Name: "1234",
				RestartPolicy: func() *v1.ContainerRestartPolicy {
					p := v1.ContainerRestartPolicyAlways
					return &p
				}(),
			},
		},
		Containers: []v1.Container{
			{Name: "regular"},
		},
	}
	tests := []struct {
		spec              *v1.PodSpec
		containerStatuses []v1.ContainerStatus
		podPhase          v1.PodPhase
		expected          v1.PodCondition
	}{
		{
			spec:              twoInitContainer,
			containerStatuses: nil,
			podPhase:          v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
				Reason: UnknownContainerStatuses,
			},
		},
		{
			spec:              noInitContainer,
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionTrue,
				Reason: "",
			},
		},
		{
			spec:              oneInitContainer,
			containerStatuses: []v1.ContainerStatus{},
			podPhase:          v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
				Reason: ContainersNotInitialized,
			},
		},
		{
			spec: twoInitContainer,
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
				getReadyStatus("5678"),
			},
			podPhase: v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionTrue,
				Reason: "",
			},
		},
		{
			spec: twoInitContainer,
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
			},
			podPhase: v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
				Reason: ContainersNotInitialized,
			},
		},
		{
			spec: twoInitContainer,
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
				getNotReadyStatus("5678"),
			},
			podPhase: v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
				Reason: ContainersNotInitialized,
			},
		},
		{
			spec: oneInitContainer,
			containerStatuses: []v1.ContainerStatus{
				getReadyStatus("1234"),
			},
			podPhase: v1.PodSucceeded,
			expected: v1.PodCondition{
				Status: v1.ConditionTrue,
				Reason: PodCompleted,
			},
		},
		{
			spec: oneRestartableInitContainer,
			containerStatuses: []v1.ContainerStatus{
				getNotStartedStatus("1234"),
			},
			podPhase: v1.PodPending,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
				Reason: ContainersNotInitialized,
			},
		},
		{
			spec: oneRestartableInitContainer,
			containerStatuses: []v1.ContainerStatus{
				getStartedStatus("1234"),
			},
			podPhase: v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionTrue,
			},
		},
		{
			spec: oneRestartableInitContainer,
			containerStatuses: []v1.ContainerStatus{
				getNotStartedStatus("1234"),
				{
					Name: "regular",
					State: v1.ContainerState{
						Running: &v1.ContainerStateRunning{},
					},
				},
			},
			podPhase: v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionTrue,
			},
		},
	}

	for _, test := range tests {
		test.expected.Type = v1.PodInitialized
		pod := &v1.Pod{Spec: *test.spec}
		condition := GeneratePodInitializedCondition(pod, &v1.PodStatus{}, test.containerStatuses, test.podPhase)
		assert.Equal(t, test.expected.Type, condition.Type)
		assert.Equal(t, test.expected.Status, condition.Status)
		assert.Equal(t, test.expected.Reason, condition.Reason)

	}
}

func TestGeneratePodReadyToStartContainersCondition(t *testing.T) {
	for desc, test := range map[string]struct {
		pod      *v1.Pod
		status   *kubecontainer.PodStatus
		expected v1.PodCondition
	}{
		"Empty pod status": {
			pod:    &v1.Pod{},
			status: &kubecontainer.PodStatus{},
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
			},
		},
		"Pod sandbox status not ready": {
			pod: &v1.Pod{},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
					},
				},
			},
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
			},
		},
		"Pod sandbox status ready but no IP configured": {
			pod: &v1.Pod{},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Network: &runtimeapi.PodSandboxNetworkStatus{
							Ip: "",
						},
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
				},
			},
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
			},
		},
		"Pod sandbox status ready and IP configured": {
			pod: &v1.Pod{},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Network: &runtimeapi.PodSandboxNetworkStatus{
							Ip: "10.0.0.10",
						},
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
				},
			},
			expected: v1.PodCondition{
				Status: v1.ConditionTrue,
			},
		},
	} {
		t.Run(desc, func(t *testing.T) {
			test.expected.Type = v1.PodReadyToStartContainers
			condition := GeneratePodReadyToStartContainersCondition(test.pod, &v1.PodStatus{}, test.status)
			require.Equal(t, test.expected.Type, condition.Type)
			require.Equal(t, test.expected.Status, condition.Status)
		})
	}
}

func getPodCondition(conditionType v1.PodConditionType, status v1.ConditionStatus, reason, message string) v1.PodCondition {
	return v1.PodCondition{
		Type:    conditionType,
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

func getStartedStatus(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name:    cName,
		Started: ptr.To(true),
	}
}

func getNotStartedStatus(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name:    cName,
		Started: ptr.To(false),
	}
}
