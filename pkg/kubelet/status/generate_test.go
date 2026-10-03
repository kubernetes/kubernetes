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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
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
		{
			spec: oneInitContainer,
			containerStatuses: []v1.ContainerStatus{{
				Name: "1234",
				State: v1.ContainerState{
					Waiting: &v1.ContainerStateWaiting{},
				},
			}, {
				Name: "regular",
				State: v1.ContainerState{
					Terminated: &v1.ContainerStateTerminated{},
				},
			}},
			podPhase: v1.PodRunning,
			expected: v1.PodCondition{
				Status: v1.ConditionTrue,
			},
		},
	}
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.ContainerRestartRules:                true,
		features.NodeDeclaredFeatures:                 true,
		features.RestartAllContainersOnContainerExits: true,
	})
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
				Status:  v1.ConditionFalse,
				Reason:  kubetypes.PodSandboxNotReadyReason,
				Message: kubetypes.PodSandboxNotReadyMsgNoPodSandbox,
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
				Status:  v1.ConditionFalse,
				Reason:  kubetypes.PodSandboxNotReadyReason,
				Message: kubetypes.PodSandboxNotReadyMsgSandboxNotReady,
			},
		},
		"Pod with multiple ready sandboxes": {
			pod: &v1.Pod{},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Network: &runtimeapi.PodSandboxNetworkStatus{
							Ip: "10.0.0.10",
						},
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(1)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
					{
						Network: &runtimeapi.PodSandboxNetworkStatus{
							Ip: "10.0.0.11",
						},
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
				},
			},
			expected: v1.PodCondition{
				Status:  v1.ConditionFalse,
				Reason:  kubetypes.PodSandboxNotReadyReason,
				Message: kubetypes.PodSandboxNotReadyMsgMultipleSandboxes,
			},
		},
		"Pod sandbox status ready but network namespace mode changed": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostNetwork: true,
				},
			},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Network: &runtimeapi.PodSandboxNetworkStatus{
							Ip: "10.0.0.10",
						},
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
						Linux: &runtimeapi.LinuxPodSandboxStatus{
							Namespaces: &runtimeapi.Namespace{
								Options: &runtimeapi.NamespaceOption{
									Network: runtimeapi.NamespaceMode_POD,
								},
							},
						},
					},
				},
			},
			expected: v1.PodCondition{
				Status:  v1.ConditionFalse,
				Reason:  kubetypes.PodSandboxNotReadyReason,
				Message: kubetypes.PodSandboxNotReadyMsgNetworkNamespaceMode,
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
				Status:  v1.ConditionFalse,
				Reason:  kubetypes.PodSandboxNotReadyReason,
				Message: kubetypes.PodSandboxNotReadyMsgNoIPAddress,
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
			require.Equal(t, test.expected.Reason, condition.Reason)
			require.Equal(t, test.expected.Message, condition.Message)
		})
	}
}

func TestGenerateAllContainersRestartingCondition(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.ContainerRestartRules:                true,
		features.NodeDeclaredFeatures:                 true,
		features.RestartAllContainersOnContainerExits: true,
	})

	restartPolicyNever := v1.ContainerRestartPolicyNever
	defaultPod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name: "container1",
			}, {
				Name:          "trigger",
				RestartPolicy: &restartPolicyNever,
				RestartPolicyRules: []v1.ContainerRestartRule{{
					Action: v1.ContainerRestartRuleActionRestartAllContainers,
					ExitCodes: &v1.ContainerRestartRuleOnExitCodes{
						Operator: v1.ContainerRestartRuleOnExitCodesOpIn,
						Values:   []int32{42},
					},
				}},
			}},
		},
	}

	for desc, test := range map[string]struct {
		podStatus    *kubecontainer.PodStatus
		oldAPIStatus *v1.PodStatus
		phase        v1.PodPhase
		expected     v1.PodCondition
	}{
		"pod pending": {
			phase: v1.PodPending,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
			},
		},
		"pod failed": {
			phase: v1.PodFailed,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
				Reason: PodFailed,
			},
		},
		"pod succeeded": {
			phase: v1.PodSucceeded,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
				Reason: PodCompleted,
			},
		},
		"container triggers RestartAllContainers rule": {
			podStatus: &kubecontainer.PodStatus{
				ContainerStatuses: []*kubecontainer.Status{
					{
						Name:  "container",
						State: kubecontainer.ContainerStateRunning,
					},
					{
						Name:     "trigger",
						State:    kubecontainer.ContainerStateExited,
						ExitCode: 42,
					},
				},
			},
			phase: v1.PodRunning,
			expected: v1.PodCondition{
				Status:  v1.ConditionTrue,
				Reason:  "RestartAllContainersStarted",
				Message: "container exited with restart policy rule",
			},
		},
		"container triggres RestartAllContainers rule, cleaning up": {
			podStatus: &kubecontainer.PodStatus{
				ContainerStatuses: []*kubecontainer.Status{
					{
						Name:  "container",
						State: kubecontainer.ContainerStateExited,
					},
					{
						Name:     "trigger",
						State:    kubecontainer.ContainerStateExited,
						ExitCode: 42,
					},
				},
			},
			oldAPIStatus: &v1.PodStatus{
				Conditions: []v1.PodCondition{{
					Type:   v1.AllContainersRestarting,
					Status: v1.ConditionTrue,
				}},
			},
			phase: v1.PodRunning,
			expected: v1.PodCondition{
				Status:  v1.ConditionTrue,
				Reason:  "RestartAllContainersStarted",
				Message: "container exited with restart policy rule",
			},
		},
		"container triggres RestartAllContainers rule, cleaned up": {
			oldAPIStatus: &v1.PodStatus{
				Conditions: []v1.PodCondition{{
					Type:   v1.AllContainersRestarting,
					Status: v1.ConditionTrue,
				}},
			},
			phase: v1.PodPending,
			expected: v1.PodCondition{
				Status: v1.ConditionFalse,
			},
		},
	} {
		t.Run(desc, func(t *testing.T) {
			test.expected.Type = v1.AllContainersRestarting
			podStatus := &kubecontainer.PodStatus{}
			if test.podStatus != nil {
				podStatus = test.podStatus
			}
			condition := GenerateAllContainersRestartingCondition(defaultPod, podStatus, test.oldAPIStatus, test.phase)
			require.Equal(t, test.expected, condition)
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

// linuxUser builds a ContainerUser reporting the given observed UID/GID/supplementalGroups.
func linuxUser(uid, gid int64, supplementalGroups ...int64) *v1.ContainerUser {
	return &v1.ContainerUser{Linux: &v1.LinuxContainerUser{UID: uid, GID: gid, SupplementalGroups: supplementalGroups}}
}

func TestGenerateInsecureIDCondition(t *testing.T) {
	hostUsersFalse := false

	for desc, test := range map[string]struct {
		generate              func(pod *v1.Pod, oldPodStatus *v1.PodStatus, containerStatuses []v1.ContainerStatus) v1.PodCondition
		expectedType          v1.PodConditionType
		pod                   *v1.Pod
		containerStatuses     []v1.ContainerStatus
		expectedStatus        v1.ConditionStatus
		expectedReason        string
		expectMessageContains []string
		expectMessageExcludes []string
	}{
		"UID: implicitly-root: no runAsUser anywhere": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(0, 0)},
			},
			expectedStatus:        v1.ConditionTrue,
			expectedReason:        ImplicitlyInsecureUserID,
			expectMessageContains: []string{"c1"},
		},
		"UID: explicitly-root: container sets runAsUser=0": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsUser: ptr.To[int64](0)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(0, 0)},
			},
			expectedStatus: v1.ConditionFalse,
		},
		"UID: explicitly-root: pod sets runAsUser=0": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{RunAsUser: ptr.To[int64](0)},
					Containers:      []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(0, 0)},
			},
			expectedStatus: v1.ConditionFalse,
		},
		"UID: non-root: actual UID is non-zero": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 1000)},
			},
			expectedStatus: v1.ConditionFalse,
		},
		"UID: user namespaces: hostUsers=false exempts pod even if reported UID is 0": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers:  &hostUsersFalse,
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(0, 0)},
			},
			expectedStatus: v1.ConditionFalse,
		},
		"UID: container not started yet: no User reported": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1"},
			},
			expectedStatus:        v1.ConditionUnknown,
			expectMessageContains: []string{"c1"},
		},
		"UID: multiple containers: only offenders named in message": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "c1"},
						{Name: "c2", SecurityContext: &v1.SecurityContext{RunAsUser: ptr.To[int64](0)}},
					},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(0, 0)},
				{Name: "c2", User: linuxUser(0, 0)},
			},
			expectedStatus:        v1.ConditionTrue,
			expectedReason:        ImplicitlyInsecureUserID,
			expectMessageContains: []string{"c1"},
			expectMessageExcludes: []string{"c2"},
		},
		"UID: ephemeral container: implicitly-root": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers:          []v1.Container{{Name: "c1", SecurityContext: &v1.SecurityContext{RunAsUser: ptr.To[int64](1000)}}},
					EphemeralContainers: []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug"}}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 1000)},
				{Name: "debug", User: linuxUser(0, 0)},
			},
			expectedStatus:        v1.ConditionTrue,
			expectedReason:        ImplicitlyInsecureUserID,
			expectMessageContains: []string{"debug"},
		},
		"GID: implicitly-root": {
			generate:     GenerateInsecureGroupIDCondition,
			expectedType: v1.InsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 0)},
			},
			expectedStatus: v1.ConditionTrue,
			expectedReason: ImplicitlyInsecureGroupID,
		},
		"GID: explicitly-root: container sets runAsGroup=0": {
			generate:     GenerateInsecureGroupIDCondition,
			expectedType: v1.InsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](0)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 0)},
			},
			expectedStatus: v1.ConditionFalse,
		},
		"GID: explicitly-root: pod sets runAsGroup=0": {
			generate:     GenerateInsecureGroupIDCondition,
			expectedType: v1.InsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{RunAsGroup: ptr.To[int64](0)},
					Containers:      []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 0)},
			},
			expectedStatus: v1.ConditionFalse,
		},
		"GID: supplementalGroups: implicitly-root via supplementalGroupsPolicy=Merge": {
			generate:     GenerateInsecureGroupIDCondition,
			expectedType: v1.InsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](1000)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 1000, 0, 1000)},
			},
			expectedStatus:        v1.ConditionTrue,
			expectedReason:        ImplicitlyInsecureGroupID,
			expectMessageContains: []string{"merged into supplementalGroups"},
		},
		"GID: supplementalGroups: explicitly-root via fsGroup=0": {
			generate:     GenerateInsecureGroupIDCondition,
			expectedType: v1.InsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{FSGroup: ptr.To[int64](0)},
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](1000)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 1000, 0, 1000)},
			},
			expectedStatus: v1.ConditionFalse,
		},
		"GID: supplementalGroups: primary GID mirroring is not double-reported": {
			generate:     GenerateInsecureGroupIDCondition,
			expectedType: v1.InsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 0, 0)},
			},
			expectedStatus:        v1.ConditionTrue,
			expectedReason:        ImplicitlyInsecureGroupID,
			expectMessageContains: []string{"without runAsGroup set"},
		},
		"GID: supplementalGroups: combined with primary-GID container in the same pod": {
			generate:     GenerateInsecureGroupIDCondition,
			expectedType: v1.InsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "c1"},
						{Name: "c2", SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](1000)}},
					},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 0)},
				{Name: "c2", User: linuxUser(1000, 1000, 0, 1000)},
			},
			expectedStatus: v1.ConditionTrue,
			expectedReason: ImplicitlyInsecureGroupID,
			expectMessageContains: []string{
				"[c1] running as GID 0 without runAsGroup set",
				"[c2] running with GID 0 merged into supplementalGroups",
			},
		},
		"GID: container not started yet: no User reported": {
			generate:     GenerateInsecureGroupIDCondition,
			expectedType: v1.InsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1"},
			},
			expectedStatus:        v1.ConditionUnknown,
			expectMessageContains: []string{"c1"},
		},
		"UID: one container insecure, one not started yet: insecure wins over unknown": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "c1"},
						{Name: "c2"},
					},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(0, 0)},
				{Name: "c2"},
			},
			expectedStatus:        v1.ConditionTrue,
			expectedReason:        ImplicitlyInsecureUserID,
			expectMessageContains: []string{"c1"},
		},
		"UID: one container secure, one not started yet: unknown wins over false": {
			generate:     GenerateInsecureUserIDCondition,
			expectedType: v1.InsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "c1"},
						{Name: "c2"},
					},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 1000)},
				{Name: "c2"},
			},
			expectedStatus:        v1.ConditionUnknown,
			expectMessageContains: []string{"c2"},
			expectMessageExcludes: []string{"c1"},
		},
	} {
		t.Run(desc, func(t *testing.T) {
			condition := test.generate(test.pod, &v1.PodStatus{}, test.containerStatuses)
			require.Equal(t, test.expectedType, condition.Type)
			require.Equal(t, test.expectedStatus, condition.Status)
			require.Equal(t, test.expectedReason, condition.Reason)
			for _, want := range test.expectMessageContains {
				assert.Contains(t, condition.Message, want)
			}
			for _, notWant := range test.expectMessageExcludes {
				assert.NotContains(t, condition.Message, notWant)
			}
		})
	}
}

func TestIsPodExplicitlyInsecureID(t *testing.T) {
	for desc, test := range map[string]struct {
		check             func(pod *v1.Pod, containerStatuses []v1.ContainerStatus) bool
		pod               *v1.Pod
		containerStatuses []v1.ContainerStatus
		want              bool
	}{
		"UID: explicit: container sets runAsUser=0": {
			check: IsPodExplicitlyInsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsUser: ptr.To[int64](0)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(0, 0)}},
			want:              true,
		},
		"UID: implicit: no runAsUser set": {
			check: IsPodExplicitlyInsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(0, 0)}},
			want:              false,
		},
		"UID: requested a different UID than observed": {
			check: IsPodExplicitlyInsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsUser: ptr.To[int64](1000)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(0, 0)}},
			want:              false,
		},
		"UID: ephemeral container: explicitly requests runAsUser=0": {
			check: IsPodExplicitlyInsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
					EphemeralContainers: []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:            "debug",
						SecurityContext: &v1.SecurityContext{RunAsUser: ptr.To[int64](0)},
					}}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 1000)},
				{Name: "debug", User: linuxUser(0, 0)},
			},
			want: true,
		},
		"UID: user namespaces: hostUsers=false exempts pod": {
			check: IsPodExplicitlyInsecureUserID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: new(bool),
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsUser: ptr.To[int64](0)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(0, 0)}},
			want:              false,
		},
		"GID: explicit: container sets runAsGroup=0": {
			check: IsPodExplicitlyInsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](0)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(1000, 0)}},
			want:              true,
		},
		"GID: implicit: no runAsGroup set": {
			check: IsPodExplicitlyInsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(1000, 0)}},
			want:              false,
		},
		"GID: requested a different GID than observed": {
			check: IsPodExplicitlyInsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](1000)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(1000, 0)}},
			want:              false,
		},
		"GID: ephemeral container: explicitly requests runAsGroup=0": {
			check: IsPodExplicitlyInsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
					EphemeralContainers: []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:            "debug",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](0)},
					}}},
				},
			},
			containerStatuses: []v1.ContainerStatus{
				{Name: "c1", User: linuxUser(1000, 1000)},
				{Name: "debug", User: linuxUser(0, 0)},
			},
			want: true,
		},
		"GID: user namespaces: hostUsers=false exempts pod": {
			check: IsPodExplicitlyInsecureGroupID,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: new(bool),
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](0)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(1000, 0)}},
			want:              false,
		},
		"supplementalGroups: implicit: supplementalGroupsPolicy=Merge pulls in GID 0": {
			check: IsPodImplicitlyInsecureSupplementalGroups,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](1000)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(1000, 1000, 0, 1000)}},
			want:              true,
		},
		"supplementalGroups: explicit: pod sets fsGroup=0": {
			check: IsPodExplicitlyInsecureSupplementalGroups,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{FSGroup: ptr.To[int64](0)},
					Containers: []v1.Container{{
						Name:            "c1",
						SecurityContext: &v1.SecurityContext{RunAsGroup: ptr.To[int64](1000)},
					}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(1000, 1000, 0, 1000)}},
			want:              true,
		},
		"supplementalGroups: primary GID mirroring is excluded from the supplementalGroups check": {
			check: IsPodImplicitlyInsecureSupplementalGroups,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(1000, 0, 0)}},
			want:              false,
		},
		"supplementalGroups: user namespaces: hostUsers=false exempts pod": {
			check: IsPodImplicitlyInsecureSupplementalGroups,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers:  new(bool),
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			containerStatuses: []v1.ContainerStatus{{Name: "c1", User: linuxUser(1000, 1000, 0, 1000)}},
			want:              false,
		},
	} {
		t.Run(desc, func(t *testing.T) {
			require.Equal(t, test.want, test.check(test.pod, test.containerStatuses))
		})
	}
}
