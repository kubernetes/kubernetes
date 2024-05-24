/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubecontainertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

func TestPodSandboxChanged(t *testing.T) {
	for desc, test := range map[string]struct {
		pod               *v1.Pod
		status            *kubecontainer.PodStatus
		expectedChanged   bool
		expectedAttempt   uint32
		expectedSandboxID string
	}{
		"Pod with no existing sandboxes": {
			pod:               &v1.Pod{},
			status:            &kubecontainer.PodStatus{},
			expectedChanged:   true,
			expectedAttempt:   0,
			expectedSandboxID: "",
		},
		"Pod with multiple ready sandbox statuses": {
			pod: &v1.Pod{},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Id:       "sandboxID2",
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(1)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
					{
						Id:       "sandboxID1",
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
				},
			},
			expectedChanged:   true,
			expectedAttempt:   2,
			expectedSandboxID: "sandboxID2",
		},
		"Pod with no ready sandbox statuses": {
			pod: &v1.Pod{},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Id:       "sandboxID2",
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(1)},
						State:    runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
					},
					{
						Id:       "sandboxID1",
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
					},
				},
			},
			expectedChanged:   true,
			expectedAttempt:   2,
			expectedSandboxID: "sandboxID2",
		},
		"Pod with ready sandbox status but network namespace mismatch": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostNetwork: true,
				},
			},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Id: "sandboxID1",
						Linux: &runtimeapi.LinuxPodSandboxStatus{
							Namespaces: &runtimeapi.Namespace{
								Options: &runtimeapi.NamespaceOption{
									Network: runtimeapi.NamespaceMode_POD,
								},
							},
						},
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
				},
			},
			expectedChanged:   true,
			expectedAttempt:   1,
			expectedSandboxID: "",
		},
		"Pod with ready sandbox status but no IP": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostNetwork: false,
				},
			},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Id: "sandboxID1",
						Network: &runtimeapi.PodSandboxNetworkStatus{
							Ip: "",
						},
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
				},
			},
			expectedChanged:   true,
			expectedAttempt:   1,
			expectedSandboxID: "sandboxID1",
		},
		"Pod with ready sandbox status with IP": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostNetwork: false,
				},
			},
			status: &kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{
					{
						Id: "sandboxID1",
						Network: &runtimeapi.PodSandboxNetworkStatus{
							Ip: "10.0.0.10",
						},
						Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
						State:    runtimeapi.PodSandboxState_SANDBOX_READY,
					},
				},
			},
			expectedChanged:   false,
			expectedAttempt:   0,
			expectedSandboxID: "sandboxID1",
		},
	} {
		t.Run(desc, func(t *testing.T) {
			changed, attempt, id := PodSandboxChanged(test.pod, test.status)
			require.Equal(t, test.expectedChanged, changed)
			require.Equal(t, test.expectedAttempt, attempt)
			require.Equal(t, test.expectedSandboxID, id)
		})
	}
}

func TestNamespacesForPod(t *testing.T) {
	for desc, test := range map[string]struct {
		input    *v1.Pod
		expected *runtimeapi.NamespaceOption
	}{
		"nil pod -> default v1 namespaces": {
			input: nil,
			expected: &runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_POD,
				Network: runtimeapi.NamespaceMode_POD,
				Pid:     runtimeapi.NamespaceMode_CONTAINER,
			},
		},
		"v1.Pod default namespaces": {
			input: &v1.Pod{},
			expected: &runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_POD,
				Network: runtimeapi.NamespaceMode_POD,
				Pid:     runtimeapi.NamespaceMode_CONTAINER,
			},
		},
		"Host Namespaces": {
			input: &v1.Pod{
				Spec: v1.PodSpec{
					HostIPC:     true,
					HostNetwork: true,
					HostPID:     true,
				},
			},
			expected: &runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_NODE,
				Network: runtimeapi.NamespaceMode_NODE,
				Pid:     runtimeapi.NamespaceMode_NODE,
			},
		},
		"Shared Process Namespace (feature enabled)": {
			input: &v1.Pod{
				Spec: v1.PodSpec{
					ShareProcessNamespace: &[]bool{true}[0],
				},
			},
			expected: &runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_POD,
				Network: runtimeapi.NamespaceMode_POD,
				Pid:     runtimeapi.NamespaceMode_POD,
			},
		},
		"Shared Process Namespace, redundant flag (feature enabled)": {
			input: &v1.Pod{
				Spec: v1.PodSpec{
					ShareProcessNamespace: &[]bool{false}[0],
				},
			},
			expected: &runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_POD,
				Network: runtimeapi.NamespaceMode_POD,
				Pid:     runtimeapi.NamespaceMode_CONTAINER,
			},
		},
	} {
		t.Run(desc, func(t *testing.T) {
			actual, err := NamespacesForPod(test.input, &kubecontainertest.FakeRuntimeHelper{}, nil)
			require.NoError(t, err)
			require.Equal(t, test.expected, actual)
		})
	}
}
