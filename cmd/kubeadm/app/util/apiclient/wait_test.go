/*
Copyright 2024 The Kubernetes Authors.

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

package apiclient

import (
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestGetControlPlaneComponents(t *testing.T) {
	getTestPod := func(command []string) *v1.Pod {
		pod := &v1.Pod{
			Spec: v1.PodSpec{},
		}
		if command != nil {
			pod.Spec.Containers = []v1.Container{{}}
			if len(command) > 0 {
				pod.Spec.Containers[0].Command = command
			}
		}
		return pod
	}
	testCases := []struct {
		name          string
		setup         func() map[string]*v1.Pod
		expected      []controlPlaneComponent
		expectedError string
	}{
		{
			name: "valid: all port and addresses from config",
			setup: func() map[string]*v1.Pod {
				var (
					pod    *v1.Pod
					podMap = map[string]*v1.Pod{}
				)
				pod = getTestPod([]string{
					constants.KubeAPIServer,
					fmt.Sprintf("--%s=%s", argAdvertiseAddress, "fd00:1::"),
					fmt.Sprintf("--%s=%s", argPort, "1111"),
				})
				podMap[constants.KubeAPIServer] = pod
				pod = getTestPod([]string{
					constants.KubeControllerManager,
					fmt.Sprintf("--%s=%s", argBindAddress, "127.0.0.1"),
					fmt.Sprintf("--%s=%s", argPort, "2222"),
				})
				podMap[constants.KubeControllerManager] = pod
				pod = getTestPod([]string{
					constants.KubeScheduler,
					fmt.Sprintf("--%s=%s", argBindAddress, "127.0.0.1"),
					fmt.Sprintf("--%s=%s", argPort, "3333"),
				})
				podMap[constants.KubeScheduler] = pod
				return podMap
			},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", addressPort: "[fd00:1::]:1111", endpoint: endpointLivez},
				{name: "kube-controller-manager", addressPort: "127.0.0.1:2222", endpoint: endpointHealthz},
				{name: "kube-scheduler", addressPort: "127.0.0.1:3333", endpoint: endpointLivez},
			},
		},
		{
			name: "valid: all port and addresses from config (alt. formatting)",
			setup: func() map[string]*v1.Pod {
				var (
					pod    *v1.Pod
					podMap = map[string]*v1.Pod{}
				)
				pod = getTestPod([]string{
					constants.KubeAPIServer,
					fmt.Sprintf("-%s=%s", argAdvertiseAddress, "fd00:1::"),
					fmt.Sprintf("-%s=%s", argPort, "1111"),
				})
				podMap[constants.KubeAPIServer] = pod
				pod = getTestPod([]string{
					constants.KubeControllerManager,
					fmt.Sprintf("-%s %s", argBindAddress, "127.0.0.1"),
					fmt.Sprintf("-%s %s", argPort, "2222"),
				})
				podMap[constants.KubeControllerManager] = pod
				pod = getTestPod([]string{
					constants.KubeScheduler,
					fmt.Sprintf("-%s %s", argBindAddress, "127.0.0.1"),
					fmt.Sprintf("-%s %s", argPort, "3333"),
				})
				podMap[constants.KubeScheduler] = pod
				return podMap
			},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", addressPort: "[fd00:1::]:1111", endpoint: endpointLivez},
				{name: "kube-controller-manager", addressPort: "127.0.0.1:2222", endpoint: endpointHealthz},
				{name: "kube-scheduler", addressPort: "127.0.0.1:3333", endpoint: endpointLivez},
			},
		},
		{
			name: "valid: default ports and addresses",
			setup: func() map[string]*v1.Pod {
				var (
					pod    *v1.Pod
					podMap = map[string]*v1.Pod{}
				)
				pod = getTestPod([]string{
					constants.KubeAPIServer,
				})
				podMap[constants.KubeAPIServer] = pod
				pod = getTestPod([]string{
					constants.KubeControllerManager,
				})
				podMap[constants.KubeControllerManager] = pod
				pod = getTestPod([]string{
					constants.KubeScheduler,
				})
				podMap[constants.KubeScheduler] = pod
				return podMap
			},
			expected: []controlPlaneComponent{
				{name: "kube-apiserver", addressPort: "192.168.0.1:6443", endpoint: endpointLivez},
				{name: "kube-controller-manager", addressPort: "127.0.0.1:10257", endpoint: endpointHealthz},
				{name: "kube-scheduler", addressPort: "127.0.0.1:10259", endpoint: endpointLivez},
			},
		},
		{
			name: "invalid: nil Pods in map",
			setup: func() map[string]*v1.Pod {
				return map[string]*v1.Pod{}
			},
			expectedError: `[got nil Pod for component "kube-apiserver", ` +
				`got nil Pod for component "kube-controller-manager", ` +
				`got nil Pod for component "kube-scheduler"]`,
		},
		{
			name: "invalid: empty commands in containers",
			setup: func() map[string]*v1.Pod {
				podMap := map[string]*v1.Pod{}
				podMap[constants.KubeAPIServer] = getTestPod([]string{})
				podMap[constants.KubeControllerManager] = getTestPod([]string{})
				podMap[constants.KubeScheduler] = getTestPod([]string{})
				return podMap
			},
			expectedError: `[the Pod has no container command starting with "kube-apiserver", ` +
				`the Pod has no container command starting with "kube-controller-manager", ` +
				`the Pod has no container command starting with "kube-scheduler"]`,
		},
		{
			name: "invalid: missing commands in containers",
			setup: func() map[string]*v1.Pod {
				var (
					pod    = getTestPod([]string{""})
					podMap = map[string]*v1.Pod{}
				)
				podMap[constants.KubeAPIServer] = pod
				podMap[constants.KubeControllerManager] = pod
				podMap[constants.KubeScheduler] = pod
				return podMap
			},
			expectedError: `[the Pod has no container command starting with "kube-apiserver", ` +
				`the Pod has no container command starting with "kube-controller-manager", ` +
				`the Pod has no container command starting with "kube-scheduler"]`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.setup()
			actual, err := getControlPlaneComponents(m, "192.168.0.1")
			if err != nil {
				if err.Error() != tc.expectedError {
					t.Fatalf("expected error:\n%v\ngot:\n%v",
						tc.expectedError, err)
				}
			}
			if !reflect.DeepEqual(tc.expected, actual) {
				t.Fatalf("expected result: %+v, got: %+v", tc.expected, actual)
			}
		})
	}
}
