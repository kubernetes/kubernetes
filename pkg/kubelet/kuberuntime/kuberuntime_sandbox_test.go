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

package kuberuntime

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/utils/pointer"
)

const testPodLogsDirectory = "/var/log/pods"

func TestGeneratePodSandboxConfig(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)
	pod := newTestPod()

	expectedLogDirectory := filepath.Join(testPodLogsDirectory, pod.Namespace+"_"+pod.Name+"_12345678")
	expectedLabels := map[string]string{
		"io.kubernetes.pod.name":      pod.Name,
		"io.kubernetes.pod.namespace": pod.Namespace,
		"io.kubernetes.pod.uid":       string(pod.UID),
	}
	expectedMetadata := &runtimeapi.PodSandboxMetadata{
		Name:      pod.Name,
		Namespace: pod.Namespace,
		Uid:       string(pod.UID),
		Attempt:   uint32(1),
	}
	expectedPortMappings := []*runtimeapi.PortMapping{
		{
			HostPort: 8080,
		},
	}

	podSandboxConfig, err := m.generatePodSandboxConfig(pod, 1)
	assert.NoError(t, err)
	assert.Equal(t, expectedLabels, podSandboxConfig.Labels)
	assert.Equal(t, expectedLogDirectory, podSandboxConfig.LogDirectory)
	assert.Equal(t, expectedMetadata, podSandboxConfig.Metadata)
	assert.Equal(t, expectedPortMappings, podSandboxConfig.PortMappings)
}

// TestCreatePodSandbox tests creating sandbox and its corresponding pod log directory.
func TestCreatePodSandbox(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)
	pod := newTestPod()

	fakeOS := m.osInterface.(*containertest.FakeOS)
	fakeOS.MkdirAllFn = func(path string, perm os.FileMode) error {
		// Check pod logs root directory is created.
		assert.Equal(t, filepath.Join(testPodLogsDirectory, pod.Namespace+"_"+pod.Name+"_12345678"), path)
		assert.Equal(t, os.FileMode(0755), perm)
		return nil
	}
	id, _, err := m.createPodSandbox(ctx, pod, 1)
	assert.NoError(t, err)
	assert.Contains(t, fakeRuntime.Called, "RunPodSandbox")
	sandboxes, err := fakeRuntime.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{Id: id})
	assert.NoError(t, err)
	assert.Equal(t, len(sandboxes), 1)
	assert.Equal(t, sandboxes[0].Id, fmt.Sprintf("%s_%s_%s_1", pod.Name, pod.Namespace, pod.UID))
	assert.Equal(t, sandboxes[0].State, runtimeapi.PodSandboxState_SANDBOX_READY)
}

func TestGeneratePodSandboxLinuxConfigSeccomp(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	tests := []struct {
		description     string
		pod             *v1.Pod
		expectedProfile v1.SeccompProfileType
	}{
		{
			description:     "no seccomp defined at pod level should return runtime/default",
			pod:             newSeccompPod(nil, nil, "", "runtime/default"),
			expectedProfile: v1.SeccompProfileTypeRuntimeDefault,
		},
		{
			description:     "seccomp field defined at pod level should not be honoured",
			pod:             newSeccompPod(&v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}, nil, "", ""),
			expectedProfile: v1.SeccompProfileTypeRuntimeDefault,
		},
		{
			description:     "seccomp field defined at container level should not be honoured",
			pod:             newSeccompPod(nil, &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}, "", ""),
			expectedProfile: v1.SeccompProfileTypeRuntimeDefault,
		},
		{
			description:     "seccomp annotation defined at pod level should not be honoured",
			pod:             newSeccompPod(nil, nil, "unconfined", ""),
			expectedProfile: v1.SeccompProfileTypeRuntimeDefault,
		},
		{
			description:     "seccomp annotation defined at container level should not be honoured",
			pod:             newSeccompPod(nil, nil, "", "unconfined"),
			expectedProfile: v1.SeccompProfileTypeRuntimeDefault,
		},
	}

	for i, test := range tests {
		config, _ := m.generatePodSandboxLinuxConfig(test.pod)
		actualProfile := config.SecurityContext.Seccomp.ProfileType.String()
		assert.EqualValues(t, test.expectedProfile, actualProfile, "TestCase[%d]: %s", i, test.description)
	}
}

// TestCreatePodSandbox_RuntimeClass tests creating sandbox with RuntimeClasses enabled.
func TestCreatePodSandbox_RuntimeClass(t *testing.T) {
	ctx := context.Background()
	rcm := runtimeclass.NewManager(rctest.NewPopulatedClient())
	defer rctest.StartManagerSync(rcm)()

	fakeRuntime, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)
	m.runtimeClassManager = rcm

	tests := map[string]struct {
		rcn             *string
		expectedHandler string
		expectError     bool
	}{
		"unspecified RuntimeClass": {rcn: nil, expectedHandler: ""},
		"valid RuntimeClass":       {rcn: pointer.String(rctest.SandboxRuntimeClass), expectedHandler: rctest.SandboxRuntimeHandler},
		"missing RuntimeClass":     {rcn: pointer.String("phantom"), expectError: true},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			fakeRuntime.Called = []string{}
			pod := newTestPod()
			pod.Spec.RuntimeClassName = test.rcn

			id, _, err := m.createPodSandbox(ctx, pod, 1)
			if test.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Contains(t, fakeRuntime.Called, "RunPodSandbox")
				assert.Equal(t, test.expectedHandler, fakeRuntime.Sandboxes[id].RuntimeHandler)
			}
		})
	}
}

func newTestPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Ports: []v1.ContainerPort{
						{
							HostPort: 8080,
						},
					},
				},
			},
		},
	}
}

func newSeccompPod(podFieldProfile, containerFieldProfile *v1.SeccompProfile, podAnnotationProfile, containerAnnotationProfile string) *v1.Pod {
	pod := newTestPod()
	if podFieldProfile != nil {
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			SeccompProfile: podFieldProfile,
		}
	}
	if containerFieldProfile != nil {
		pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
			SeccompProfile: containerFieldProfile,
		}
	}
	return pod
}
