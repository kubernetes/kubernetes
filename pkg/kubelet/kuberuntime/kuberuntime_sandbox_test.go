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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	apitest "k8s.io/cri-api/pkg/apis/testing"
	"k8s.io/kubernetes/pkg/features"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/utils/pointer"
)

// TestCreatePodSandbox tests creating sandbox and its corresponding pod log directory.
func TestCreatePodSandbox(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)
	pod := newTestPod()

	fakeOS := m.osInterface.(*containertest.FakeOS)
	fakeOS.MkdirAllFn = func(path string, perm os.FileMode) error {
		// Check pod logs root directory is created.
		assert.Equal(t, filepath.Join(podLogsRootDirectory, pod.Namespace+"_"+pod.Name+"_12345678"), path)
		assert.Equal(t, os.FileMode(0755), perm)
		return nil
	}
	id, _, err := m.createPodSandbox(pod, 1)
	assert.NoError(t, err)
	assert.Contains(t, fakeRuntime.Called, "RunPodSandbox")
	sandboxes, err := fakeRuntime.ListPodSandbox(&runtimeapi.PodSandboxFilter{Id: id})
	assert.NoError(t, err)
	assert.Equal(t, len(sandboxes), 1)
	// TODO Check pod sandbox configuration
}

func TestGeneratePodSandboxLinuxConfigSeccomp(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	tests := []struct {
		description     string
		pod             *v1.Pod
		expectedProfile string
	}{
		{
			description:     "no seccomp defined at pod level should return runtime/default",
			pod:             newSeccompPod(nil, nil, "", "runtime/default"),
			expectedProfile: "runtime/default",
		},
		{
			description:     "seccomp field defined at pod level should not be honoured",
			pod:             newSeccompPod(&v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}, nil, "", ""),
			expectedProfile: "runtime/default",
		},
		{
			description:     "seccomp field defined at container level should not be honoured",
			pod:             newSeccompPod(nil, &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}, "", ""),
			expectedProfile: "runtime/default",
		},
		{
			description:     "seccomp annotation defined at pod level should not be honoured",
			pod:             newSeccompPod(nil, nil, "unconfined", ""),
			expectedProfile: "runtime/default",
		},
		{
			description:     "seccomp annotation defined at container level should not be honoured",
			pod:             newSeccompPod(nil, nil, "", "unconfined"),
			expectedProfile: "runtime/default",
		},
	}

	for i, test := range tests {
		config, _ := m.generatePodSandboxLinuxConfig(test.pod)
		actualProfile := config.SecurityContext.SeccompProfilePath
		assert.Equal(t, test.expectedProfile, actualProfile, "TestCase[%d]: %s", i, test.description)
	}
}

// TestCreatePodSandbox_RuntimeClass tests creating sandbox with RuntimeClasses enabled.
func TestCreatePodSandbox_RuntimeClass(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClass, true)()

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
		"valid RuntimeClass":       {rcn: pointer.StringPtr(rctest.SandboxRuntimeClass), expectedHandler: rctest.SandboxRuntimeHandler},
		"missing RuntimeClass":     {rcn: pointer.StringPtr("phantom"), expectError: true},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			fakeRuntime.Called = []string{}
			pod := newTestPod()
			pod.Spec.RuntimeClassName = test.rcn

			id, _, err := m.createPodSandbox(pod, 1)
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
				},
			},
		},
	}
}

func newSeccompPod(podFieldProfile, containerFieldProfile *v1.SeccompProfile, podAnnotationProfile, containerAnnotationProfile string) *v1.Pod {
	pod := newTestPod()
	if podAnnotationProfile != "" {
		pod.Annotations = map[string]string{v1.SeccompPodAnnotationKey: podAnnotationProfile}
	}
	if containerAnnotationProfile != "" {
		pod.Annotations = map[string]string{v1.SeccompContainerAnnotationKeyPrefix + "": containerAnnotationProfile}
	}
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

// TestDeleteSandbox tests removing the sandbox.
func TestDeleteSandbox(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)
	pod := &v1.Pod{
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
				},
			},
		},
	}

	sandbox := makeFakePodSandbox(t, m, sandboxTemplate{
		pod:       pod,
		createdAt: fakeCreatedAt,
		state:     runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
	})
	fakeRuntime.SetFakeSandboxes([]*apitest.FakePodSandbox{sandbox})

	err = m.DeleteSandbox(sandbox.Id)
	assert.NoError(t, err)
	assert.Contains(t, fakeRuntime.Called, "StopPodSandbox")
	assert.Contains(t, fakeRuntime.Called, "RemovePodSandbox")
	containers, err := fakeRuntime.ListPodSandbox(&runtimeapi.PodSandboxFilter{Id: sandbox.Id})
	assert.NoError(t, err)
	assert.Empty(t, containers)
}
