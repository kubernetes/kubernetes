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
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/kubernetes/pkg/features"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
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
		assert.Equal(t, filepath.Join(podLogsRootDirectory, "12345678"), path)
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

// TestCreatePodSandbox_RuntimeClass tests creating sandbox with RuntimeClasses enabled.
func TestCreatePodSandbox_RuntimeClass(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClass, true)()

	rcm := runtimeclass.NewManager(rctest.NewPopulatedDynamicClient())
	defer rctest.StartManagerSync(t, rcm)()

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
			pod := newTestPod()
			pod.Spec.RuntimeClassName = test.rcn

			id, _, err := m.createPodSandbox(pod, 1)
			if test.expectError {
				assert.Error(t, err)
				assert.Contains(t, fakeRuntime.Called, "RunPodSandbox")
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
