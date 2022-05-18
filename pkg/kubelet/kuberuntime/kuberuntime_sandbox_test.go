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
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
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

func TestGeneratePodSandboxWindowsConfig(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	const containerName = "container"
	gmsaCreds := "gmsa-creds"
	userName := "SYSTEM"
	trueVar := true
	falseVar := false

	testCases := []struct {
		name                      string
		hostProcessFeatureEnabled bool
		podSpec                   *v1.PodSpec
		expectedWindowsConfig     *runtimeapi.WindowsPodSandboxConfig
		expectedError             error
	}{
		{
			name:                      "Empty PodSecurityContext",
			hostProcessFeatureEnabled: false,
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{{
					Name: containerName,
				}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{},
			},
			expectedError: nil,
		},
		{
			name:                      "GMSACredentialSpec in PodSecurityContext",
			hostProcessFeatureEnabled: false,
			podSpec: &v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						GMSACredentialSpec: &gmsaCreds,
					},
				},
				Containers: []v1.Container{{
					Name: containerName,
				}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{
					CredentialSpec: "gmsa-creds",
				},
			},
			expectedError: nil,
		},
		{
			name:                      "RunAsUserName in PodSecurityContext",
			hostProcessFeatureEnabled: false,
			podSpec: &v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						RunAsUserName: &userName,
					},
				},
				Containers: []v1.Container{{
					Name: containerName,
				}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{
					RunAsUsername: "SYSTEM",
				},
			},
			expectedError: nil,
		},
		{
			name:                      "Pod with HostProcess containers and feature gate disabled",
			hostProcessFeatureEnabled: false,
			podSpec: &v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess: &trueVar,
					},
				},
				Containers: []v1.Container{{
					Name: containerName,
				}},
			},
			expectedWindowsConfig: nil,
			expectedError:         fmt.Errorf("pod contains HostProcess containers but feature 'WindowsHostProcessContainers' is not enabled"),
		},
		{
			name:                      "Pod with HostProcess containers and non-HostProcess containers",
			hostProcessFeatureEnabled: true,
			podSpec: &v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess: &trueVar,
					},
				},
				Containers: []v1.Container{{
					Name: containerName,
				}, {
					Name: containerName,
					SecurityContext: &v1.SecurityContext{
						WindowsOptions: &v1.WindowsSecurityContextOptions{
							HostProcess: &falseVar,
						},
					},
				}},
			},
			expectedWindowsConfig: nil,
			expectedError:         fmt.Errorf("pod must not contain both HostProcess and non-HostProcess containers"),
		},
		{
			name:                      "Pod with HostProcess containers and HostNetwork not set",
			hostProcessFeatureEnabled: true,
			podSpec: &v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess: &trueVar,
					},
				},
				Containers: []v1.Container{{
					Name: containerName,
				}},
			},
			expectedWindowsConfig: nil,
			expectedError:         fmt.Errorf("hostNetwork is required if Pod contains HostProcess containers"),
		},
		{
			name:                      "Pod with HostProcess containers and HostNetwork set",
			hostProcessFeatureEnabled: true,
			podSpec: &v1.PodSpec{
				HostNetwork: true,
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess: &trueVar,
					},
				},
				Containers: []v1.Container{{
					Name: containerName,
				}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{
					HostProcess: true,
				},
			},
			expectedError: nil,
		},
		{
			name:                      "Pod's WindowsOptions.HostProcess set to false and pod has HostProcess containers",
			hostProcessFeatureEnabled: true,
			podSpec: &v1.PodSpec{
				HostNetwork: true,
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess: &falseVar,
					},
				},
				Containers: []v1.Container{{
					Name: containerName,
					SecurityContext: &v1.SecurityContext{
						WindowsOptions: &v1.WindowsSecurityContextOptions{
							HostProcess: &trueVar,
						},
					},
				}},
			},
			expectedWindowsConfig: nil,
			expectedError:         fmt.Errorf("pod must not contain any HostProcess containers if Pod's WindowsOptions.HostProcess is set to false"),
		},
		{
			name:                      "Pod's security context doesn't specify HostProcess containers but Container's security context does",
			hostProcessFeatureEnabled: true,
			podSpec: &v1.PodSpec{
				HostNetwork: true,
				Containers: []v1.Container{{
					Name: containerName,
					SecurityContext: &v1.SecurityContext{
						WindowsOptions: &v1.WindowsSecurityContextOptions{
							HostProcess: &trueVar,
						},
					},
				}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{
					HostProcess: true,
				},
			},
			expectedError: nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WindowsHostProcessContainers, testCase.hostProcessFeatureEnabled)()
			pod := &v1.Pod{}
			pod.Spec = *testCase.podSpec

			wc, err := m.generatePodSandboxWindowsConfig(pod)

			assert.Equal(t, wc, testCase.expectedWindowsConfig)
			assert.Equal(t, err, testCase.expectedError)
		})
	}
}
