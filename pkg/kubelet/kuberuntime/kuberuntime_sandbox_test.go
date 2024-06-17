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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
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

func TestGeneratePodSandboxWindowsConfig_HostProcess(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	const containerName = "container"
	gmsaCreds := "gmsa-creds"
	userName := "SYSTEM"
	trueVar := true
	falseVar := false

	testCases := []struct {
		name                  string
		podSpec               *v1.PodSpec
		expectedWindowsConfig *runtimeapi.WindowsPodSandboxConfig
		expectedError         error
	}{
		{
			name: "Empty PodSecurityContext",
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
			name: "GMSACredentialSpec in PodSecurityContext",
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
			name: "RunAsUserName in PodSecurityContext",
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
			name: "Pod with HostProcess containers and non-HostProcess containers",
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
			name: "Pod with HostProcess containers and HostNetwork not set",
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
			name: "Pod with HostProcess containers and HostNetwork set",
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
			name: "Pod's WindowsOptions.HostProcess set to false and pod has HostProcess containers",
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
			name: "Pod's security context doesn't specify HostProcess containers but Container's security context does",
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
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WindowsHostNetwork, false)
			pod := &v1.Pod{}
			pod.Spec = *testCase.podSpec

			wc, err := m.generatePodSandboxWindowsConfig(pod)

			assert.Equal(t, testCase.expectedWindowsConfig, wc)
			assert.Equal(t, testCase.expectedError, err)
		})
	}
}

func TestGeneratePodSandboxWindowsConfig_HostNetwork(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	const containerName = "container"

	testCases := []struct {
		name                      string
		hostNetworkFeatureEnabled bool
		podSpec                   *v1.PodSpec
		expectedWindowsConfig     *runtimeapi.WindowsPodSandboxConfig
	}{
		{
			name:                      "feature disabled, hostNetwork=false",
			hostNetworkFeatureEnabled: false,
			podSpec: &v1.PodSpec{
				HostNetwork: false,
				Containers:  []v1.Container{{Name: containerName}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{},
			},
		},
		{
			name:                      "feature disabled, hostNetwork=true",
			hostNetworkFeatureEnabled: false,
			podSpec: &v1.PodSpec{
				HostNetwork: true,
				Containers:  []v1.Container{{Name: containerName}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{},
			}},
		{
			name:                      "feature enabled, hostNetwork=false",
			hostNetworkFeatureEnabled: true,
			podSpec: &v1.PodSpec{
				HostNetwork: false,
				Containers:  []v1.Container{{Name: containerName}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{
					NamespaceOptions: &runtimeapi.WindowsNamespaceOption{
						Network: runtimeapi.NamespaceMode_POD,
					},
				},
			},
		},
		{
			name:                      "feature enabled, hostNetwork=true",
			hostNetworkFeatureEnabled: true,
			podSpec: &v1.PodSpec{
				HostNetwork: true,
				Containers:  []v1.Container{{Name: containerName}},
			},
			expectedWindowsConfig: &runtimeapi.WindowsPodSandboxConfig{
				SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{
					NamespaceOptions: &runtimeapi.WindowsNamespaceOption{
						Network: runtimeapi.NamespaceMode_NODE,
					},
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WindowsHostNetwork, testCase.hostNetworkFeatureEnabled)
			pod := &v1.Pod{}
			pod.Spec = *testCase.podSpec

			wc, err := m.generatePodSandboxWindowsConfig(pod)

			assert.Equal(t, testCase.expectedWindowsConfig, wc)
			assert.Equal(t, nil, err)
		})
	}
}
