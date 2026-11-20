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
	"net"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

const testPodLogsDirectory = "/var/log/pods"

func TestGeneratePodSandboxConfig(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
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

	podSandboxConfig, err := m.generatePodSandboxConfig(tCtx, pod, 1)
	assert.NoError(t, err)
	assert.Equal(t, expectedLabels, podSandboxConfig.Labels)
	assert.Equal(t, expectedLogDirectory, podSandboxConfig.LogDirectory)
	assert.Equal(t, expectedMetadata, podSandboxConfig.Metadata)
	assert.Equal(t, expectedPortMappings, podSandboxConfig.PortMappings)
	assert.Equal(t, runtimeapi.PodNetwork_POD_NETWORK_DEFAULT, podSandboxConfig.PodNetwork)
}

func TestGeneratePodSandboxConfigPodNetwork(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	// Host networking is requested through the network namespace option, as
	// before the field; pod_network only applies to POD network namespaces.
	for desc, tc := range map[string]struct {
		podNetwork       *v1.PodNetwork
		hostNetwork      bool
		expected         runtimeapi.PodNetwork
		expectedNetnsOpt runtimeapi.NamespaceMode
	}{
		"unset":       {expected: runtimeapi.PodNetwork_POD_NETWORK_DEFAULT, expectedNetnsOpt: runtimeapi.NamespaceMode_POD},
		"hostNetwork": {hostNetwork: true, expected: runtimeapi.PodNetwork_POD_NETWORK_DEFAULT, expectedNetnsOpt: runtimeapi.NamespaceMode_NODE},
		"Default":     {podNetwork: new(v1.PodNetworkDefault), expected: runtimeapi.PodNetwork_POD_NETWORK_DEFAULT, expectedNetnsOpt: runtimeapi.NamespaceMode_POD},
		"Isolated":    {podNetwork: new(v1.PodNetworkIsolated), expected: runtimeapi.PodNetwork_POD_NETWORK_ISOLATED, expectedNetnsOpt: runtimeapi.NamespaceMode_POD},
	} {
		t.Run(desc, func(t *testing.T) {
			pod := newTestPod()
			pod.Spec.PodNetwork = tc.podNetwork
			pod.Spec.HostNetwork = tc.hostNetwork
			// Namespace options are only generated with a (defaulted) security context.
			pod.Spec.SecurityContext = &v1.PodSecurityContext{}
			podSandboxConfig, err := m.generatePodSandboxConfig(tCtx, pod, 1)
			require.NoError(t, err)
			assert.Equal(t, tc.expected, podSandboxConfig.PodNetwork)
			assert.Equal(t, tc.expectedNetnsOpt, podSandboxConfig.GetLinux().GetSecurityContext().GetNamespaceOptions().GetNetwork())
		})
	}
}

// TestCreatePodSandbox tests creating sandbox and its corresponding pod log directory.
func TestCreatePodSandbox(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	pod := newTestPod()

	fakeOS := m.osInterface.(*containertest.FakeOS)
	fakeOS.MkdirAllFn = func(path string, perm os.FileMode) error {
		// Check pod logs root directory is created.
		assert.Equal(t, filepath.Join(testPodLogsDirectory, pod.Namespace+"_"+pod.Name+"_12345678"), path)
		assert.Equal(t, os.FileMode(0755), perm)
		return nil
	}
	id, _, err := m.createPodSandbox(tCtx, pod, 1)
	assert.NoError(t, err)
	assert.Contains(t, fakeRuntime.Called, "RunPodSandbox")
	sandboxes, err := fakeRuntime.ListPodSandbox(tCtx, &runtimeapi.PodSandboxFilter{Id: id})
	assert.NoError(t, err)
	assert.Len(t, sandboxes, 1)
	assert.Equal(t, sandboxes[0].Id, fmt.Sprintf("%s_%s_%s_1", pod.Name, pod.Namespace, pod.UID))
	assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_READY, sandboxes[0].State)
}

func TestGeneratePodSandboxLinuxConfigSeccomp(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
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
		config, _ := m.generatePodSandboxLinuxConfig(tCtx, test.pod)
		actualProfile := config.SecurityContext.Seccomp.ProfileType.String()
		assert.EqualValues(t, test.expectedProfile, actualProfile, "TestCase[%d]: %s", i, test.description)
	}
}

// TestCreatePodSandbox_RuntimeClass tests creating sandbox with RuntimeClasses enabled.
func TestCreatePodSandbox_RuntimeClass(t *testing.T) {
	tCtx := ktesting.Init(t)
	rcm := runtimeclass.NewManager(rctest.NewPopulatedClient())
	defer rctest.StartManagerSync(tCtx, rcm)()

	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	m.runtimeClassManager = rcm

	tests := map[string]struct {
		rcn             *string
		expectedHandler string
		expectError     bool
	}{
		"unspecified RuntimeClass": {rcn: nil, expectedHandler: ""},
		"valid RuntimeClass":       {rcn: ptr.To(rctest.SandboxRuntimeClass), expectedHandler: rctest.SandboxRuntimeHandler},
		"missing RuntimeClass":     {rcn: ptr.To("phantom"), expectError: true},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			fakeRuntime.Called = []string{}
			pod := newTestPod()
			pod.Spec.RuntimeClassName = test.rcn

			id, _, err := m.createPodSandbox(tCtx, pod, 1)
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
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
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
			pod := &v1.Pod{}
			pod.Spec = *testCase.podSpec

			wc, err := m.generatePodSandboxWindowsConfig(pod)

			assert.Equal(t, testCase.expectedWindowsConfig, wc)
			assert.Equal(t, testCase.expectedError, err)
		})
	}
}

func TestDeterminePodSandboxIPs(t *testing.T) {
	testcases := []struct {
		name        string
		nodeIP      string
		network     *runtimeapi.PodSandboxNetworkStatus
		podIPs      []string
		expectedIPs []string
	}{
		{
			name:        "Nil network",
			network:     nil,
			expectedIPs: []string{},
		},
		{
			name:        "Empty network",
			network:     &runtimeapi.PodSandboxNetworkStatus{},
			expectedIPs: []string{},
		},
		{
			name: "Unparseable primary IP",
			network: &runtimeapi.PodSandboxNetworkStatus{
				Ip: "not-an-ip",
			},
			expectedIPs: nil,
		},
		{
			name: "Unparseable additional IP",
			network: &runtimeapi.PodSandboxNetworkStatus{
				Ip:            "10.0.0.1",
				AdditionalIps: []*runtimeapi.PodIP{{Ip: "not-an-ip"}},
			},
			expectedIPs: nil,
		},
		{
			name:        "Simple",
			nodeIP:      "",
			podIPs:      []string{"10.0.0.1"},
			expectedIPs: []string{"10.0.0.1"},
		},
		{
			name:        "Dual-stack",
			nodeIP:      "",
			podIPs:      []string{"10.0.0.1", "fd01::1234"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Dual-stack with explicit node IP",
			nodeIP:      "192.168.1.1",
			podIPs:      []string{"10.0.0.1", "fd01::1234"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Dual-stack with CRI returning wrong family first",
			nodeIP:      "",
			podIPs:      []string{"fd01::1234", "10.0.0.1"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Dual-stack with explicit node IP with CRI returning wrong family first",
			nodeIP:      "192.168.1.1",
			podIPs:      []string{"fd01::1234", "10.0.0.1"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Dual-stack with IPv6 node IP",
			nodeIP:      "fd00::5678",
			podIPs:      []string{"10.0.0.1", "fd01::1234"},
			expectedIPs: []string{"fd01::1234", "10.0.0.1"},
		},
		{
			name:        "Dual-stack with IPv6 node IP, other CRI order",
			nodeIP:      "fd00::5678",
			podIPs:      []string{"fd01::1234", "10.0.0.1"},
			expectedIPs: []string{"fd01::1234", "10.0.0.1"},
		},
		{
			name:        "No Pod IP matching Node IP",
			nodeIP:      "fd00::5678",
			podIPs:      []string{"10.0.0.1"},
			expectedIPs: []string{"10.0.0.1"},
		},
		{
			name:        "No Pod IP matching (unspecified) Node IP",
			nodeIP:      "",
			podIPs:      []string{"fd01::1234"},
			expectedIPs: []string{"fd01::1234"},
		},
		{
			name:        "Multiple IPv4 IPs",
			nodeIP:      "",
			podIPs:      []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
			expectedIPs: []string{"10.0.0.1"},
		},
		{
			name:        "Multiple Dual-Stack IPs",
			nodeIP:      "",
			podIPs:      []string{"10.0.0.1", "10.0.0.2", "fd01::1234", "10.0.0.3", "fd01::5678"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Badly-formatted IPs from CRI",
			nodeIP:      "",
			podIPs:      []string{"010.000.000.001", "fd01:0:0:0:0:0:0:1234"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			_, _, m, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)
			if tc.nodeIP != "" {
				m.nodeIPs = []net.IP{netutils.ParseIPSloppy(tc.nodeIP)}
			}

			network := tc.network
			if len(tc.podIPs) > 0 {
				network = &runtimeapi.PodSandboxNetworkStatus{
					Ip: tc.podIPs[0],
				}
				for _, ip := range tc.podIPs[1:] {
					network.AdditionalIps = append(network.AdditionalIps, &runtimeapi.PodIP{Ip: ip})
				}
			}
			podSandbox := &runtimeapi.PodSandboxStatus{
				Network: network,
			}

			podIPs := m.determinePodSandboxIPs(tCtx, "test-ns", "test-pod", podSandbox)
			assert.Equal(t, tc.expectedIPs, podIPs)
		})
	}
}
