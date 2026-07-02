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
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/flowcontrol"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitest "k8s.io/cri-api/pkg/apis/testing"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/kuberuntime/util"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

const testPodLogsDirectory = "/var/log/pods"

type recordingInternalContainerLifecycle struct {
	preCreated  []string
	preStarted  map[string]string
	preStart    func(container *v1.Container, containerID string) error
	postStopped []string
	postStop    func(containerID string) error
}

func (r *recordingInternalContainerLifecycle) PreCreateContainer(_ klog.Logger, _ *v1.Pod, container *v1.Container, config *runtimeapi.ContainerConfig) error {
	r.preCreated = append(r.preCreated, container.Name)
	config.Annotations["test.kubernetes.io/pre-create"] = "called"
	return nil
}

func (r *recordingInternalContainerLifecycle) PreStartContainer(_ klog.Logger, _ *v1.Pod, container *v1.Container, containerID string) error {
	if r.preStarted == nil {
		r.preStarted = make(map[string]string)
	}
	r.preStarted[container.Name] = containerID
	if r.preStart != nil {
		return r.preStart(container, containerID)
	}
	return nil
}

func (r *recordingInternalContainerLifecycle) PostStopContainer(_ klog.Logger, containerID string) error {
	r.postStopped = append(r.postStopped, containerID)
	if r.postStop != nil {
		return r.postStop(containerID)
	}
	return nil
}

func TestRestoreCheckpointDisappearanceNeverStartsFreshSandbox(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, cachedReady := range []bool{false, true} {
		t.Run(map[bool]string{false: "live checkpoint missing", true: "ready cached checkpoint"}[cachedReady], func(t *testing.T) {
			ctx := ktesting.Init(t)
			runtime, _, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			helper := &testRuntimeHelper{FakeRuntimeHelper: m.runtimeHelper.(*containertest.FakeRuntimeHelper)}
			if !cachedReady {
				helper.getPodCheckpointPathError = &kubecontainer.RestoreError{Reason: events.CheckpointNotReady, Err: apierrors.NewNotFound(schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, "checkpoint")}
			}
			m.runtimeHelper = helper
			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			result := m.SyncPod(ctx, pod, &kubecontainer.PodStatus{ID: pod.UID}, nil, flowcontrol.NewBackOff(time.Second, time.Minute), false)
			require.NotContains(t, runtime.Called, "RunPodSandbox", "a lost checkpoint reference cannot authorize ordinary startup")
			if cachedReady {
				require.NoError(t, result.Error())
				require.Len(t, runtime.RestoredPods, 1)
			} else {
				require.Error(t, result.Error())
				require.ErrorIs(t, result.Error(), helper.getPodCheckpointPathError)
				require.Empty(t, runtime.RestoredPods)
				require.NotContains(t, runtime.Called, "CreateContainer")
				require.NotContains(t, runtime.Called, "StartContainer")
			}
		})
	}
}

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
		"missing RuntimeClass":     {rcn: new("phantom"), expectError: true},
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

func TestRestorePodSandbox_RuntimeClass(t *testing.T) {
	tCtx := ktesting.Init(t)
	rcm := runtimeclass.NewManager(rctest.NewPopulatedClient())
	defer rctest.StartManagerSync(ktesting.Init(t), rcm)()

	tests := map[string]struct {
		runtimeClassName *string
		expectedHandler  string
		hostNetwork      bool
		expectError      bool
	}{
		"unspecified RuntimeClass":        {expectedHandler: ""},
		"valid RuntimeClass":              {runtimeClassName: ptr.To(rctest.SandboxRuntimeClass), expectedHandler: rctest.SandboxRuntimeHandler},
		"host network with runtime class": {runtimeClassName: ptr.To(rctest.SandboxRuntimeClass), expectedHandler: rctest.SandboxRuntimeHandler, hostNetwork: true},
		"missing RuntimeClass":            {runtimeClassName: new("phantom"), expectError: true},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if test.hostNetwork && runtime.GOOS == "windows" {
				t.Skip("host-network namespace status requires Linux")
			}
			fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)
			m.runtimeClassManager = rcm
			pod := newTestPod()
			pod.Spec.HostNetwork = test.hostNetwork
			pod.Spec.SecurityContext = &v1.PodSecurityContext{}
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			pod.Spec.RuntimeClassName = test.runtimeClassName

			restored, _, err := m.restorePodSandbox(tCtx, pod, 1, nil)
			if test.expectError {
				require.Error(t, err)
				assert.Empty(t, fakeRuntime.RestoredPods)
				return
			}
			require.NoError(t, err)
			require.Len(t, fakeRuntime.RestoredPods, 1)
			assert.Equal(t, test.expectedHandler, fakeRuntime.RestoredPods[0].RuntimeHandler)
			assert.Empty(t, fakeRuntime.RestoredPods[0].Options)
			response, err := fakeRuntime.PodSandboxStatus(tCtx, restored.id, false)
			require.NoError(t, err)
			assert.Equal(t, test.expectedHandler, response.Status.RuntimeHandler)
			assert.Equal(t, util.NetworkNamespaceForPod(pod), response.Status.GetLinux().GetNamespaces().GetOptions().GetNetwork())
			changed, attempt, sandboxID, reason := util.PodSandboxChanged(klog.FromContext(tCtx), pod, &kubecontainer.PodStatus{SandboxStatuses: []*runtimeapi.PodSandboxStatus{response.Status}})
			assert.False(t, changed, reason)
			assert.Equal(t, uint32(1), attempt)
			assert.Equal(t, restored.id, sandboxID)
		})
	}
}

// TestRestorePodSandboxPidNamespaceFromSpec verifies that restorePodSandbox no
// longer forces pod-level PID namespace sharing and instead derives the PID
// namespace mode from the pod spec, matching the normal createPodSandbox path.
func TestRestorePodSandboxPidNamespaceFromSpec(t *testing.T) {
	tCtx := ktesting.Init(t)

	tests := []struct {
		name     string
		mutate   func(*v1.Pod)
		expected runtimeapi.NamespaceMode
	}{
		{
			name:     "default keeps per-container PID namespace",
			mutate:   func(*v1.Pod) {},
			expected: runtimeapi.NamespaceMode_CONTAINER,
		},
		{
			name:     "ShareProcessNamespace shares the pod PID namespace",
			mutate:   func(p *v1.Pod) { p.Spec.ShareProcessNamespace = new(true) },
			expected: runtimeapi.NamespaceMode_POD,
		},
		{
			name:     "HostPID uses the node PID namespace",
			mutate:   func(p *v1.Pod) { p.Spec.HostPID = true },
			expected: runtimeapi.NamespaceMode_NODE,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)

			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			// NamespaceOptions are only derived when a PodSecurityContext is
			// present (true on both the create and restore paths), so set one.
			pod.Spec.SecurityContext = &v1.PodSecurityContext{}
			tc.mutate(pod)

			_, _, err = m.restorePodSandbox(tCtx, pod, 0, nil)
			require.NoError(t, err)

			require.Len(t, fakeRuntime.RestoredPods, 1)
			cfg := fakeRuntime.RestoredPods[0].Config
			require.NotNil(t, cfg)
			require.NotNil(t, cfg.Linux)
			require.NotNil(t, cfg.Linux.SecurityContext)
			require.NotNil(t, cfg.Linux.SecurityContext.NamespaceOptions)
			assert.Equal(t, tc.expected, cfg.Linux.SecurityContext.NamespaceOptions.Pid)
		})
	}
}

// TestRestorePodSandboxSelectsRestorableContainers verifies that only
// long-running application containers are included in the restore request.
func TestRestorePodSandboxSelectsRestorableContainers(t *testing.T) {
	const pinnedImage = "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ImageVolume, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, fakeImage, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	expectedStartedContainers := 0
	lifecycle := &recordingInternalContainerLifecycle{preStart: func(_ *v1.Container, containerID string) error {
		assert.Equal(t, runtimeapi.ContainerState_CONTAINER_CREATED, fakeRuntime.Containers[containerID].State)
		startedContainers := 0
		for _, call := range fakeRuntime.GetCalls() {
			if call == "StartContainer" {
				startedContainers++
			}
		}
		assert.Equal(t, expectedStartedContainers, startedContainers, "PreStartContainer must run before its StartContainer call")
		expectedStartedContainers++
		return nil
	}}
	m.internalLifecycle = lifecycle

	pod := newTestPod()
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
	for i := range pod.Spec.Containers {
		pod.Spec.Containers[i].Image = pinnedImage
	}
	pod.Spec.Volumes = []v1.Volume{{
		Name: "image-data",
		VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{
			Reference:  "image-volume:latest",
			PullPolicy: v1.PullAlways,
		}},
	}}
	pod.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{{Name: "image-data", MountPath: "/data"}}
	pod.Spec.InitContainers = []v1.Container{
		{Name: "completed-init", Image: "busybox"},
		{Name: "sidecar", Image: pinnedImage, RestartPolicy: ptr.To(v1.ContainerRestartPolicyAlways)},
	}
	pod.Spec.EphemeralContainers = []v1.EphemeralContainer{
		{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debugger"}},
	}

	_, _, err = m.restorePodSandbox(tCtx, pod, 0, nil)
	require.NoError(t, err)

	require.Len(t, fakeRuntime.RestoredPods, 1)
	var names []string
	for _, cc := range fakeRuntime.RestoredPods[0].ContainerConfigs {
		names = append(names, cc.Metadata.Name)
		require.NotNil(t, cc.Image)
		assert.Equal(t, pinnedImage, cc.Image.Image, "CRI restore must receive the captured digest")
		assert.NotEmpty(t, cc.LogPath)
		assert.NotEmpty(t, cc.Labels)
		assert.Equal(t, "called", cc.Annotations["test.kubernetes.io/pre-create"])
	}
	assert.Contains(t, names, "foo") // regular container from newTestPod
	assert.Contains(t, names, "sidecar")
	assert.NotContains(t, names, "completed-init")
	assert.NotContains(t, names, "debugger")
	assert.ElementsMatch(t, []string{"sidecar", "foo"}, lifecycle.preCreated)
	expectedStarted := map[string]string{}
	request := fakeRuntime.RestoredPods[0]
	sandboxID := apitest.BuildSandboxName(request.Config.Metadata)
	for _, config := range request.ContainerConfigs {
		expectedStarted[config.Metadata.Name] = apitest.BuildContainerName(config.Metadata, sandboxID)
	}
	assert.Equal(t, expectedStarted, lifecycle.preStarted)
	assert.Equal(t, 2, expectedStartedContainers)
	runtimeHelper := m.runtimeHelper.(*containertest.FakeRuntimeHelper)
	imageVolume := runtimeHelper.ContainerImageVolumes[pod.UID]["foo"]["image-data"]
	require.NotNil(t, imageVolume)
	assert.Equal(t, "image-volume:latest", imageVolume.UserSpecifiedImage)
	assert.Contains(t, fakeImage.Called, "PullImage")
}

type restoredSandboxStatusRuntime struct {
	internalapi.RuntimeService
	response *runtimeapi.PodSandboxStatusResponse
}

func (r *restoredSandboxStatusRuntime) PodSandboxStatus(ctx context.Context, sandboxID string, verbose bool) (*runtimeapi.PodSandboxStatusResponse, error) {
	if _, err := r.RuntimeService.PodSandboxStatus(ctx, sandboxID, verbose); err != nil {
		return nil, err
	}
	return r.response, nil
}

func TestRestorePodSandboxPreparesHostsBeforeStart(t *testing.T) {
	status := func(primary string, additional ...string) *runtimeapi.PodSandboxStatusResponse {
		result := &runtimeapi.PodSandboxStatusResponse{Status: &runtimeapi.PodSandboxStatus{
			State:   runtimeapi.PodSandboxState_SANDBOX_READY,
			Network: &runtimeapi.PodSandboxNetworkStatus{Ip: primary},
		}}
		for _, ip := range additional {
			result.Status.Network.AdditionalIps = append(result.Status.Network.AdditionalIps, &runtimeapi.PodIP{Ip: ip})
		}
		return result
	}
	for _, tc := range []struct {
		name        string
		response    *runtimeapi.PodSandboxStatusResponse
		hostNetwork bool
		wantIPs     []string
		wantErr     string
	}{
		{name: "IPv4", response: status("192.0.2.1"), wantIPs: []string{"192.0.2.1"}},
		{name: "IPv6", response: status("2001:db8::1"), wantIPs: []string{"2001:db8::1"}},
		{name: "dual stack", response: status("192.0.2.1", "2001:db8::1"), wantIPs: []string{"192.0.2.1", "2001:db8::1"}},
		{name: "host network", response: status(""), hostNetwork: true},
		{name: "missing IP", response: status(""), wantErr: "no Pod IP"},
		{name: "invalid primary IP", response: status("invalid"), wantErr: "no Pod IP"},
		{name: "invalid additional IP", response: status("192.0.2.1", "invalid"), wantErr: "no Pod IP"},
		{name: "sandbox not ready", response: &runtimeapi.PodSandboxStatusResponse{Status: &runtimeapi.PodSandboxStatus{State: runtimeapi.PodSandboxState_SANDBOX_NOTREADY}}, wantErr: "not ready"},
		{name: "nil status", response: &runtimeapi.PodSandboxStatusResponse{}, wantErr: "status is nil"},
		{name: "nil response", wantErr: "status is nil"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := ktesting.Init(t)
			fakeRuntime, _, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			m.runtimeService = &restoredSandboxStatusRuntime{RuntimeService: m.runtimeService, response: tc.response}
			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			pod.Spec.HostNetwork = tc.hostNetwork
			pod.Spec.InitContainers = []v1.Container{{Name: "sidecar", Image: "busybox", RestartPolicy: new(v1.ContainerRestartPolicyAlways)}}
			helper := m.runtimeHelper.(*containertest.FakeRuntimeHelper)
			lifecycle := &recordingInternalContainerLifecycle{preStart: func(_ *v1.Container, containerID string) error {
				ips, prepared := helper.PodRestoreIPs[pod.UID]
				assert.True(t, prepared, "hosts preparation must finish before the first start hook")
				assert.Equal(t, tc.wantIPs, ips)
				assert.Equal(t, runtimeapi.ContainerState_CONTAINER_CREATED, fakeRuntime.Containers[containerID].State)
				return nil
			}}
			m.internalLifecycle = lifecycle

			restored, _, err := m.restorePodSandbox(ctx, pod, 0, nil)
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				assert.Nil(t, restored)
				assert.Empty(t, lifecycle.preStarted)
				assert.NotContains(t, fakeRuntime.GetCalls(), "StartContainer")
				assert.Empty(t, lifecycle.postStopped, "Pod termination stops the restored containers")
				assert.NotEmpty(t, fakeRuntime.Sandboxes, "Pod termination removes the restored sandbox")
				return
			}
			require.NoError(t, err)
			require.NotNil(t, restored)
			assert.Equal(t, tc.response.Status, restored.status)
			assert.Len(t, lifecycle.preStarted, 2)
			for _, container := range fakeRuntime.Containers {
				assert.Equal(t, runtimeapi.ContainerState_CONTAINER_RUNNING, container.State)
			}
		})
	}
}

func TestRestorePodSandboxRecordsStartBeforeRuntime(t *testing.T) {
	for _, markErr := range []error{nil, errors.New("marker write failed")} {
		t.Run(fmt.Sprintf("mark-fails=%t", markErr != nil), func(t *testing.T) {
			tCtx := ktesting.Init(t)
			fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)
			helper := m.runtimeHelper.(*containertest.FakeRuntimeHelper)
			helper.PodRestoreStartedErr = markErr
			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}

			_, _, err = m.restorePodSandbox(tCtx, pod, 0, nil)
			if markErr != nil {
				require.ErrorIs(t, err, markErr)
				require.Equal(t, events.RestorePreparationFailed, kubecontainer.RestoreErrorReason(err), "a restore without a start record must be retried")
				require.Empty(t, fakeRuntime.RestoredPods, "RestorePod must not run without a start record")
				return
			}
			require.NoError(t, err)
			require.Len(t, fakeRuntime.RestoredPods, 1)
			require.True(t, helper.PodRestoreStarted[pod.UID])
		})
	}
}

func TestRestorePodSandboxStopsWhenPreStartFails(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	pod := newTestPod()
	lifecycle := &recordingInternalContainerLifecycle{preStart: func(container *v1.Container, id string) error {
		return fmt.Errorf("reject %s", container.Name)
	}}
	m.internalLifecycle = lifecycle
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}

	_, _, err = m.restorePodSandbox(tCtx, pod, 0, nil)
	require.ErrorContains(t, err, "reject")
	assert.NotContains(t, fakeRuntime.GetCalls(), "StartContainer")
	require.Len(t, fakeRuntime.RestoredPods, 1)
	assert.NotEmpty(t, fakeRuntime.Sandboxes, "Pod termination removes the restored sandbox")
}

func TestRestorePodSandboxStopsWhenStartFails(t *testing.T) {
	for _, started := range []int{0, 1} {
		t.Run(fmt.Sprintf("started=%d", started), func(t *testing.T) {
			ctx := ktesting.Init(t)
			fakeRuntime, _, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			lifecycle := &recordingInternalContainerLifecycle{}
			m.internalLifecycle = lifecycle
			for range started {
				fakeRuntime.InjectError("StartContainer", nil)
			}
			failure := errors.New("start failed")
			fakeRuntime.InjectError("StartContainer", failure)
			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{Name: "second", Image: "busybox"})

			_, _, err = m.restorePodSandbox(ctx, pod, 0, nil)
			require.ErrorIs(t, err, failure)
			require.Len(t, lifecycle.preStarted, started+1)
			require.Len(t, fakeRuntime.RestoredPods, 1)
			request := fakeRuntime.RestoredPods[0]
			require.Len(t, request.ContainerConfigs, 2)
			assert.Empty(t, lifecycle.postStopped, "Pod termination stops the restored containers")
			assert.NotEmpty(t, fakeRuntime.Sandboxes, "Pod termination removes the restored sandbox")
		})
	}
}

type malformedRestoreResponseRuntime struct {
	internalapi.RuntimeService
	otherContainerID string
}

func (r *malformedRestoreResponseRuntime) RestorePod(ctx context.Context, request *runtimeapi.RestorePodRequest) (*runtimeapi.RestorePodResponse, error) {
	response, err := r.RuntimeService.RestorePod(ctx, request)
	if err == nil {
		response.RestoredContainers[0].Name = "unexpected"
		response.RestoredContainers[0].ContainerId = r.otherContainerID
	}
	return response, err
}

func TestRestorePodSandboxLeavesOtherContainersAfterInvalidResponse(t *testing.T) {
	ctx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(ctx)
	require.NoError(t, err)
	otherID, err := fakeRuntime.CreateContainer(ctx, "other-sandbox", &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{Name: "other"},
		Image:    &runtimeapi.ImageSpec{Image: "busybox"},
	}, nil)
	require.NoError(t, err)
	m.runtimeService = &malformedRestoreResponseRuntime{RuntimeService: m.runtimeService, otherContainerID: otherID}
	lifecycle := &recordingInternalContainerLifecycle{}
	m.internalLifecycle = lifecycle
	pod := newTestPod()
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}

	_, _, err = m.restorePodSandbox(ctx, pod, 0, nil)
	require.ErrorContains(t, err, "unexpected name")
	assert.Empty(t, lifecycle.preStarted)
	require.Len(t, fakeRuntime.RestoredPods, 1)
	require.Len(t, fakeRuntime.RestoredPods[0].ContainerConfigs, 1)
	assert.Empty(t, lifecycle.postStopped)
	assert.Contains(t, fakeRuntime.Containers, otherID)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_CREATED, fakeRuntime.Containers[otherID].State)
}

func TestRestoreContainerConfigUsesImageUser(t *testing.T) {
	tests := []struct {
		name      string
		runAsUser *int64
		wantErr   string
	}{
		{name: "image root is rejected", wantErr: "image will run as root"},
		{name: "explicit root is rejected", runAsUser: ptr.To[int64](0), wantErr: "runAsUser breaks non-root policy"},
		{name: "explicit non-root is preserved", runAsUser: ptr.To[int64](1000)},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if runtime.GOOS == "windows" && tc.runAsUser != nil {
				t.Skip("runAsUser non-root policy enforcement and config.GetLinux() are Linux-only")
			}
			tCtx := ktesting.Init(t)
			_, fakeImage, m, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)
			pod := newTestPod()
			pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
				RunAsNonRoot: new(true),
				RunAsUser:    tc.runAsUser,
			}

			config, cleanup, err := m.generateContainerConfig(tCtx, &pod.Spec.Containers[0], pod, 0, "", pod.Spec.Containers[0].Image, nil, nil, nil)
			if cleanup != nil {
				defer cleanup()
			}
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
				require.NotNil(t, config)
				if tc.runAsUser != nil {
					require.Equal(t, *tc.runAsUser, config.GetLinux().GetSecurityContext().GetRunAsUser().GetValue())
				}
			}
			assert.Contains(t, fakeImage.Called, "ImageStatus")
		})
	}
}

func TestValidateRestorePodResponse(t *testing.T) {
	request := &runtimeapi.RestorePodRequest{ContainerConfigs: []*runtimeapi.ContainerConfig{
		{Metadata: &runtimeapi.ContainerMetadata{Name: "sidecar"}},
		{Metadata: &runtimeapi.ContainerMetadata{Name: "app"}},
	}}
	valid := func() *runtimeapi.RestorePodResponse {
		return &runtimeapi.RestorePodResponse{
			PodSandboxId: "sandbox-id",
			RestoredContainers: []*runtimeapi.RestoredContainer{
				{Name: "sidecar", ContainerId: "sidecar-id"},
				{Name: "app", ContainerId: "app-id"},
			},
		}
	}

	tests := []struct {
		name     string
		response func() *runtimeapi.RestorePodResponse
		wantErr  string
	}{
		{name: "valid", response: valid},
		{name: "nil response", response: func() *runtimeapi.RestorePodResponse { return nil }, wantErr: "response is nil"},
		{name: "empty sandbox ID", response: func() *runtimeapi.RestorePodResponse { r := valid(); r.PodSandboxId = ""; return r }, wantErr: "pod sandbox ID is empty"},
		{name: "missing container", response: func() *runtimeapi.RestorePodResponse {
			r := valid()
			r.RestoredContainers = r.RestoredContainers[:1]
			return r
		}, wantErr: "count"},
		{name: "nil container", response: func() *runtimeapi.RestorePodResponse { r := valid(); r.RestoredContainers[1] = nil; return r }, wantErr: "is nil"},
		{name: "unexpected name", response: func() *runtimeapi.RestorePodResponse { r := valid(); r.RestoredContainers[1].Name = "other"; return r }, wantErr: "unexpected name"},
		{name: "duplicate name", response: func() *runtimeapi.RestorePodResponse {
			r := valid()
			r.RestoredContainers[1].Name = "sidecar"
			return r
		}, wantErr: "duplicate name"},
		{name: "empty container ID", response: func() *runtimeapi.RestorePodResponse {
			r := valid()
			r.RestoredContainers[1].ContainerId = ""
			return r
		}, wantErr: "empty container ID"},
		{name: "duplicate container ID", response: func() *runtimeapi.RestorePodResponse {
			r := valid()
			r.RestoredContainers[1].ContainerId = "sidecar-id"
			return r
		}, wantErr: "duplicate container ID"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := validateRestorePodResponse(request, tc.response())
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, map[string]string{"sidecar": "sidecar-id", "app": "app-id"}, ids)
		})
	}
}

// TestAcquireReleaseRestore verifies the per-pod in-flight restore guard:
// acquiring a key succeeds once, blocks a second acquire of the same key, is
// independent across keys, and can be re-acquired after release. The key is the
// pod's namespace/name (not UID), since each restore attempt is admitted under
// a fresh pod UID.
func TestAcquireReleaseRestore(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	const keyA, keyB = "ns/pod-a", "ns/pod-b"

	assert.True(t, m.acquireRestore(keyA), "first acquire should succeed")
	assert.False(t, m.acquireRestore(keyA), "second acquire of the same key should be rejected")
	assert.True(t, m.acquireRestore(keyB), "a different key is independent")

	m.releaseRestore(keyA)
	assert.True(t, m.acquireRestore(keyA), "acquire after release should succeed")
}

// TestRestorePodSandboxRejectsConcurrentRestore verifies that restorePodSandbox
// rejects a restore while one is already in flight for the same pod UID without
// calling the runtime, and that the guard is released after a restore finishes.
func TestRestorePodSandboxRejectsConcurrentRestore(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newTestPod()
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint-1"}
	restoreKey := pod.Namespace + "/" + pod.Name

	// Simulate a restore already running for this pod.
	require.True(t, m.acquireRestore(restoreKey))

	// A second restore for the same pod is rejected without reaching the runtime.
	_, msg, err := m.restorePodSandbox(tCtx, pod, 0, nil)
	require.Error(t, err)
	assert.Contains(t, msg, "already in progress")
	assert.Empty(t, fakeRuntime.RestoredPods)

	// Once the in-flight restore completes, a new restore proceeds.
	m.releaseRestore(restoreKey)
	_, _, err = m.restorePodSandbox(tCtx, pod, 0, nil)
	require.NoError(t, err)
	require.Len(t, fakeRuntime.RestoredPods, 1)

	// A successful restore releases the guard, so the pod can be acquired again.
	assert.True(t, m.acquireRestore(restoreKey), "guard should be released after a successful restore")
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
func TestRestoreContainerImagesAndCredentials(t *testing.T) {
	for _, uid := range []int64{0, 1000} {
		t.Run(fmt.Sprintf("uid=%d", uid), func(t *testing.T) {
			ctx := ktesting.Init(t)
			_, images, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			pod := newTestPod()
			pod.Spec.InitContainers = nil
			pod.Spec.Containers = pod.Spec.Containers[:1]
			container := &pod.Spec.Containers[0]
			container.Image = "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
			container.ImagePullPolicy = v1.PullIfNotPresent
			container.SecurityContext = nil
			pod.Spec.SecurityContext = nil
			config, err := m.generatePodSandboxConfig(ctx, pod, 0)
			require.NoError(t, err)
			require.NoError(t, m.pullRestoreContainerImages(ctx, pod, config, "", nil))
			assert.Contains(t, images.Called, "PullImage", "missing application images must be pulled before RestorePod")
			images.Images[container.Image].Uid = &runtimeapi.Int64Value{Value: uid}
			configs, cleanup, err := m.generateContainerConfigsForRestore(ctx, pod, config, nil)
			require.NoError(t, err)
			defer cleanup()
			require.Len(t, configs, 1)
			require.Equal(t, &runtimeapi.Int64Value{Value: uid}, configs[0].GetLinux().GetSecurityContext().RunAsUser)
			images.Called = nil
			require.NoError(t, m.pullRestoreContainerImages(ctx, pod, config, "", nil))
			assert.NotContains(t, images.Called, "PullImage", "IfNotPresent must reuse the resolved image")
		})
	}
}

func TestRestorePodSandboxPassesDestinationSecurityContext(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux process security context")
	}
	for _, override := range []bool{false, true} {
		t.Run(fmt.Sprintf("container-overrides=%t", override), func(t *testing.T) {
			ctx := ktesting.Init(t)
			runtimeService, _, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			pod.Spec.SecurityContext = &v1.PodSecurityContext{
				RunAsUser: ptr.To[int64](1000), RunAsGroup: ptr.To[int64](2000),
				SupplementalGroups: []int64{3000},
				SELinuxOptions:     &v1.SELinuxOptions{User: "system_u", Role: "system_r", Type: "container_t", Level: "s0:c1,c2"},
				SeccompProfile:     &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault},
				AppArmorProfile:    &v1.AppArmorProfile{Type: v1.AppArmorProfileTypeRuntimeDefault},
			}
			pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
				AllowPrivilegeEscalation: new(false), ReadOnlyRootFilesystem: new(true),
				Capabilities: &v1.Capabilities{Drop: []v1.Capability{"ALL"}},
			}
			wantUID, wantGID, wantLevel := int64(1000), int64(2000), "s0:c1,c2"
			if override {
				security := pod.Spec.Containers[0].SecurityContext
				security.RunAsUser, security.RunAsGroup = ptr.To[int64](1001), ptr.To[int64](2001)
				security.SELinuxOptions = &v1.SELinuxOptions{Level: "s0:c3,c4"}
				security.SeccompProfile = &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: new("restore.json")}
				security.AppArmorProfile = &v1.AppArmorProfile{Type: v1.AppArmorProfileTypeLocalhost, LocalhostProfile: new("restore-profile")}
				wantUID, wantGID, wantLevel = 1001, 2001, "s0:c3,c4"
			}
			_, _, err = m.restorePodSandbox(ctx, pod, 0, nil)
			require.NoError(t, err)
			require.Len(t, runtimeService.RestoredPods, 1)
			request := runtimeService.RestoredPods[0]
			require.Len(t, request.ContainerConfigs, 1)
			security := request.ContainerConfigs[0].GetLinux().GetSecurityContext()
			require.NotNil(t, security)
			require.Equal(t, wantUID, security.GetRunAsUser().GetValue())
			require.Equal(t, wantGID, security.GetRunAsGroup().GetValue())
			require.Equal(t, []int64{3000}, security.GetSupplementalGroups())
			require.Equal(t, []string{"ALL"}, security.GetCapabilities().GetDropCapabilities())
			require.Empty(t, security.GetCapabilities().GetAddCapabilities())
			require.True(t, security.GetNoNewPrivs())
			require.True(t, security.GetReadonlyRootfs())
			require.False(t, security.GetPrivileged())
			require.Equal(t, wantLevel, security.GetSelinuxOptions().GetLevel())
			if override {
				require.Equal(t, runtimeapi.SecurityProfile_Localhost, security.GetSeccomp().GetProfileType())
				require.Equal(t, filepath.Join(m.seccompProfileRoot, "restore.json"), security.GetSeccomp().GetLocalhostRef())
				require.Equal(t, runtimeapi.SecurityProfile_Localhost, security.GetApparmor().GetProfileType())
				require.Equal(t, "restore-profile", security.GetApparmor().GetLocalhostRef())
			} else {
				require.Equal(t, runtimeapi.SecurityProfile_RuntimeDefault, security.GetSeccomp().GetProfileType())
				require.Equal(t, runtimeapi.SecurityProfile_RuntimeDefault, security.GetApparmor().GetProfileType())
			}
		})
	}
}

func TestRestorePodSandboxLogSymlinks(t *testing.T) {
	for _, tc := range []struct {
		name       string
		statErr    error
		symlinkErr error
		wantLinks  bool
	}{
		{name: "existing logs", wantLinks: true},
		{name: "missing logs", statErr: os.ErrNotExist},
		{name: "stat error", statErr: os.ErrPermission, wantLinks: true},
		{name: "symlink error", symlinkErr: os.ErrPermission, wantLinks: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := ktesting.Init(t)
			runtime, _, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			links := map[string]string{}
			osInterface := m.osInterface.(*containertest.FakeOS)
			osInterface.StatFn = func(string) (os.FileInfo, error) { return nil, tc.statErr }
			osInterface.SymlinkFn = func(target, link string) error {
				links[link] = target
				return tc.symlinkErr
			}
			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			pod.Spec.InitContainers = []v1.Container{{Name: "sidecar", Image: "busybox", RestartPolicy: &containerRestartPolicyAlways}}
			_, _, err = m.restorePodSandbox(ctx, pod, 0, nil)
			require.NoError(t, err)
			if !tc.wantLinks {
				require.Empty(t, links)
				return
			}
			require.Len(t, runtime.RestoredPods, 1)
			request := runtime.RestoredPods[0]
			require.Len(t, request.ContainerConfigs, 2)
			sandboxID := apitest.BuildSandboxName(request.Config.Metadata)
			expected := map[string]string{}
			for _, config := range request.ContainerConfigs {
				name := config.Metadata.Name
				id := apitest.BuildContainerName(config.Metadata, sandboxID)
				expected[legacyLogSymlink(id, name, pod.Name, pod.Namespace)] = filepath.Join(testPodLogsDirectory, pod.Namespace+"_"+pod.Name+"_"+string(pod.UID), name, "0.log")
				require.Equal(t, runtimeapi.ContainerState_CONTAINER_RUNNING, runtime.Containers[id].State)
			}
			require.Equal(t, expected, links)
		})
	}
}

func TestRestorePodSandboxIgnoresUnverifiedPodIPs(t *testing.T) {
	ctx := ktesting.Init(t)
	runtime, _, m, err := createTestRuntimeManager(ctx)
	require.NoError(t, err)
	helper := &testRuntimeHelper{FakeRuntimeHelper: m.runtimeHelper.(*containertest.FakeRuntimeHelper)}
	m.runtimeHelper = helper
	pod := newTestPod()
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
	pod.Labels = map[string]string{"io.kubernetes.pod.ip": "203.0.113.10"}
	pod.Status.PodIP = "203.0.113.11"
	pod.Status.PodIPs = []v1.PodIP{{IP: "203.0.113.11"}, {IP: "2001:db8::11"}}

	_, _, err = m.restorePodSandbox(ctx, pod, 0, nil)
	require.NoError(t, err)
	require.Empty(t, helper.runContainerPodIP)
	require.Empty(t, helper.runContainerPodIPs)
	require.Len(t, runtime.RestoredPods, 1)
	require.Equal(t, pod.Labels["io.kubernetes.pod.ip"], runtime.RestoredPods[0].Config.Labels["io.kubernetes.pod.ip"])
}

// A runtime without RestorePod starts no work, so the restore fails without
// keeping the restore lock.
func TestRestorePodSandboxUnimplementedDoesNotStart(t *testing.T) {
	ctx := ktesting.Init(t)
	runtime, _, m, err := createTestRuntimeManager(ctx)
	require.NoError(t, err)
	pod := newTestPod()
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
	runtime.InjectError("RestorePod", grpcstatus.Error(codes.Unimplemented, "unknown method RestorePod"))
	_, _, err = m.restorePodSandbox(ctx, pod, 0, nil)
	require.ErrorContains(t, err, "runtime does not support Pod restore")
	require.Equal(t, events.RestoreFailed, kubecontainer.RestoreErrorReason(err))
	require.NotContains(t, runtime.Called, "StartContainer")
	key := pod.Namespace + "/" + pod.Name
	require.True(t, m.acquireRestore(key), "a restore that started no runtime work must release the lock")
	m.releaseRestore(key)
}

// As with createPodSandbox, a restore that fails after the Pod was deleted is
// not reported, because Pod termination removes what it created.
func TestRestorePodSandboxForSyncIgnoresFailureAfterPodDeletion(t *testing.T) {
	tCtx := ktesting.Init(t)
	runtime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	pod := newTestPod()
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
	m.podStateProvider.(*fakePodStateProvider).removed[pod.UID] = struct{}{}
	runtime.InjectError("RestorePod", context.Canceled)

	result := kubecontainer.NewSyncResult(kubecontainer.RestorePodSandbox, pod.Name)
	_, _, started := m.restorePodSandboxForSync(tCtx, pod, 0, nil, result)
	require.False(t, started)
	require.NoError(t, result.Error)
	key := pod.Namespace + "/" + pod.Name
	require.True(t, m.acquireRestore(key), "a failed restore must release the restore lock")
	m.releaseRestore(key)
}

type cancelAfterRestoreResponseRuntime struct {
	internalapi.RuntimeService
	cancel context.CancelFunc
}

func (r *cancelAfterRestoreResponseRuntime) RestorePod(ctx context.Context, request *runtimeapi.RestorePodRequest) (*runtimeapi.RestorePodResponse, error) {
	response, err := r.RuntimeService.RestorePod(ctx, request)
	r.cancel()
	return response, err
}

func TestRestorePodSandboxDoesNotStartContainersAfterCancellation(t *testing.T) {
	tCtx := ktesting.Init(t)
	runtime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()
	m.runtimeService = &cancelAfterRestoreResponseRuntime{RuntimeService: m.runtimeService, cancel: cancel}
	pod := newTestPod()
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
	_, _, err = m.restorePodSandbox(ctx, pod, 0, nil)
	require.ErrorIs(t, err, context.Canceled)
	require.Len(t, runtime.RestoredPods, 1)
	require.NotContains(t, runtime.Called, "StartContainer")
}

type restoreStartBoundaryRuntime struct {
	internalapi.RuntimeService
	restoreContext context.Context
	startContexts  []context.Context
	startedIDs     []string
	beforeStart    func(context.Context) error
	afterStart     func() error
}

func (r *restoreStartBoundaryRuntime) RestorePod(ctx context.Context, request *runtimeapi.RestorePodRequest) (*runtimeapi.RestorePodResponse, error) {
	r.restoreContext = ctx
	return r.RuntimeService.RestorePod(ctx, request)
}

func (r *restoreStartBoundaryRuntime) StartContainer(ctx context.Context, containerID string) error {
	r.startedIDs = append(r.startedIDs, containerID)
	if r.beforeStart != nil {
		if err := r.beforeStart(ctx); err != nil {
			return err
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	// Model the CRI client's per-call context without cancelling the Pod worker.
	startCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	r.startContexts = append(r.startContexts, startCtx)
	if err := r.RuntimeService.StartContainer(startCtx, containerID); err != nil {
		return err
	}
	if r.afterStart != nil {
		return r.afterStart()
	}
	return nil
}
