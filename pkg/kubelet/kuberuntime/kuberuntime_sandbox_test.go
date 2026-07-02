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
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

const testPodLogsDirectory = "/var/log/pods"

type recordingInternalContainerLifecycle struct {
	preCreated []string
	preStarted map[string]string
	preStart   func(container *v1.Container, containerID string) error
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

func (*recordingInternalContainerLifecycle) PostStopContainer(klog.Logger, string) error {
	return nil
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
	defer rctest.StartManagerSync(rcm)()

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
	defer rctest.StartManagerSync(rcm)()

	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	m.runtimeClassManager = rcm

	tests := map[string]struct {
		runtimeClassName *string
		expectedHandler  string
		expectError      bool
	}{
		"unspecified RuntimeClass": {expectedHandler: ""},
		"valid RuntimeClass":       {runtimeClassName: ptr.To(rctest.SandboxRuntimeClass), expectedHandler: rctest.SandboxRuntimeHandler},
		"missing RuntimeClass":     {runtimeClassName: new("phantom"), expectError: true},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			fakeRuntime.RestoredPods = nil
			pod := newTestPod()
			pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}
			pod.Spec.RuntimeClassName = test.runtimeClassName

			_, _, err := m.restorePodSandbox(tCtx, pod, 1, nil)
			if test.expectError {
				require.Error(t, err)
				assert.Empty(t, fakeRuntime.RestoredPods)
				return
			}
			require.NoError(t, err)
			require.Len(t, fakeRuntime.RestoredPods, 1)
			assert.Equal(t, test.expectedHandler, fakeRuntime.RestoredPods[0].RuntimeHandler)
			assert.Empty(t, fakeRuntime.RestoredPods[0].Options)
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ImageVolume, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, fakeImage, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	expectedStartedContainers := 0
	lifecycle := &recordingInternalContainerLifecycle{preStart: func(_ *v1.Container, _ string) error {
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
		{Name: "sidecar", Image: "busybox", RestartPolicy: ptr.To(v1.ContainerRestartPolicyAlways)},
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
		assert.Equal(t, "busybox", cc.Image.Image)
		assert.NotEmpty(t, cc.LogPath)
		assert.NotEmpty(t, cc.Labels)
		assert.Equal(t, "called", cc.Annotations["test.kubernetes.io/pre-create"])
	}
	assert.Contains(t, names, "foo") // regular container from newTestPod
	assert.Contains(t, names, "sidecar")
	assert.NotContains(t, names, "completed-init")
	assert.NotContains(t, names, "debugger")
	assert.ElementsMatch(t, []string{"sidecar", "foo"}, lifecycle.preCreated)
	assert.Equal(t, map[string]string{
		"sidecar": "fake-restored-container-sidecar",
		"foo":     "fake-restored-container-foo",
	}, lifecycle.preStarted)
	assert.Equal(t, 2, expectedStartedContainers)
	runtimeHelper := m.runtimeHelper.(*containertest.FakeRuntimeHelper)
	imageVolume := runtimeHelper.ContainerImageVolumes["foo"]["image-data"]
	require.NotNil(t, imageVolume)
	assert.Equal(t, "image-volume:latest", imageVolume.UserSpecifiedImage)
	assert.Contains(t, fakeImage.Called, "PullImage")
}

func TestRestorePodSandboxCleansUpWhenPreStartFails(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	m.internalLifecycle = &recordingInternalContainerLifecycle{preStart: func(container *v1.Container, _ string) error {
		return fmt.Errorf("reject %s", container.Name)
	}}
	pod := newTestPod()
	pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: "checkpoint"}

	_, _, err = m.restorePodSandbox(tCtx, pod, 0, nil)
	require.ErrorContains(t, err, "reject")
	assert.Contains(t, fakeRuntime.GetCalls(), "StopPodSandbox")
	assert.Contains(t, fakeRuntime.GetCalls(), "RemovePodSandbox")
	assert.NotContains(t, fakeRuntime.GetCalls(), "StartContainer")
	assert.Empty(t, fakeRuntime.Sandboxes)
	assert.Empty(t, fakeRuntime.Containers)
}

func TestGenerateContainerConfigForRestoreDoesNotInspectImageUser(t *testing.T) {
	tests := []struct {
		name      string
		runAsUser *int64
		wantErr   string
	}{
		{name: "checkpoint credentials are authoritative"},
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

			config, cleanup, err := m.generateContainerConfigForRestore(tCtx, &pod.Spec.Containers[0], pod, 0, "", pod.Spec.Containers[0].Image, nil, nil)
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
			assert.NotContains(t, fakeImage.Called, "ImageStatus")
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
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WindowsHostNetwork, false)
			pod := &v1.Pod{}
			pod.Spec = *testCase.podSpec

			wc, err := m.generatePodSandboxWindowsConfig(pod)

			assert.Equal(t, testCase.expectedWindowsConfig, wc)
			assert.Equal(t, testCase.expectedError, err)
		})
	}
}
