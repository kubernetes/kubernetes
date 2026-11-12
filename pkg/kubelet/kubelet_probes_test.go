/*
Copyright The Kubernetes Authors.

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

package kubelet

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	probetest "k8s.io/kubernetes/pkg/kubelet/prober/testing"
	"k8s.io/kubernetes/pkg/kubelet/status"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeletutil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/utils/ptr"
)

// TestNewProbeManager verifies probe manager selection based on the ContainerScopedProbes gate.
func TestNewProbeManager(t *testing.T) {
	for _, tc := range []struct {
		name     string
		gate     bool
		injected bool
		wantType string
	}{{
		name:     "gate off selects legacy manager",
		wantType: "*prober.manager",
	}, {
		name:     "gate on selects container-scoped manager",
		gate:     true,
		wantType: "*prober.containerBoundManager",
	}, {
		name:     "injected manager wins with gate off",
		injected: true,
		wantType: "testing.FakeManager",
	}, {
		name:     "injected manager wins with gate on",
		gate:     true,
		injected: true,
		wantType: "testing.FakeManager",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerScopedProbes, tc.gate)

			podManager := kubepod.NewBasicPodManager()
			klet := &Kubelet{
				livenessManager:  proberesults.NewManager(),
				readinessManager: proberesults.NewManager(),
				startupManager:   proberesults.NewManager(),
				statusManager: status.NewManager(&fake.Clientset{}, podManager,
					&statustest.FakePodDeletionSafetyProvider{}, kubeletutil.NewPodStartupLatencyTracker()),
			}
			kubeDeps := &Dependencies{Recorder: &record.FakeRecorder{}}
			if tc.injected {
				kubeDeps.ProbeManager = probetest.FakeManager{}
			}

			got := newProbeManager(tCtx, klet, kubeDeps)

			if fmt.Sprintf("%T", got) != tc.wantType {
				t.Errorf("newProbeManager returned %T, want %s", got, tc.wantType)
			}
		})
	}
}

// recordingProbeManager records the pod and runtime status passed to EnsureProbes
// during pod sync reconciliation, as well as call order relative to UpdatePodStatus.
type recordingProbeManager struct {
	probetest.FakeManager
	mu              sync.Mutex
	ensuredPods     []*v1.Pod
	ensuredStatuses []*kubecontainer.PodStatus
	callOrder       []string
}

func (m *recordingProbeManager) EnsureProbes(_ context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callOrder = append(m.callOrder, "EnsureProbes")
	m.ensuredPods = append(m.ensuredPods, pod)
	m.ensuredStatuses = append(m.ensuredStatuses, podStatus)
}

func (m *recordingProbeManager) UpdatePodStatus(_ context.Context, _ *v1.Pod, _ *v1.PodStatus) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callOrder = append(m.callOrder, "UpdatePodStatus")
}

func (m *recordingProbeManager) recorded() ([]*v1.Pod, []*kubecontainer.PodStatus) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*v1.Pod(nil), m.ensuredPods...), append([]*kubecontainer.PodStatus(nil), m.ensuredStatuses...)
}

func (m *recordingProbeManager) order() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.callOrder...)
}

func TestSyncPodEnsuresProbes(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("ContainerScopedProbes=%t", enabled), func(t *testing.T) {
			tCtx := ktesting.Init(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerScopedProbes, enabled)
			testKubelet := newTestKubelet(t, false)
			defer testKubelet.Cleanup()
			kubelet := testKubelet.kubelet
			probes := &recordingProbeManager{}
			kubelet.probeManager = probes
			pod := podWithUIDNameNsSpec("12345678", "probed", "new", v1.PodSpec{
				Containers: []v1.Container{{Name: "foo"}},
			})
			kubelet.podManager.SetPods([]*v1.Pod{pod})
			podStatus := &kubecontainer.PodStatus{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}
			_, _, err := kubelet.SyncPod(tCtx, kubetypes.SyncPodUpdate, pod, nil, podStatus)
			require.NoError(t, err)
			pods, statuses := probes.recorded()
			require.Len(t, statuses, 1)
			require.Same(t, pod, pods[0])
			require.Same(t, podStatus, statuses[0])
			if enabled {
				assert.Equal(t, []string{"EnsureProbes", "UpdatePodStatus"}, probes.order())
			} else {
				assert.Equal(t, []string{"UpdatePodStatus", "EnsureProbes"}, probes.order())
			}
		})
	}
}

func TestSyncPodAdoptsStaticPodProbes(t *testing.T) {
	for _, sidecar := range []bool{false, true} {
		for _, tc := range []struct {
			name        string
			mirror      bool
			matchingPod bool
			containerID string
			started     *bool
			ready       bool
			wantStarted bool
			wantReady   bool
		}{
			{name: "matching mirror preserves started and ready", mirror: true, matchingPod: true, containerID: "test://current", started: new(true), ready: true, wantStarted: true, wantReady: true},
			{name: "matching mirror preserves not ready", mirror: true, matchingPod: true, containerID: "test://current", started: new(true), wantStarted: true},
			{name: "startup not yet complete", mirror: true, matchingPod: true, containerID: "test://current", started: new(false)},
			{name: "mirror missing"},
			{name: "mirror belongs to another static pod revision", mirror: true, containerID: "test://current", started: new(true), ready: true},
			{name: "container replaced", mirror: true, matchingPod: true, containerID: "test://previous", started: new(true), ready: true},
			{name: "container ID missing", mirror: true, matchingPod: true, started: new(true), ready: true},
			{name: "startup status missing", mirror: true, matchingPod: true, containerID: "test://current"},
		} {
			t.Run(fmt.Sprintf("sidecar=%t/%s", sidecar, tc.name), func(t *testing.T) {
				tCtx := ktesting.Init(t)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerScopedProbes, true)
				testKubelet := newTestKubelet(t, false)
				defer testKubelet.Cleanup()
				kl := testKubelet.kubelet
				kl.probeManager = newProbeManager(tCtx, kl, &Dependencies{Recorder: &record.FakeRecorder{}})
				probe := &v1.Probe{
					ProbeHandler:        v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"true"}}},
					InitialDelaySeconds: 3600,
					PeriodSeconds:       1,
					SuccessThreshold:    1,
					FailureThreshold:    1,
				}
				container := v1.Container{Name: "probed", StartupProbe: probe.DeepCopy(), ReadinessProbe: probe.DeepCopy()}
				pod := podWithUIDNameNsSpec("static-uid", "static-pod", "default", v1.PodSpec{
					Containers: []v1.Container{container},
				})
				pod.Annotations = map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource, kubetypes.ConfigHashAnnotationKey: "current-hash"}
				if sidecar {
					container.RestartPolicy = ptr.To(v1.ContainerRestartPolicyAlways)
					pod.Spec.InitContainers = []v1.Container{container}
					pod.Spec.Containers = []v1.Container{{Name: "app"}}
				}
				defer kl.probeManager.RemovePod(pod)
				var mirror *v1.Pod
				if tc.mirror {
					mirror = pod.DeepCopy()
					mirror.UID = "mirror-uid"
					mirror.Annotations = map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.ApiserverSource, kubetypes.ConfigMirrorAnnotationKey: "old-hash"}
					if tc.matchingPod {
						mirror.Annotations[kubetypes.ConfigMirrorAnnotationKey] = "current-hash"
					}
					cs := []v1.ContainerStatus{{Name: container.Name, ContainerID: tc.containerID, Started: tc.started, Ready: tc.ready}}
					if sidecar {
						mirror.Status.InitContainerStatuses = cs
					} else {
						mirror.Status.ContainerStatuses = cs
					}
				}
				originalPod := pod.DeepCopy()
				originalMirror := mirror.DeepCopy()
				pods := []*v1.Pod{pod}
				if mirror != nil {
					pods = append(pods, mirror)
				}
				kl.podManager.SetPods(pods)
				id := kubecontainer.ContainerID{Type: "test", ID: "current"}
				runtimeStatus := &kubecontainer.PodStatus{
					ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace,
					IPs:               []string{"10.0.0.2"},
					ContainerStatuses: []*kubecontainer.Status{{Name: container.Name, ID: id, State: kubecontainer.ContainerStateRunning, StartedAt: time.Now()}},
				}
				_, _, err := kl.SyncPod(tCtx, kubetypes.SyncPodUpdate, pod, mirror, runtimeStatus)
				require.NoError(t, err)
				wantStartup := proberesults.Unknown
				if tc.wantStarted {
					wantStartup = proberesults.Success
				}
				gotStartup, found := kl.startupManager.Get(id)
				require.True(t, found)
				assert.Equal(t, wantStartup, gotStartup)
				gotStatus, found := kl.statusManager.GetPodStatus(pod.UID)
				require.True(t, found)
				statuses := gotStatus.ContainerStatuses
				if sidecar {
					statuses = gotStatus.InitContainerStatuses
				}
				require.Len(t, statuses, 1)
				assert.Equal(t, new(tc.wantStarted), statuses[0].Started)
				assert.Equal(t, tc.wantReady, statuses[0].Ready)
				assert.Equal(t, originalPod, pod)
				assert.Equal(t, originalMirror, mirror)
				assert.Empty(t, kl.startupManager.Updates())
				assert.Empty(t, kl.readinessManager.Updates())
			})
		}
	}
}
