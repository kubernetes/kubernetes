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
	"testing"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	probetest "k8s.io/kubernetes/pkg/kubelet/prober/testing"
	"k8s.io/kubernetes/pkg/kubelet/status"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeletutil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestNewProbeManager checks the one place the ContainerScopedProbes gate is
// consulted.
func TestNewProbeManager(t *testing.T) {
	for _, tc := range []struct {
		name     string
		gate     bool
		injected bool
		wantType string
	}{{
		name:     "gate off selects the legacy manager",
		wantType: "*prober.manager",
	}, {
		name:     "gate on selects the container-scoped manager",
		gate:     true,
		wantType: "*prober.instanceBoundManager",
	}, {
		name:     "an injected manager wins over either",
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

// recordingProbeManager records the pod statuses SyncPod reconciles probes
// against.
type recordingProbeManager struct {
	probetest.FakeManager
	ensured []*kubecontainer.PodStatus
}

func (m *recordingProbeManager) EnsureProbes(_ context.Context, _ *v1.Pod, podStatus *kubecontainer.PodStatus) {
	m.ensured = append(m.ensured, podStatus)
}

// TestSyncPodEnsuresProbes checks that a pod sync reconciles probes against the
// same runtime state that drives the runtime sync. That is what lets the
// container-scoped manager treat its own view as current: it reads the pod
// status before any container in this sync has started or stopped.
func TestSyncPodEnsuresProbes(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	probes := &recordingProbeManager{}
	kubelet.probeManager = probes

	pod := podWithUIDNameNsSpec("12345678", "probed", "new", v1.PodSpec{
		Containers: []v1.Container{{Name: "foo"}},
	})
	kubelet.podManager.SetPods([]*v1.Pod{pod})

	podStatus := &kubecontainer.PodStatus{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}
	if _, _, err := kubelet.SyncPod(tCtx, kubetypes.SyncPodUpdate, pod, nil, podStatus); err != nil {
		t.Fatalf("SyncPod: %v", err)
	}

	if len(probes.ensured) != 1 {
		t.Fatalf("EnsureProbes called %d times, want 1", len(probes.ensured))
	}
	if probes.ensured[0] != podStatus {
		t.Error("EnsureProbes was given a different pod status than the one driving the sync")
	}
}
