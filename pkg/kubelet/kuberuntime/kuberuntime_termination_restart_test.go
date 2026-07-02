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

package kuberuntime

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitest "k8s.io/cri-api/pkg/apis/testing"

	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func newSidecarRestartTestPod() *v1.Pod {
	restartPolicy := v1.ContainerRestartPolicyAlways
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "sidecar-restart-test-uid",
			Name:      "test-pod",
			Namespace: "default",
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:            "sidecar",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					RestartPolicy:   &restartPolicy,
				},
			},
			Containers: []v1.Container{
				{
					Name:            "main",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}
}

func countCallsOf(calls []string, name string) int {
	n := 0
	for _, c := range calls {
		if c == name {
			n++
		}
	}
	return n
}

func setSidecarStateExited(fakeRuntime *apitest.FakeRuntimeService, containerName string) {
	fakeRuntime.Lock()
	defer fakeRuntime.Unlock()
	for _, c := range fakeRuntime.Containers {
		if c.Metadata.Name == containerName {
			c.State = runtimeapi.ContainerState_CONTAINER_EXITED
		}
	}
}

// buildRunningPodForKill converts fake containers into a kubecontainer.Pod for
// use with killContainersWithSyncResult.
func buildRunningPodForKill(tCtx ktesting.TContext, m *kubeGenericRuntimeManager,
	pod *v1.Pod, fakeSandbox *apitest.FakePodSandbox,
	fakeContainers []*apitest.FakeContainer) kubecontainer.Pod {
	t := tCtx.TB()
	containers := make([]*kubecontainer.Container, len(fakeContainers))
	for i, fc := range fakeContainers {
		c, err := m.toKubeContainer(tCtx, &runtimeapi.Container{
			Id:       fc.Id,
			Metadata: fc.Metadata,
			State:    fc.State,
			Image:    fc.Image,
			ImageRef: fc.ImageRef,
			Labels:   fc.Labels,
		})
		require.NoError(t, err)
		containers[i] = c
	}
	return kubecontainer.Pod{
		ID:         pod.UID,
		Name:       pod.Name,
		Namespace:  pod.Namespace,
		Containers: containers,
		Sandboxes: []*kubecontainer.Container{
			{
				ID: kubecontainer.ContainerID{
					ID:   fakeSandbox.Id,
					Type: apitest.FakeRuntimeName,
				},
			},
		},
	}
}

// TestRestartSidecarDuringTermination_ExitsBeforeTurn verifies that a sidecar which has
// exited before its ordered termination turn (main container still running) is restarted,
// and that a repeat call for the same exited instance does not restart it again (dedup).
func TestRestartSidecarDuringTermination_ExitsBeforeTurn(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()

	// All containers start RUNNING; then mark sidecar as exited to simulate premature exit.
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	setSidecarStateExited(fakeRuntime, "sidecar")

	// GetPodStatus needs pod.Containers to query ContainerStatus per container.
	rp := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)

	// A sidecar that has exited before its turn must be restarted.
	createsBefore := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	handled := m.restartSidecarDuringTerminationIfExited(tCtx, pod, &rp, &pod.Spec.InitContainers[0], 30, kubecontainer.ContainerID{})

	assert.Positive(t, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-createsBefore,
		"expected exited sidecar to be restarted (CreateContainer called) before its SIGTERM turn")
	assert.False(t, handled.IsEmpty(), "the handled exited-instance ID should be returned for dedup")

	// Calling again with the same exited instance must NOT restart again: the
	// replacement is simply not visible in GetPodStatus yet. This guards against
	// the CreateContainer name-collision storm observed in node e2e.
	createsAfterFirst := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	again := m.restartSidecarDuringTerminationIfExited(tCtx, pod, &rp, &pod.Spec.InitContainers[0], 30, handled)
	assert.Equal(t, handled, again, "handled ID should be unchanged for the same exited instance")
	assert.Equal(t, 0, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-createsAfterFirst,
		"must not restart the same exited instance twice")
}

// TestRestartSidecarDuringTermination_RunningNoRestart verifies that a sidecar
// which is still running is not restarted.
func TestRestartSidecarDuringTermination_RunningNoRestart(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	// sidecar is left RUNNING.
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	rp := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)

	createsBefore := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	m.restartSidecarDuringTerminationIfExited(tCtx, pod, &rp, &pod.Spec.InitContainers[0], 30, kubecontainer.ContainerID{})

	assert.Equal(t, 0, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-createsBefore,
		"running sidecar must not be restarted")
}

// TestKillRestartableInitContainer_PrereqsMet verifies that when the sidecar's
// SIGTERM turn has already arrived (prereqs met), it is gracefully stopped
// without being restarted, and the call returns promptly.
func TestKillRestartableInitContainer_PrereqsMet(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)

	// Only sidecar listed as running → main channel is pre-closed → prereqs met immediately.
	termOrdering := newTerminationOrdering(pod, []string{"sidecar"})

	rp := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	var snapshot *kubecontainer.Container
	for _, c := range rp.Containers {
		if c.Name == "sidecar" {
			snapshot = c
		}
	}
	require.NotNil(t, snapshot)

	createsBefore := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	stopsBefore := countCallsOf(fakeRuntime.GetCalls(), "StopContainer")

	done := make(chan *kubecontainer.SyncResult, 1)
	go func() {
		done <- m.killRestartableInitContainerWithSyncResult(tCtx, pod, &rp, snapshot,
			&pod.Spec.InitContainers[0], 30, termOrdering)
	}()
	select {
	case res := <-done:
		require.NotNil(t, res)
		require.NoError(t, res.Error)
	case <-time.After(5 * time.Second):
		t.Fatal("killRestartableInitContainerWithSyncResult did not return despite prereqs being met")
	}

	assert.Equal(t, 0, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-createsBefore,
		"sidecar must not be restarted when its SIGTERM turn has already arrived")
	assert.Positive(t, countCallsOf(fakeRuntime.GetCalls(), "StopContainer")-stopsBefore,
		"sidecar should be stopped once its turn arrives")
}

// TestKillRestartableInitContainer_RestartsThenKills exercises the full sequence:
// a sidecar that has exited before its ordered turn is restarted while it waits,
// and once its turn arrives the live (restarted) instance is gracefully stopped
// in order — it is not killed before its turn.
func TestKillRestartableInitContainer_RestartsThenKills(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	// Sidecar has exited on its own; main is still running, so the sidecar's
	// ordered turn has not arrived yet.
	setSidecarStateExited(fakeRuntime, "sidecar")

	// Both containers listed as running → main's channel is open → sidecar prereqs
	// are not met until we close it below.
	termOrdering := newTerminationOrdering(pod, []string{"main", "sidecar"})

	rp := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	var snapshot *kubecontainer.Container
	for _, c := range rp.Containers {
		if c.Name == "sidecar" {
			snapshot = c
		}
	}
	require.NotNil(t, snapshot)

	createsBefore := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	stopsBefore := countCallsOf(fakeRuntime.GetCalls(), "StopContainer")

	done := make(chan *kubecontainer.SyncResult, 1)
	go func() {
		done <- m.killRestartableInitContainerWithSyncResult(tCtx, pod, &rp, snapshot,
			&pod.Spec.InitContainers[0], 30, termOrdering)
	}()

	// The watcher must restart the exited sidecar while it waits for its turn.
	require.Eventually(t, func() bool {
		return countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-createsBefore > 0
	}, 5*time.Second, 50*time.Millisecond, "exited sidecar should be restarted while waiting for its turn")

	// It must not be stopped before its ordered turn arrives.
	assert.Equal(t, 0, countCallsOf(fakeRuntime.GetCalls(), "StopContainer")-stopsBefore,
		"sidecar must not be stopped before its ordered turn")

	// Release the sidecar's turn; it should now be gracefully stopped and return.
	termOrdering.containerTerminated("main")

	select {
	case res := <-done:
		require.NotNil(t, res)
		require.NoError(t, res.Error)
	case <-time.After(5 * time.Second):
		t.Fatal("killRestartableInitContainerWithSyncResult did not return after its turn arrived")
	}

	assert.Positive(t, countCallsOf(fakeRuntime.GetCalls(), "StopContainer")-stopsBefore,
		"sidecar should be gracefully stopped once its turn arrives")
}

// TestKillRestartableInitContainer_GraceBound verifies that the wait for a
// sidecar's ordered turn is bounded by the grace period: if a prerequisite
// container never reports as terminated, the call still returns once the grace
// period elapses rather than hanging the pod in Terminating forever.
func TestKillRestartableInitContainer_GraceBound(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)

	// Both running, but main's channel is never closed: the sidecar's turn never
	// arrives on its own, so only the grace deadline can end the wait.
	termOrdering := newTerminationOrdering(pod, []string{"main", "sidecar"})

	rp := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	var snapshot *kubecontainer.Container
	for _, c := range rp.Containers {
		if c.Name == "sidecar" {
			snapshot = c
		}
	}
	require.NotNil(t, snapshot)

	const grace = int64(3)
	done := make(chan *kubecontainer.SyncResult, 1)
	start := time.Now()
	go func() {
		done <- m.killRestartableInitContainerWithSyncResult(tCtx, pod, &rp, snapshot,
			&pod.Spec.InitContainers[0], grace, termOrdering)
	}()

	select {
	case res := <-done:
		require.NotNil(t, res)
		require.NoError(t, res.Error)
		assert.GreaterOrEqual(t, time.Since(start), 2*time.Second,
			"should wait for roughly the grace period before giving up, not return immediately")
	case <-time.After(15 * time.Second):
		t.Fatal("killRestartableInitContainerWithSyncResult hung waiting for a turn that never arrives")
	}

	assert.Positive(t, countCallsOf(fakeRuntime.GetCalls(), "StopContainer"),
		"sidecar should still be stopped after the grace deadline")
}

// TestKillContainers_GateDisabled_NoRestart verifies that with the feature gate disabled,
// no sidecar restart occurs even when terminating=true.
func TestKillContainers_GateDisabled_NoRestart(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, false)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	setSidecarStateExited(fakeRuntime, "sidecar")

	runningPod := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	grace := int64(30)
	m.killContainersWithSyncResult(tCtx, pod, runningPod, &grace, true)

	assert.Equal(t, 0, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"),
		"gate disabled: no sidecar restart expected")
}

// TestKillContainers_NotTerminating_NoRestart verifies that with terminating=false,
// no sidecar restart watcher is spawned (sandbox-replacement path).
func TestKillContainers_NotTerminating_NoRestart(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	setSidecarStateExited(fakeRuntime, "sidecar")

	runningPod := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	grace := int64(30)
	m.killContainersWithSyncResult(tCtx, pod, runningPod, &grace, false)

	assert.Equal(t, 0, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"),
		"terminating=false: no sidecar restart expected")
}
