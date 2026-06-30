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
	"context"
	"fmt"
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

// setPodCacheFromFakeRuntime computes the pod status from the fake runtime and
// writes it into the manager's PLEG cache, so the event-driven restart path (which
// reads m.podCache via GetNewerThan/Get) observes the current fake-runtime state.
// Returns the status that was cached.
func setPodCacheFromFakeRuntime(tCtx ktesting.TContext, m *kubeGenericRuntimeManager, rp *kubecontainer.Pod) *kubecontainer.PodStatus {
	t := tCtx.TB()
	status, err := m.GetPodStatus(tCtx, rp)
	require.NoError(t, err)
	require.NotNil(t, status)
	cache, ok := m.podCache.(kubecontainer.Cache)
	require.True(t, ok, "test podCache must be a writable kubecontainer.Cache")
	cache.Set(rp.ID, status, nil, time.Now())
	return status
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

	// The exited status the restart path acts on (as PLEG would supply it).
	status, err := m.GetPodStatus(tCtx, &rp)
	require.NoError(t, err)

	// A sidecar that has exited before its turn must be restarted.
	createsBefore := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	handled := m.restartSidecarDuringTerminationIfExited(tCtx, pod, &pod.Spec.InitContainers[0], 30, kubecontainer.ContainerID{}, status)

	assert.Positive(t, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-createsBefore,
		"expected exited sidecar to be restarted (CreateContainer called) before its SIGTERM turn")
	assert.False(t, handled.IsEmpty(), "the handled exited-instance ID should be returned for dedup")

	// Calling again for the same exited instance must NOT restart again: the
	// replacement is simply not visible in the cache yet. This guards against the
	// CreateContainer name-collision storm observed in node e2e.
	createsAfterFirst := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	again := m.restartSidecarDuringTerminationIfExited(tCtx, pod, &pod.Spec.InitContainers[0], 30, handled, status)
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

	status, err := m.GetPodStatus(tCtx, &rp)
	require.NoError(t, err)
	createsBefore := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	m.restartSidecarDuringTerminationIfExited(tCtx, pod, &pod.Spec.InitContainers[0], 30, kubecontainer.ContainerID{}, status)

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

	// The kill path resolves the live instance from the PLEG cache.
	setPodCacheFromFakeRuntime(tCtx, m, &rp)

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

	// Seed the PLEG cache with the current (exited) sidecar status the restart
	// path observes; PLEG would supply it in production.
	setPodCacheFromFakeRuntime(tCtx, m, &rp)

	done := make(chan *kubecontainer.SyncResult, 1)
	go func() {
		done <- m.killRestartableInitContainerWithSyncResult(tCtx, pod, &rp, snapshot,
			&pod.Spec.InitContainers[0], 30, termOrdering)
	}()

	// The sidecar must be restarted while it waits for its turn.
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

	// The kill path resolves the live instance from the PLEG cache.
	setPodCacheFromFakeRuntime(tCtx, m, &rp)

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

// buildRunningPodExcluding builds a kubecontainer.Pod from the fake containers
// whose names are NOT in exclude, mirroring ConvertPodStatusToRunningPod
// (helpers.go:352) dropping containers that are no longer Running (e.g. a sidecar
// that has already exited when termination begins).
func buildRunningPodExcluding(tCtx ktesting.TContext, m *kubeGenericRuntimeManager,
	pod *v1.Pod, fakeSandbox *apitest.FakePodSandbox,
	fakeContainers []*apitest.FakeContainer, exclude ...string) kubecontainer.Pod {
	excluded := map[string]struct{}{}
	for _, name := range exclude {
		excluded[name] = struct{}{}
	}
	var kept []*apitest.FakeContainer
	for _, fc := range fakeContainers {
		if _, ok := excluded[fc.Metadata.Name]; ok {
			continue
		}
		kept = append(kept, fc)
	}
	return buildRunningPodForKill(tCtx, m, pod, fakeSandbox, kept)
}

// TestKillContainers_Terminating_ExitedSidecarAbsentFromRunningPod_Restarts is the
// core regression for KEP-4438 edge case A: a restartable init container that has
// already exited when termination begins is absent from runningPod.Containers
// (ConvertPodStatusToRunningPod keeps only Running containers), yet it must still
// be restarted within the grace window because it is present in the pod spec.
//
// The main container's StopContainer is made to fail so its ordering channel stays
// open for the whole (short) grace, deterministically giving the sidecar watcher
// time to observe the exit and restart it before its own ordered turn arrives.
//
// Against pre-fix code this fails (silent miss: the exited sidecar was never
// dispatched, so CreateContainer is never called) rather than panicking.
func TestKillContainers_Terminating_ExitedSidecarAbsentFromRunningPod_Restarts(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	// The sidecar has already exited when termination begins.
	setSidecarStateExited(fakeRuntime, "sidecar")

	// runningPod excludes the exited sidecar, exactly as ConvertPodStatusToRunningPod
	// would; the sidecar remains in pod.Spec.InitContainers.
	runningPod := buildRunningPodExcluding(tCtx, m, pod, fakeSandbox, fakeContainers, "sidecar")

	// Seed the PLEG cache from the full fake-runtime state (sidecar exited, main
	// running) so the restart path observes the exit. The cache is keyed by pod UID,
	// so it is shared with runningPod; getPodContainerStatuses only queries the
	// containers it is given, so it must be seeded from the full container set.
	fullPod := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	setPodCacheFromFakeRuntime(tCtx, m, &fullPod)

	// Make the main container's stop fail so its ordering channel never closes,
	// keeping the sidecar waiting for the full grace period.
	fakeRuntime.InjectError("StopContainer", fmt.Errorf("injected stop error for main"))

	createsBefore := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	grace := int64(2)
	m.killContainersWithSyncResult(tCtx, pod, runningPod, &grace, true)

	assert.Positive(t, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-createsBefore,
		"an exited sidecar absent from runningPod.Containers must still be restarted during grace")
}

// TestKillContainers_Terminating_ExtraSidecar_NoPanicOneRestartNoHang covers the
// extra-sidecar dispatch guards (KEP-4438): feeding the extra sidecar's name into
// the termination ordering must prevent a close-of-closed-channel panic, the
// buffer/wg sizing must not deadlock wg.Wait, and the sidecar must be restarted
// exactly once (per-exited-instance dedup). Completion of this test is itself the
// no-panic / no-hang assertion: a double-close would crash the test binary via the
// re-raised HandleCrash panic, and an undersized buffer would hang wg.Wait.
func TestKillContainers_Terminating_ExtraSidecar_NoPanicOneRestartNoHang(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	setSidecarStateExited(fakeRuntime, "sidecar")

	runningPod := buildRunningPodExcluding(tCtx, m, pod, fakeSandbox, fakeContainers, "sidecar")
	fullPod := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	setPodCacheFromFakeRuntime(tCtx, m, &fullPod)
	fakeRuntime.InjectError("StopContainer", fmt.Errorf("injected stop error for main"))

	createsBefore := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
	grace := int64(2)

	done := make(chan struct{})
	go func() {
		defer close(done)
		m.killContainersWithSyncResult(tCtx, pod, runningPod, &grace, true)
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("killContainersWithSyncResult hung: extra-sidecar buffer/wg sizing deadlocked wg.Wait")
	}

	assert.Equal(t, 1, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-createsBefore,
		"the exited sidecar must be restarted exactly once (per-exited-instance dedup)")
}

// TestKillContainers_SharedPath_ZeroDelta_GateOff proves the C1 reorder is
// behavior-neutral when there are no extra sidecars (feature gate off): the
// returned sync-result set must be exactly one KillContainer per running
// container, with no added/dropped results and no restart. This asserts
// result-set equality, distinguishing it from the coarser
// TestKillContainers_GateDisabled_NoRestart (which only checks that no restart
// happened).
func TestKillContainers_SharedPath_ZeroDelta_GateOff(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, false)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)

	runningPod := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	setPodCacheFromFakeRuntime(tCtx, m, &runningPod)

	// Expected target set: one KillContainer result per running container.
	expected := map[string]struct{}{}
	for _, c := range runningPod.Containers {
		expected[c.Name] = struct{}{}
	}

	grace := int64(30)
	results := m.killContainersWithSyncResult(tCtx, pod, runningPod, &grace, true)

	got := map[string]struct{}{}
	for _, r := range results {
		require.Equal(t, kubecontainer.KillContainer, r.Action, "shared path must only produce KillContainer results")
		name, ok := r.Target.(string)
		require.True(t, ok, "result target must be a container name string")
		got[name] = struct{}{}
	}
	assert.Equal(t, expected, got,
		"shared-path result set must equal exactly the running containers (zero delta when no extra sidecars)")
	assert.Len(t, results, len(runningPod.Containers),
		"exactly one result per running container, none added or dropped")
	assert.Equal(t, 0, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"),
		"no restart on the shared path with the gate off")
}

// TestKillRestartableInitContainer_NilSnapshot_PrereqsMet verifies that the
// extra-sidecar path (dispatched with a nil snapshot) does not nil-deref: with the
// sidecar's ordered turn already met, the call returns promptly and stops the live
// instance resolved from the PLEG cache (not from the absent snapshot).
func TestKillRestartableInitContainer_NilSnapshot_PrereqsMet(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)

	// Only sidecar listed as running → main channel is pre-closed → prereqs met.
	termOrdering := newTerminationOrdering(pod, []string{"sidecar"})

	rp := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	setPodCacheFromFakeRuntime(tCtx, m, &rp)

	stopsBefore := countCallsOf(fakeRuntime.GetCalls(), "StopContainer")

	done := make(chan *kubecontainer.SyncResult, 1)
	go func() {
		// nil snapshot: the extra-sidecar dispatch path.
		done <- m.killRestartableInitContainerWithSyncResult(tCtx, pod, &rp, nil,
			&pod.Spec.InitContainers[0], 30, termOrdering)
	}()
	select {
	case res := <-done:
		require.NotNil(t, res)
		require.NoError(t, res.Error)
	case <-time.After(5 * time.Second):
		t.Fatal("killRestartableInitContainerWithSyncResult with nil snapshot did not return despite prereqs met")
	}

	assert.Positive(t, countCallsOf(fakeRuntime.GetCalls(), "StopContainer")-stopsBefore,
		"the live sidecar instance resolved from the cache must be stopped")
}

// TestKillRestartableInitContainer_NilSnapshot_StopsLiveInstance verifies that with
// a nil snapshot the stopped instance is the one currently live per the PLEG cache
// (the restarted instance in production), not one derived from the absent snapshot.
func TestKillRestartableInitContainer_NilSnapshot_StopsLiveInstance(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)

	// Only sidecar listed as running → prereqs met immediately.
	termOrdering := newTerminationOrdering(pod, []string{"sidecar"})

	rp := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	setPodCacheFromFakeRuntime(tCtx, m, &rp)

	// The live sidecar instance ID the cache resolves to.
	cached, err := m.podCache.Get(rp.ID)
	require.NoError(t, err)
	liveCS := cached.FindContainerStatusByName("sidecar")
	require.NotNil(t, liveCS)
	require.False(t, liveCS.ID.IsEmpty())

	done := make(chan *kubecontainer.SyncResult, 1)
	go func() {
		done <- m.killRestartableInitContainerWithSyncResult(tCtx, pod, &rp, nil,
			&pod.Spec.InitContainers[0], 30, termOrdering)
	}()
	select {
	case res := <-done:
		require.NotNil(t, res)
		require.NoError(t, res.Error)
	case <-time.After(5 * time.Second):
		t.Fatal("killRestartableInitContainerWithSyncResult with nil snapshot did not return")
	}

	// StopContainer sets the target fake container to EXITED; assert the live
	// (cache-resolved) instance is the one that was stopped.
	fakeRuntime.Lock()
	stopped := fakeRuntime.Containers[liveCS.ID.ID]
	fakeRuntime.Unlock()
	require.NotNil(t, stopped, "the cache-resolved live instance must exist in the runtime")
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, stopped.State,
		"the live instance resolved from the cache must be the one stopped")
}

// TestRestartSidecarDuringTermination_RestartedEachExitNoBackoff verifies the
// no-restart-backoff behavior at the unit level: every distinct premature exit of
// the sidecar triggers a fresh restart for the remainder of the grace window; there
// is no backoff that suppresses later restarts (only same-instance duplicates are
// deduplicated).
func TestRestartSidecarDuringTermination_RestartedEachExitNoBackoff(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	setSidecarStateExited(fakeRuntime, "sidecar")

	rp := buildRunningPodForKill(tCtx, m, pod, fakeSandbox, fakeContainers)
	status, err := m.GetPodStatus(tCtx, &rp)
	require.NoError(t, err)
	cs := status.FindContainerStatusByName("sidecar")
	require.NotNil(t, cs)
	require.Equal(t, kubecontainer.ContainerStateExited, cs.State)

	handled := kubecontainer.ContainerID{}
	for i := range 3 {
		// Simulate a fresh premature exit: a new exited instance ID each iteration.
		cs.ID = kubecontainer.ContainerID{Type: apitest.FakeRuntimeName, ID: fmt.Sprintf("sidecar-exit-%d", i)}
		before := countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")
		handled = m.restartSidecarDuringTerminationIfExited(tCtx, pod, &pod.Spec.InitContainers[0], 30, handled, status)
		assert.Equal(t, 1, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer")-before,
			"each distinct exit must trigger exactly one restart (no backoff), iteration %d", i)
		assert.Equal(t, cs.ID, handled, "the handled ID must advance to the just-restarted exited instance, iteration %d", i)
	}
}

// TestKillContainers_ShortGraceExcluded_NoRestart verifies the gracePeriodOverride>1
// guard: with a grace of 1s (or less) an already-exited sidecar absent from
// runningPod.Containers is NOT dispatched for restart, so no CreateContainer occurs.
func TestKillContainers_ShortGraceExcluded_NoRestart(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate,
		features.SidecarsRestartableDuringPodTermination, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	pod := newSidecarRestartTestPod()
	fakeSandbox, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	setSidecarStateExited(fakeRuntime, "sidecar")

	runningPod := buildRunningPodExcluding(tCtx, m, pod, fakeSandbox, fakeContainers, "sidecar")
	setPodCacheFromFakeRuntime(tCtx, m, &runningPod)

	grace := int64(1)
	m.killContainersWithSyncResult(tCtx, pod, runningPod, &grace, true)

	assert.Equal(t, 0, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"),
		"grace <= 1s must exclude the sidecar restart-during-termination path (gracePeriodOverride>1 guard)")
}

// TestWatchPodStatus_StaleGlobalTimestamp_NoBusyLoop is a regression test for the
// GetNewerThan cursor in watchPodStatus. Cursoring on status.TimeStamp busy-loops
// once a PLEG relist advances the cache's global timestamp past the (unchanged)
// status timestamp: from then on every GetNewerThan returns immediately. The
// watcher must instead advance its cursor by wall-clock time, so a relist that
// does not change the pod's status wakes it at most once.
func TestWatchPodStatus_StaleGlobalTimestamp_NoBusyLoop(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	cache, ok := m.podCache.(kubecontainer.Cache)
	require.True(t, ok, "test podCache must be a writable kubecontainer.Cache")

	const uid = "watch-cursor-uid"
	runningPod := &kubecontainer.Pod{ID: uid, Name: "test-pod", Namespace: "default"}

	// Seed a status whose own TimeStamp is well in the past. modified/observed are
	// set to the same past time, so an old-style status.TimeStamp cursor would sit
	// at this value permanently.
	seedTime := time.Now().Add(-time.Hour)
	cache.Set(uid, &kubecontainer.PodStatus{ID: uid, TimeStamp: seedTime}, nil, seedTime)

	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()
	ch := m.watchPodStatus(ctx, runningPod)

	// The first call returns the current cached status.
	select {
	case status := <-ch:
		require.NotNil(t, status)
	case <-time.After(5 * time.Second):
		t.Fatal("watchPodStatus did not deliver the initial status")
	}

	// Simulate a PLEG relist that advances the cache's global timestamp but leaves
	// the pod's status unchanged. The relist time is newer than the seed (so an
	// old-style status.TimeStamp cursor would fire) but older than the watcher's
	// wall-clock cursor (so the fixed cursor stays blocked).
	cache.UpdateTime(time.Now().Add(-time.Minute))

	// With the wall-clock cursor the watcher stays blocked: a relist that did not
	// change the status must not produce another delivery. With the old cursor this
	// channel would fire repeatedly (busy loop).
	select {
	case <-ch:
		t.Fatal("watchPodStatus delivered again after a status-preserving relist: the cursor is busy-looping")
	case <-time.After(500 * time.Millisecond):
		// No spurious delivery: correct.
	}

	// Clean up the watcher goroutine: cancel, then wake its blocked GetNewerThan with
	// a genuinely newer global timestamp so it observes the cancelled context and
	// returns (GetNewerThan itself is not cancellable).
	cancel()
	cache.UpdateTime(time.Now().Add(time.Hour))
}
