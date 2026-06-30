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
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/flowcontrol"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitest "k8s.io/cri-api/pkg/apis/testing"
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

type terminationStopCall struct {
	id    string
	grace int64
	done  chan struct{}
}

type terminationRuntime struct {
	internalapi.RuntimeService
	calls chan terminationStopCall
}

func (r *terminationRuntime) StopContainer(ctx context.Context, id string, grace int64) error {
	call := terminationStopCall{id: id, grace: grace, done: make(chan struct{})}
	r.calls <- call
	defer close(call.done)
	<-ctx.Done()
	return ctx.Err()
}

func nextTerminationStop(t *testing.T, r *terminationRuntime) terminationStopCall {
	t.Helper()
	select {
	case call := <-r.calls:
		return call
	case <-time.After(5 * time.Second):
		t.Fatal("expected a StopContainer request")
	}
	return terminationStopCall{}
}

func setupTerminationTest(t *testing.T) (ktesting.TContext, *kubeGenericRuntimeManager, *apitest.FakeRuntimeService, *v1.Pod, *kubecontainer.PodStatus, *terminationRuntime) {
	t.Helper()
	ctx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(ctx)
	require.NoError(t, err)
	pod := newSidecarRestartTestPod()
	makeAndSetFakePod(ctx, m, fakeRuntime, pod)
	rp, err := m.GetPod(ctx, pod.UID)
	require.NoError(t, err)
	status, err := m.GetPodStatus(ctx, rp)
	require.NoError(t, err)
	r := &terminationRuntime{RuntimeService: fakeRuntime, calls: make(chan terminationStopCall, 20)}
	m.runtimeService = r
	t.Cleanup(func() {
		for _, ops := range m.terminations {
			for _, op := range ops.containers {
				if op.hook != nil {
					op.hook.cancel()
				}
				if op.stop != nil {
					op.stop.cancel()
				}
			}
		}
	})
	return ctx, m, fakeRuntime, pod, status, r
}

func TestContainersToKeepDuringTermination(t *testing.T) {
	pod := newSidecarRestartTestPod()
	sidecar := pod.Spec.InitContainers[0].DeepCopy()
	sidecar.Name = "later"
	pod.Spec.InitContainers = append(pod.Spec.InitContainers, *sidecar)
	for _, tc := range []struct {
		name   string
		states map[string]kubecontainer.State
		want   map[string]bool
	}{
		{"application running", map[string]kubecontainer.State{"main": kubecontainer.ContainerStateRunning, "sidecar": kubecontainer.ContainerStateExited, "later": kubecontainer.ContainerStateExited}, map[string]bool{"sidecar": true, "later": true}},
		{"last sidecar draining", map[string]kubecontainer.State{"main": kubecontainer.ContainerStateExited, "sidecar": kubecontainer.ContainerStateExited, "later": kubecontainer.ContainerStateRunning}, map[string]bool{"sidecar": true, "later": false}},
		{"last sidecar exited", map[string]kubecontainer.State{"main": kubecontainer.ContainerStateExited, "sidecar": kubecontainer.ContainerStateRunning, "later": kubecontainer.ContainerStateExited}, map[string]bool{"sidecar": false, "later": false}},
		{"application state unknown", map[string]kubecontainer.State{"main": kubecontainer.ContainerStateUnknown}, map[string]bool{"sidecar": true, "later": true}},
		{"never started", nil, map[string]bool{"sidecar": false, "later": false}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status := &kubecontainer.PodStatus{}
			for name, state := range tc.states {
				status.ContainerStatuses = append(status.ContainerStatuses, &kubecontainer.Status{Name: name, State: state})
			}
			assert.Equal(t, tc.want, containersToKeepDuringTermination(pod, status))
		})
	}
}

func TestSyncTerminatingPodRestartsAndDeduplicates(t *testing.T) {
	ctx, m, fakeRuntime, pod, status, r := setupTerminationTest(t)
	status.FindContainerStatusByName("sidecar").State = kubecontainer.ContainerStateExited
	setSidecarStateExited(fakeRuntime, "sidecar")
	deadline := time.Now().Add(time.Minute)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	done, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.False(t, done)
	require.Equal(t, 1, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
	call := nextTerminationStop(t, r)
	require.Equal(t, status.FindContainerStatusByName("main").ID.ID, call.id)
	// A repeated PLEG observation must not create another instance or Stop RPC.
	done, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.False(t, done)
	require.Equal(t, 1, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
	require.Empty(t, r.calls)
	// A fresh observation contains the replacement ID. Once the application
	// exits, that live replacement must receive its ordered StopContainer call.
	rp, err := m.GetPod(ctx, pod.UID)
	require.NoError(t, err)
	status, err = m.GetPodStatus(ctx, rp)
	require.NoError(t, err)
	status.FindContainerStatusByName("main").State = kubecontainer.ContainerStateExited
	replacement := status.FindContainerStatusByName("sidecar")
	require.Equal(t, kubecontainer.ContainerStateRunning, replacement.State)
	done, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.False(t, done)
	call = nextTerminationStop(t, r)
	require.Equal(t, replacement.ID.ID, call.id)
	replacement.State = kubecontainer.ContainerStateExited
	done, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.True(t, done)
	require.Empty(t, m.terminations)
}

func TestSyncTerminatingPodDeadline(t *testing.T) {
	ctx, m, fakeRuntime, pod, status, r := setupTerminationTest(t)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	deadline := time.Now().Add(time.Minute)
	done, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.False(t, done)
	original := nextTerminationStop(t, r)
	require.Greater(t, original.grace, int64(50))
	deadline = time.Now().Add(-time.Second)
	done, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.False(t, done)
	// Expiry cancels the original call and stops both containers immediately,
	// even though the application's status has not changed.
	for range 2 {
		call := nextTerminationStop(t, r)
		require.Zero(t, call.grace)
	}
	select {
	case <-original.done:
	case <-time.After(5 * time.Second):
		t.Fatal("superseded StopContainer call was not cancelled")
	}
	status.FindContainerStatusByName("sidecar").State = kubecontainer.ContainerStateExited
	_, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.Zero(t, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
}

func TestSyncTerminatingPodDoesNotStartIneligibleSidecars(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mutate    func(*kubecontainer.PodStatus)
		remaining time.Duration
	}{
		{"still running", func(*kubecontainer.PodStatus) {}, time.Minute},
		{"never started", func(s *kubecontainer.PodStatus) {
			s.ContainerStatuses = slices.DeleteFunc(s.ContainerStatuses, func(cs *kubecontainer.Status) bool { return cs.Name == "sidecar" })
		}, time.Minute},
		{"ordered turn", func(s *kubecontainer.PodStatus) {
			for _, cs := range s.ContainerStatuses {
				cs.State = kubecontainer.ContainerStateExited
			}
		}, time.Minute},
		{"sandbox not ready", func(s *kubecontainer.PodStatus) {
			s.FindContainerStatusByName("sidecar").State = kubecontainer.ContainerStateExited
			s.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
		}, time.Minute},
		{"short remaining grace", func(s *kubecontainer.PodStatus) {
			s.FindContainerStatusByName("sidecar").State = kubecontainer.ContainerStateExited
		}, time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, m, fakeRuntime, pod, status, _ := setupTerminationTest(t)
			tc.mutate(status)
			_, err := m.SyncTerminatingPod(ctx, pod, status, nil, flowcontrol.NewBackOff(time.Second, time.Minute), time.Now().Add(tc.remaining))
			require.NoError(t, err)
			require.Zero(t, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
		})
	}
}

func TestSyncTerminatingPodRetriesFailedStart(t *testing.T) {
	ctx, m, fakeRuntime, pod, status, _ := setupTerminationTest(t)
	status.FindContainerStatusByName("sidecar").State = kubecontainer.ContainerStateExited
	setSidecarStateExited(fakeRuntime, "sidecar")
	fakeRuntime.InjectError("CreateContainer", errors.New("test create failure"))
	// Expired backoff makes the retry deterministic without waiting in this test.
	status.FindContainerStatusByName("sidecar").FinishedAt = time.Now().Add(-time.Hour)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	deadline := time.Now().Add(time.Minute)
	_, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.ErrorContains(t, err, "test create failure")
	require.Empty(t, m.terminations[pod.UID].restarted)
	_, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.Equal(t, 2, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
}

func TestSyncTerminatingPodRestartBackoff(t *testing.T) {
	ctx, m, fakeRuntime, pod, status, _ := setupTerminationTest(t)
	cs := status.FindContainerStatusByName("sidecar")
	cs.State = kubecontainer.ContainerStateExited
	cs.FinishedAt = time.Now()
	backOff := flowcontrol.NewBackOff(time.Minute, time.Minute)
	backOff.Next(GetBackoffKey(pod, &pod.Spec.InitContainers[0]), cs.FinishedAt)
	done, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, time.Now().Add(time.Minute))
	require.False(t, done)
	require.ErrorIs(t, err, kubecontainer.ErrCrashLoopBackOff)
	require.Zero(t, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
}

type terminationHookRunner struct {
	calls   chan kubecontainer.ContainerID
	release chan struct{}
}

func (r *terminationHookRunner) Run(ctx context.Context, id kubecontainer.ContainerID, pod *v1.Pod, container *v1.Container, handler *v1.LifecycleHandler) (string, error) {
	r.calls <- id
	select {
	case <-r.release:
		return "", nil
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func TestSyncTerminatingPodPreStopDoesNotBlockReconciliation(t *testing.T) {
	ctx, m, fakeRuntime, pod, status, r := setupTerminationTest(t)
	hook := &v1.Lifecycle{PreStop: &v1.LifecycleHandler{Exec: &v1.ExecAction{Command: []string{"drain"}}}}
	pod.Spec.Containers[0].Lifecycle = hook
	pod.Spec.InitContainers[0].Lifecycle = hook
	runner := &terminationHookRunner{calls: make(chan kubecontainer.ContainerID, 4), release: make(chan struct{})}
	m.runner = runner
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	deadline := time.Now().Add(time.Minute)
	done, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.False(t, done)
	// Both hooks run immediately, including the sidecar's, while the pod worker
	// remains free to observe and restart a sidecar that exits during either hook.
	for range 2 {
		select {
		case <-runner.calls:
		case <-time.After(5 * time.Second):
			t.Fatal("preStop hook was not started")
		}
	}
	require.Empty(t, r.calls)
	status.FindContainerStatusByName("sidecar").State = kubecontainer.ContainerStateExited
	setSidecarStateExited(fakeRuntime, "sidecar")
	done, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.False(t, done)
	require.Equal(t, 1, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
	require.Empty(t, runner.calls, "reconciliation must not run the application's hook twice")
	close(runner.release)
	mainID := status.FindContainerStatusByName("main").ID
	require.Eventually(t, func() bool {
		return len(m.terminations[pod.UID].containers[mainID].hook.done) > 0
	}, 5*time.Second, time.Millisecond)
	_, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	call := nextTerminationStop(t, r)
	require.Equal(t, mainID.ID, call.id)
}

func TestSyncTerminatingPodReconstructsAfterKubeletRestart(t *testing.T) {
	ctx, m, fakeRuntime, pod, status, r := setupTerminationTest(t)
	deadline := time.Now().Add(20 * time.Second)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	_, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	nextTerminationStop(t, r)
	// The replacement runtime manager has no in-process termination records.
	_, _, restarted, err := createTestRuntimeManager(ctx)
	require.NoError(t, err)
	restarted.runtimeService = r
	status.FindContainerStatusByName("sidecar").State = kubecontainer.ContainerStateExited
	setSidecarStateExited(fakeRuntime, "sidecar")
	done, err := restarted.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.False(t, done)
	require.Equal(t, 1, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
	call := nextTerminationStop(t, r)
	require.LessOrEqual(t, call.grace, int64(20), "kubelet restart must not renew the grace budget")
	for _, cs := range status.ContainerStatuses {
		cs.State = kubecontainer.ContainerStateExited
	}
	_, err = restarted.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
}

func TestSyncTerminatingPodRetriesPartialStart(t *testing.T) {
	ctx, m, fakeRuntime, pod, status, _ := setupTerminationTest(t)
	status.FindContainerStatusByName("sidecar").State = kubecontainer.ContainerStateExited
	setSidecarStateExited(fakeRuntime, "sidecar")
	fakeRuntime.InjectError("StartContainer", errors.New("test start failure"))
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	deadline := time.Now().Add(time.Minute)
	_, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.ErrorContains(t, err, "test start failure")
	rp, err := m.GetPod(ctx, pod.UID)
	require.NoError(t, err)
	status, err = m.GetPodStatus(ctx, rp)
	require.NoError(t, err)
	cs := status.FindContainerStatusByName("sidecar")
	require.Equal(t, kubecontainer.ContainerStateCreated, cs.State)
	require.Positive(t, cs.RestartCount)
	_, err = m.SyncTerminatingPod(ctx, pod, status, nil, backOff, deadline)
	require.NoError(t, err)
	require.Equal(t, 2, countCallsOf(fakeRuntime.GetCalls(), "CreateContainer"))
	require.Equal(t, 1, countCallsOf(fakeRuntime.GetCalls(), "RemoveContainer"))
}

func TestSyncTerminatingPodDeadlineStopsUnknownContainer(t *testing.T) {
	ctx, m, _, pod, status, r := setupTerminationTest(t)
	status.ContainerStatuses = []*kubecontainer.Status{{Name: "removed", ID: kubecontainer.ContainerID{ID: "removed"}, State: kubecontainer.ContainerStateRunning}}
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	_, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, time.Now().Add(time.Minute))
	require.ErrorContains(t, err, "missing spec")
	done, err := m.SyncTerminatingPod(ctx, pod, status, nil, backOff, time.Now().Add(-time.Second))
	require.NoError(t, err)
	require.False(t, done)
	call := nextTerminationStop(t, r)
	require.Equal(t, "removed", call.id)
	require.Zero(t, call.grace)
}
