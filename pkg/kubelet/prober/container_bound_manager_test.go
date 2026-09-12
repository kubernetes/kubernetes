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

package prober

import (
	"context"
	"maps"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// testProbeSpec never reaches a threshold, so workers created from it keep
// probing forever and publish nothing. Tests that care about a probe outcome
// override the threshold or the fake prober's answer.
var testProbeSpec = v1.Probe{PeriodSeconds: 1, FailureThreshold: 1000}

func newTestContainerBoundManager(tCtx ktesting.TContext) *containerBoundManager {
	m := NewContainerBoundManager(
		tCtx,
		results.NewManager(),
		results.NewManager(),
		results.NewManager(),
		nil, // runner
		&record.FakeRecorder{},
	).(*containerBoundManager)
	// Don't actually execute probes. Failure keeps workers alive without
	// publishing anything, given testProbeSpec's threshold.
	m.prober.exec = &syncExecProber{fakeExecProber: fakeExecProber{result: probe.Failure}}
	return m
}

// testExecProber returns the fake probe handler installed by
// newTestContainerBoundManager, whose answer can be changed while workers are
// running.
func testExecProber(m *containerBoundManager) *syncExecProber {
	return m.prober.exec.(*syncExecProber)
}

// probedTestPod returns the standard test pod with the given probe types set on
// its container.
func probedTestPod(probeTypes ...probeType) *v1.Pod {
	pod := getTestPod()
	for _, probeType := range probeTypes {
		setTestProbe(pod, probeType, testProbeSpec)
	}
	return pod
}

func testID(id string) kubecontainer.ContainerID {
	return kubecontainer.ContainerID{Type: "test", ID: id}
}

func getContainerBoundWorker(m *containerBoundManager, podUID types.UID, containerName string, probeType probeType) (*containerBoundWorker, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	w, ok := m.workers[probeKey{podUID, containerName, probeType}]
	return w, ok
}

// boundIDs describes which container instance each of a container's probe slots
// is bound to, for compact assertions.
func boundIDs(m *containerBoundManager, pod *v1.Pod, containerName string) map[string]string {
	got := map[string]string{}
	for _, probeType := range allProbeTypes {
		if w, ok := getContainerBoundWorker(m, pod.UID, containerName, probeType); ok {
			got[probeType.String()] = w.containerID.ID
		}
	}
	return got
}

func trackedContainerCount(m *containerBoundManager) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.containers)
}

func probedMetricCount(m *containerBoundManager) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.probedMetrics)
}

func startTestProbes(tCtx ktesting.TContext, m *containerBoundManager, pod *v1.Pod, containerID kubecontainer.ContainerID) {
	m.StartProbes(tCtx, pod, &pod.Spec.Containers[0], containerID, []string{"1.2.3.4"}, time.Now())
}

func mustGetWorker(tCtx ktesting.TContext, m *containerBoundManager, podUID types.UID, containerName string, probeType probeType) *containerBoundWorker {
	tCtx.Helper()
	w, ok := getContainerBoundWorker(m, podUID, containerName, probeType)
	if !ok {
		tCtx.Fatalf("expected %v worker for container %q to be registered", probeType, containerName)
	}
	return w
}

func assertResult(tCtx ktesting.TContext, m *containerBoundManager, id kubecontainer.ContainerID, probeType probeType, want results.Result) {
	tCtx.Helper()
	got, ok := m.resultsManager(probeType).Get(id)
	if !ok || got != want {
		tCtx.Errorf("%v result for %v = (%v, %v), want (%v, true)", probeType, id.ID, got, ok, want)
	}
}

func assertNoResult(tCtx ktesting.TContext, m *containerBoundManager, id kubecontainer.ContainerID, probeTypes ...probeType) {
	tCtx.Helper()
	for _, pt := range probeTypes {
		if got, ok := m.resultsManager(pt).Get(id); ok {
			tCtx.Errorf("%v result for %v unexpectedly survived in cache with value %v", pt, id.ID, got)
		}
	}
}

func assertContainerStatus(tCtx ktesting.TContext, status v1.ContainerStatus, wantStarted bool, wantReady bool) {
	tCtx.Helper()
	if got := status.Started; got == nil || *got != wantStarted {
		tCtx.Errorf("Started = %v, want %v", got, wantStarted)
	}
	if got := status.Ready; got != wantReady {
		tCtx.Errorf("Ready = %v, want %v", got, wantReady)
	}
}

// runtimePodStatus builds the runtime's view of a pod with a single container
// instance, which is what EnsureProbes reconciles against.
func runtimePodStatus(containerName string, containerID kubecontainer.ContainerID, state kubecontainer.State) *kubecontainer.PodStatus {
	return &kubecontainer.PodStatus{
		IPs: []string{"1.2.3.4"},
		ContainerStatuses: []*kubecontainer.Status{{
			Name:      containerName,
			ID:        containerID,
			State:     state,
			StartedAt: time.Now(),
		}},
	}
}

// reportedStatus builds the container status the kubelet last sent to the API
// server, which adoption reconstructs probe state from.
func reportedStatus(containerID string, started *bool, ready bool) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name:        testContainerName,
		ContainerID: containerID,
		Started:     started,
		Ready:       ready,
		State:       v1.ContainerState{Running: &v1.ContainerStateRunning{}},
	}
}

// TestContainerBoundManagerStartProbesIsIdempotent checks that StartProbes is
// idempotent when called multiple times for the same container instance.
func TestContainerBoundManagerStartProbesIsIdempotent(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness, liveness)

		startTestProbes(tCtx, m, pod, testID("a"))
		firstReadiness := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)
		firstLiveness := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness)

		for range 3 {
			startTestProbes(tCtx, m, pod, testID("a"))
		}

		if got := m.workerCount(); got != 2 {
			tCtx.Errorf("worker count = %d, want 2 (one readiness, one liveness)", got)
		}
		if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness); got != firstReadiness {
			tCtx.Error("repeated StartProbes replaced the readiness worker instead of leaving it alone")
		}
		if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness); got != firstLiveness {
			tCtx.Error("repeated StartProbes replaced the liveness worker instead of leaving it alone")
		}
		if got := trackedContainerCount(m); got != 1 {
			tCtx.Errorf("tracked container count = %d, want 1", got)
		}
	})
}

// TestContainerBoundManagerStartProbesReplacesStaleWorkers checks that a new
// container instance takes over its predecessor's slots, and that the results
// cached under the old instance's ID do not linger.
func TestContainerBoundManagerStartProbesReplacesStaleWorkers(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness, liveness)

		startTestProbes(tCtx, m, pod, testID("old"))
		old := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)

		startTestProbes(tCtx, m, pod, testID("new"))

		if old.ctx.Err() == nil {
			tCtx.Error("worker for the replaced container was not stopped")
		}
		want := map[string]string{"Readiness": "new", "Liveness": "new"}
		if got := boundIDs(m, pod, testContainerName); !maps.Equal(got, want) {
			tCtx.Errorf("slots bound to %v, want %v", got, want)
		}
		assertNoResult(tCtx, m, testID("old"), readiness, liveness)
		assertResult(tCtx, m, testID("new"), readiness, results.Failure)
		assertResult(tCtx, m, testID("new"), liveness, results.Success)
		if got := trackedContainerCount(m); got != 1 {
			tCtx.Errorf("tracked container count = %d, want 1", got)
		}
	})
}

// TestContainerBoundManagerLateStopIsIgnored checks that a stop event for a
// container that has already been replaced does not take down its successor.
func TestContainerBoundManagerLateStopIsIgnored(t *testing.T) {
	for _, tc := range []struct {
		name string
		stop func(*containerBoundManager, kubecontainer.ContainerID)
	}{{
		name: "StopProbes(allProbes)",
		stop: func(m *containerBoundManager, id kubecontainer.ContainerID) {
			m.StopProbes(id, allProbes)
		},
	}, {
		name: "StopProbes(liveness|startup)",
		stop: func(m *containerBoundManager, id kubecontainer.ContainerID) {
			m.StopProbes(id, liveness|startup)
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				pod := probedTestPod(readiness, liveness)

				startTestProbes(tCtx, m, pod, testID("old"))
				startTestProbes(tCtx, m, pod, testID("new"))
				currentLiveness := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness)
				currentReadiness := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)

				tc.stop(m, testID("old"))

				if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness); got != currentLiveness {
					tCtx.Error("a stop for the previous container instance took down the current liveness worker")
				}
				if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness); got != currentReadiness {
					tCtx.Error("a stop for the previous container instance took down the current readiness worker")
				}
				if currentLiveness.ctx.Err() != nil {
					tCtx.Error("current liveness worker's probes were cancelled by a stop for the previous instance")
				}
				if currentReadiness.ctx.Err() != nil {
					tCtx.Error("current readiness worker's probes were cancelled by a stop for the previous instance")
				}
				if got := m.workerCount(); got != 2 {
					tCtx.Errorf("worker count = %d, want 2", got)
				}
				assertResult(tCtx, m, testID("new"), readiness, results.Failure)
				assertResult(tCtx, m, testID("new"), liveness, results.Success)
				if got := trackedContainerCount(m); got != 1 {
					tCtx.Errorf("tracked container count = %d, want 1", got)
				}
			})
		})
	}
}

// TestContainerBoundManagerStopProbesLivenessAndStartup checks that StopProbes with
// (liveness|startup) stops liveness and startup workers while leaving readiness running
// and retaining cached results.
func TestContainerBoundManagerStopProbesLivenessAndStartup(t *testing.T) {
	t.Run("while startup probe is running", func(t *testing.T) {
		ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
			m := newTestContainerBoundManager(tCtx)
			defer m.CleanupPods(nil)
			pod := probedTestPod(startup, readiness, liveness)

			startTestProbes(tCtx, m, pod, testID("a"))
			m.StopProbes(testID("a"), liveness|startup)

			if got := m.workerCount(); got != 0 {
				tCtx.Errorf("worker count = %d after StopProbes(liveness|startup), want 0 (startup stopped, readiness not yet started)", got)
			}
			assertResult(tCtx, m, testID("a"), startup, results.Unknown)
			if got := trackedContainerCount(m); got != 1 {
				tCtx.Errorf("tracked container count = %d, want 1", got)
			}
		})
	})

	t.Run("after startup passed", func(t *testing.T) {
		ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
			m := newTestContainerBoundManager(tCtx)
			defer m.CleanupPods(nil)
			pod := probedTestPod(startup, readiness, liveness)

			// Seed startup as passed so readiness and liveness start directly.
			m.startupManager.Seed(testID("a"), results.Success)
			startTestProbes(tCtx, m, pod, testID("a"))
			m.StopProbes(testID("a"), liveness|startup)

			want := map[string]string{"Readiness": "a"}
			if got := boundIDs(m, pod, testContainerName); !maps.Equal(got, want) {
				tCtx.Errorf("slots bound to %v, want %v", got, want)
			}
			// Results survive: the container is still running, and its last known
			// state is still what the pod status should report.
			assertResult(tCtx, m, testID("a"), liveness, results.Success)
			assertResult(tCtx, m, testID("a"), readiness, results.Failure)
			if got := trackedContainerCount(m); got != 1 {
				tCtx.Errorf("tracked container count = %d, want 1", got)
			}
		})
	})
}

// TestContainerBoundManagerStopLivenessAndStartup verifies the backward-compatible
// Manager interface method StopLivenessAndStartup stops liveness and startup workers
// for the entire pod while preserving readiness.
func TestContainerBoundManagerStopLivenessAndStartup(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness, liveness)

		startTestProbes(tCtx, m, pod, testID("a"))
		m.StopLivenessAndStartup(pod)

		want := map[string]string{"Readiness": "a"}
		if got := boundIDs(m, pod, testContainerName); !maps.Equal(got, want) {
			tCtx.Errorf("slots bound to %v, want %v", got, want)
		}
		assertResult(tCtx, m, testID("a"), liveness, results.Success)
		assertResult(tCtx, m, testID("a"), readiness, results.Failure)
		if got := trackedContainerCount(m); got != 1 {
			tCtx.Errorf("tracked container count = %d, want 1", got)
		}
	})
}

// TestContainerBoundManagerStopProbes checks the post-stop teardown: every
// worker for the instance goes away, along with everything cached about it.
func TestContainerBoundManagerStopProbes(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness, liveness)

		startTestProbes(tCtx, m, pod, testID("a"))
		if m.workerCount() != 2 {
			tCtx.Fatalf("worker count = %d before StopProbes, want 2", m.workerCount())
		}
		m.StopProbes(testID("a"), allProbes)

		if got := m.workerCount(); got != 0 {
			tCtx.Errorf("worker count = %d after StopProbes, want 0", got)
		}
		assertNoResult(tCtx, m, testID("a"), allProbeTypes[:]...)
		if got := trackedContainerCount(m); got != 0 {
			tCtx.Errorf("tracked container count = %d after StopProbes(allProbes), want 0", got)
		}
	})
}

// TestContainerBoundManagerStopCancelsInFlightProbe checks that a probe already
// executing when the container starts being killed is aborted rather than left
// running inside a terminating container.
func TestContainerBoundManagerStopCancelsInFlightProbe(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		runner := &hangingRunner{entered: make(chan struct{})}
		m.prober = newProber(runner, &record.FakeRecorder{})
		pod := probedTestPod(liveness)

		startTestProbes(tCtx, m, pod, testID("a"))
		w := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness)

		<-runner.entered
		m.StopProbes(testID("a"), liveness|startup)

		if w.ctx.Err() == nil {
			tCtx.Error("in-flight probe was not cancelled")
		}
		tCtx.Wait()
		assertResult(tCtx, m, testID("a"), liveness, results.Success)
	})
}

// TestContainerBoundManagerRemoveWorkerComparesIdentity checks that a worker
// that exits just as its slot is refilled does not delete its replacement.
func TestContainerBoundManagerRemoveWorkerComparesIdentity(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(liveness)

		startTestProbes(tCtx, m, pod, testID("old"))
		old := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness)
		startTestProbes(tCtx, m, pod, testID("new"))
		replacement := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness)

		// Simulate the stopped worker's goroutine deferred exit hook (onExit) running
		// after a replacement worker has already taken the slot.
		m.removeWorker(old)

		if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness); got != replacement {
			tCtx.Error("an exiting worker deleted the slot belonging to its replacement")
		}
	})
}

// TestContainerBoundManagerRestartStorm checks that multiple containers of a pod
// restarting simultaneously are handled as independent worker replacements.
func TestContainerBoundManagerRestartStorm(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)

		pod := getTestPod()
		pod.Spec.Containers = nil
		names := []string{"c0", "c1", "c2"}
		for _, name := range names {
			container := v1.Container{Name: name, LivenessProbe: testProbeSpec.DeepCopy()}
			container.LivenessProbe.ProbeHandler = v1.ProbeHandler{Exec: &v1.ExecAction{}}
			pod.Spec.Containers = append(pod.Spec.Containers, container)
		}

		start := func(generation string) {
			for i := range pod.Spec.Containers {
				c := &pod.Spec.Containers[i]
				m.StartProbes(tCtx, pod, c, testID(c.Name+"-"+generation), []string{"1.2.3.4"}, time.Now())
			}
		}

		start("v1")
		start("v2")

		if got := m.workerCount(); got != len(names) {
			tCtx.Errorf("worker count = %d, want %d", got, len(names))
		}
		if got := trackedContainerCount(m); got != len(names) {
			tCtx.Errorf("tracked container count = %d, want %d", got, len(names))
		}
		for _, name := range names {
			want := map[string]string{"Liveness": name + "-v2"}
			if got := boundIDs(m, pod, name); !maps.Equal(got, want) {
				tCtx.Errorf("container %s bound to %v, want %v", name, got, want)
			}
			assertNoResult(tCtx, m, testID(name+"-v1"), liveness)
			assertResult(tCtx, m, testID(name+"-v2"), liveness, results.Success)
		}
	})
}

// TestContainerBoundManagerStartupGate checks that readiness and liveness only
// begin once the startup probe has passed, and that they start immediately when
// it does rather than waiting for the next sync.
func TestContainerBoundManagerStartupGate(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(startup, readiness, liveness)

		startTestProbes(tCtx, m, pod, testID("a"))

		want := map[string]string{"Startup": "a"}
		if got := boundIDs(m, pod, testContainerName); !maps.Equal(got, want) {
			tCtx.Fatalf("before startup succeeded, slots bound to %v, want %v", got, want)
		}

		// Let the startup probe pass.
		testExecProber(m).set(probe.Success, nil)
		time.Sleep(2 * time.Second)
		tCtx.Wait()

		want = map[string]string{"Readiness": "a", "Liveness": "a"}
		if got := boundIDs(m, pod, testContainerName); !maps.Equal(got, want) {
			tCtx.Errorf("after startup succeeded, slots bound to %v, want %v", got, want)
		}
		// The startup result outlives its worker: the runtime still reads it to
		// decide the container has started.
		assertResult(tCtx, m, testID("a"), startup, results.Success)
	})
}

// TestContainerBoundManagerStartProbesSkipsStartupWhenAlreadyStarted checks the
// other half of the gate: a container already known to have started gets its
// readiness and liveness workers directly. This is the kubelet-restart case,
// where the startup result is reconstructed before probes are started.
func TestContainerBoundManagerStartProbesSkipsStartupWhenAlreadyStarted(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(startup, readiness, liveness)

		m.startupManager.Seed(testID("a"), results.Success)
		startTestProbes(tCtx, m, pod, testID("a"))

		want := map[string]string{"Readiness": "a", "Liveness": "a"}
		if got := boundIDs(m, pod, testContainerName); !maps.Equal(got, want) {
			tCtx.Errorf("slots bound to %v, want %v", got, want)
		}
	})
}

// TestContainerBoundManagerStartupCallbackAfterKill checks that a startup
// probe that succeeds just as its container is being killed does not bring up
// probes for a container that is on its way out.
func TestContainerBoundManagerStartupCallbackAfterKill(t *testing.T) {
	for _, tc := range []struct {
		name        string
		after       func(*containerBoundManager, *v1.Pod, ktesting.TContext)
		wantWorkers int
	}{{
		name: "container killed",
		after: func(m *containerBoundManager, pod *v1.Pod, tCtx ktesting.TContext) {
			m.StopProbes(testID("a"), allProbes)
		},
		wantWorkers: 0,
	}, {
		name: "container replaced",
		after: func(m *containerBoundManager, pod *v1.Pod, tCtx ktesting.TContext) {
			startTestProbes(tCtx, m, pod, testID("b"))
		},
		wantWorkers: 1, // container "b" startup worker remains
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				pod := probedTestPod(startup, readiness, liveness)

				startTestProbes(tCtx, m, pod, testID("a"))
				w := mustGetWorker(tCtx, m, pod.UID, testContainerName, startup)

				tc.after(m, pod, tCtx)

				// Simulate the startup worker's in-flight probe succeeding and invoking
				// onStartupSucceeded after the container has already been killed or replaced.
				m.onStartupSucceeded(w)

				for _, probeType := range []probeType{readiness, liveness} {
					if got, ok := getContainerBoundWorker(m, pod.UID, testContainerName, probeType); ok && got.containerID == testID("a") {
						tCtx.Errorf("%v worker was created for a container that is gone", probeType)
					}
				}
				if got := m.workerCount(); got != tc.wantWorkers {
					tCtx.Errorf("worker count = %d, want %d", got, tc.wantWorkers)
				}
			})
		})
	}
}

// TestContainerBoundManagerPodTeardown checks the two pod-scoped exits:
// RemovePod for a pod the sync loop knows about, and CleanupPods for one it
// does not.
func TestContainerBoundManagerPodTeardown(t *testing.T) {
	for _, tc := range []struct {
		name     string
		teardown func(*containerBoundManager, *v1.Pod)
	}{{
		name:     "RemovePod",
		teardown: (*containerBoundManager).RemovePod,
	}, {
		name: "CleanupPods",
		teardown: func(m *containerBoundManager, pod *v1.Pod) {
			m.CleanupPods(map[types.UID]sets.Empty{"some-other-pod": {}})
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				pod := probedTestPod(readiness, liveness)

				startTestProbes(tCtx, m, pod, testID("a"))
				if m.workerCount() != 2 {
					tCtx.Fatalf("worker count = %d before teardown, want 2", m.workerCount())
				}
				tc.teardown(m, pod)

				if got := m.workerCount(); got != 0 {
					tCtx.Errorf("worker count = %d after teardown, want 0", got)
				}
				assertNoResult(tCtx, m, testID("a"), readiness, liveness)
				if got := trackedContainerCount(m); got != 0 {
					tCtx.Errorf("tracked container count = %d after teardown, want 0", got)
				}
				if got := probedMetricCount(m); got != 0 {
					tCtx.Errorf("probedMetricCount = %d after teardown, want 0", got)
				}
			})
		})
	}
}

// TestContainerBoundManagerAdoptionSeeding verifies how initial probe results
// (startup and readiness) are seeded when adopting an already-running container
// (such as after a kubelet restart) based on the status reported in the API.
func TestContainerBoundManagerAdoptionSeeding(t *testing.T) {
	started, notStarted := true, false
	for _, tc := range []struct {
		name          string
		reported      *v1.ContainerStatus
		wantStartup   results.Result
		wantReadiness results.Result
	}{{
		name:          "API says started and ready",
		reported:      new(reportedStatus("test://a", &started, true)),
		wantStartup:   results.Success,
		wantReadiness: results.Success,
	}, {
		name:          "API says neither started nor ready",
		reported:      new(reportedStatus("test://a", &notStarted, false)),
		wantStartup:   results.Unknown,
		wantReadiness: results.Failure,
	}, {
		name:          "API says started but not ready",
		reported:      new(reportedStatus("test://a", &started, false)),
		wantStartup:   results.Success,
		wantReadiness: results.Failure,
	}, {
		name: "API status is about a previous container instance",
		// The container restarted while the kubelet was down, so what was
		// reported says nothing about the instance running now.
		reported:      new(reportedStatus("test://previous", &started, true)),
		wantStartup:   results.Unknown,
		wantReadiness: results.Failure,
	}, {
		name: "API status has no container ID yet",
		// The container transitioned before its ID was written to API status;
		// adoption seeds optimistically from reported fields for the running instance.
		reported:      new(reportedStatus("", &started, true)),
		wantStartup:   results.Success,
		wantReadiness: results.Success,
	}, {
		name:          "nothing was ever reported",
		wantStartup:   results.Unknown,
		wantReadiness: results.Failure,
	}, {
		name:          "API reports no Started field",
		reported:      new(reportedStatus("test://a", nil, true)),
		wantStartup:   results.Unknown,
		wantReadiness: results.Success,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)

				pod := probedTestPod(startup, readiness, liveness)
				if tc.reported != nil {
					pod.Status.ContainerStatuses = []v1.ContainerStatus{*tc.reported}
				}

				m.EnsureProbes(tCtx, pod, runtimePodStatus(testContainerName, testID("a"), kubecontainer.ContainerStateRunning))

				assertResult(tCtx, m, testID("a"), startup, tc.wantStartup)
				assertResult(tCtx, m, testID("a"), readiness, tc.wantReadiness)
				// Never kill a container over anything that happened
				// before the restart.
				assertResult(tCtx, m, testID("a"), liveness, results.Success)

				// Seeding the cache during adoption restores known state rather than
				// recording a new transition, so it must not emit updates to the sync loop.
				for _, probeType := range allProbeTypes {
					select {
					case update := <-m.resultsManager(probeType).Updates():
						tCtx.Errorf("adoption emitted a %v update %v; seeded results must not be announced", probeType, update)
					default:
					}
				}
			})
		})
	}
}

// TestContainerBoundManagerKubeletRestartDoesNotFlapStatus is the whole point of
// adoption seeding: bringing the kubelet back up must not change what the pod's
// status says about its containers.
func TestContainerBoundManagerKubeletRestartDoesNotFlapStatus(t *testing.T) {
	started, notStarted := true, false
	for _, tc := range []struct {
		name        string
		probeTypes  []probeType
		started     *bool
		ready       bool
		wantStarted bool
		wantReady   bool
	}{{
		name:        "ready pod stays ready",
		probeTypes:  []probeType{startup, readiness},
		started:     &started,
		ready:       true,
		wantStarted: true,
		wantReady:   true,
	}, {
		name:        "not-ready pod stays not ready",
		probeTypes:  []probeType{startup, readiness},
		started:     &started,
		ready:       false,
		wantStarted: true,
		wantReady:   false,
	}, {
		name:        "container that had not started yet stays not started",
		probeTypes:  []probeType{startup, readiness},
		started:     &notStarted,
		ready:       false,
		wantStarted: false,
		wantReady:   false,
	}, {
		name:        "readiness-only ready container stays ready",
		probeTypes:  []probeType{readiness},
		started:     &started,
		ready:       true,
		wantStarted: true,
		wantReady:   true,
	}, {
		name:        "readiness-only not-ready container stays not ready",
		probeTypes:  []probeType{readiness},
		started:     &started,
		ready:       false,
		wantStarted: true,
		wantReady:   false,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				// A fresh manager, as after a kubelet restart: it has never
				// seen this pod and has nothing cached.
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)

				pod := probedTestPod(tc.probeTypes...)
				pod.Status.ContainerStatuses = []v1.ContainerStatus{reportedStatus("test://a", tc.started, tc.ready)}

				m.EnsureProbes(tCtx, pod, runtimePodStatus(testContainerName, testID("a"), kubecontainer.ContainerStateRunning))

				status := &v1.PodStatus{ContainerStatuses: []v1.ContainerStatus{reportedStatus("test://a", nil, false)}}
				m.UpdatePodStatus(tCtx, pod, status)

				assertContainerStatus(tCtx, status.ContainerStatuses[0], tc.wantStarted, tc.wantReady)
			})
		})
	}
}

// TestContainerBoundManagerEnsureProbesStopsDeadContainers checks that EnsureProbes
// stops probing a container that exited on its own or disappeared from runtime status.
func TestContainerBoundManagerEnsureProbesStopsDeadContainers(t *testing.T) {
	for _, tc := range []struct {
		name  string
		state *kubecontainer.State
	}{
		{name: "exited", state: new(kubecontainer.ContainerStateExited)},
		{name: "created but never started", state: new(kubecontainer.ContainerStateCreated)},
		{name: "unknown", state: new(kubecontainer.ContainerStateUnknown)},
		{name: "container missing from runtime status", state: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				pod := probedTestPod(readiness, liveness)

				startTestProbes(tCtx, m, pod, testID("a"))
				if m.workerCount() != 2 {
					tCtx.Fatalf("worker count = %d before reconcile, want 2", m.workerCount())
				}

				var podStatus *kubecontainer.PodStatus
				if tc.state != nil {
					podStatus = runtimePodStatus(testContainerName, testID("a"), *tc.state)
				} else {
					podStatus = &kubecontainer.PodStatus{IPs: []string{"1.2.3.4"}}
				}
				m.EnsureProbes(tCtx, pod, podStatus)

				if got := m.workerCount(); got != 0 {
					tCtx.Errorf("worker count = %d, want 0", got)
				}
				assertNoResult(tCtx, m, testID("a"), readiness, liveness)
				if got := trackedContainerCount(m); got != 0 {
					tCtx.Errorf("tracked container count = %d, want 0", got)
				}
			})
		})
	}
}

// TestContainerBoundManagerEnsureProbesNilPodStatusIsNoop verifies that EnsureProbes
// safely does nothing when runtime pod status is nil.
func TestContainerBoundManagerEnsureProbesNilPodStatusIsNoop(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness)

		m.EnsureProbes(tCtx, pod, nil)
		if got := m.workerCount(); got != 0 {
			tCtx.Errorf("worker count = %d, want 0", got)
		}
	})
}

// TestContainerBoundManagerEnsureProbesAdoptsMissedStart checks that a container
// that started without its hook firing, or one that crashed immediately after
// it did, is properly reconciled by the next sync.
func TestContainerBoundManagerEnsureProbesAdoptsMissedStart(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness, liveness)

		// The hook never fired for this container; reconciliation adopts it.
		m.EnsureProbes(tCtx, pod, runtimePodStatus(testContainerName, testID("a"), kubecontainer.ContainerStateRunning))
		want := map[string]string{"Readiness": "a", "Liveness": "a"}
		if got := boundIDs(m, pod, testContainerName); !maps.Equal(got, want) {
			tCtx.Fatalf("slots bound to %v, want %v", got, want)
		}
		adoptedReadiness := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)
		adoptedLiveness := mustGetWorker(tCtx, m, pod.UID, testContainerName, liveness)

		// The container crashed and came back while probes for the old
		// instance were still running.
		m.EnsureProbes(tCtx, pod, runtimePodStatus(testContainerName, testID("b"), kubecontainer.ContainerStateRunning))

		if adoptedReadiness.ctx.Err() == nil {
			tCtx.Error("readiness worker for the previous instance kept running")
		}
		if adoptedLiveness.ctx.Err() == nil {
			tCtx.Error("liveness worker for the previous instance kept running")
		}
		want = map[string]string{"Readiness": "b", "Liveness": "b"}
		if got := boundIDs(m, pod, testContainerName); !maps.Equal(got, want) {
			tCtx.Errorf("slots bound to %v, want %v", got, want)
		}
		assertNoResult(tCtx, m, testID("a"), readiness, liveness)
		assertResult(tCtx, m, testID("b"), readiness, results.Failure)
		assertResult(tCtx, m, testID("b"), liveness, results.Success)
	})
}

// TestContainerBoundManagerWorkersSurviveCallerContextCancellation verifies that
// probe workers are bound to the manager's lifetime and survive cancellation of
// the pod sync context passed into StartProbes and EnsureProbes.
func TestContainerBoundManagerWorkersSurviveCallerContextCancellation(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness, liveness)

		callCtx, cancel := context.WithCancel(tCtx)
		m.StartProbes(callCtx, pod, &pod.Spec.Containers[0], testID("a"), []string{"1.2.3.4"}, time.Now())
		cancel()

		w := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)
		if err := w.ctx.Err(); err != nil {
			tCtx.Errorf("worker context was canceled when caller sync context was canceled: %v", err)
		}
	})
}

// TestContainerBoundManagerLevelAndEdgeAgree checks ordering: EnsureProbes state
// reconciliation runs before runtime sync, ensuring a subsequent StartProbes
// hook is a no-op for containers already adopted during reconciliation.
func TestContainerBoundManagerLevelAndEdgeAgree(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness)

		m.EnsureProbes(tCtx, pod, runtimePodStatus(testContainerName, testID("a"), kubecontainer.ContainerStateRunning))
		adopted := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)

		startTestProbes(tCtx, m, pod, testID("a"))

		if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness); got != adopted {
			tCtx.Error("the start hook replaced or dropped a worker that reconciliation had already created")
		}
		if got := m.workerCount(); got != 1 {
			tCtx.Errorf("worker count = %d, want 1", got)
		}
	})
}

// TestContainerBoundManagerEnsureProbesKeepsStartedContainer checks that a
// container whose only probe was a startup probe, which has already passed and
// whose worker is therefore gone, is not mistaken for one that needs adopting
// and re-probed on every sync.
func TestContainerBoundManagerEnsureProbesKeepsStartedContainer(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(startup)
		runtimeStatus := runtimePodStatus(testContainerName, testID("a"), kubecontainer.ContainerStateRunning)

		testExecProber(m).set(probe.Success, nil)
		startTestProbes(tCtx, m, pod, testID("a"))
		tCtx.Wait()
		if got := m.workerCount(); got != 0 {
			tCtx.Fatalf("worker count = %d after the startup probe passed, want 0", got)
		}

		// The API has not caught up yet, so the pod object still says the
		// container has not started. Reconciliation must not take that as
		// license to re-run the startup probe.
		m.EnsureProbes(tCtx, pod, runtimeStatus)

		if got := m.workerCount(); got != 0 {
			tCtx.Errorf("worker count = %d, want 0: a started container was re-adopted", got)
		}
		assertResult(tCtx, m, testID("a"), startup, results.Success)
	})
}

// TestContainerBoundManagerUpdatePodStatus verifies how container Started and Ready
// conditions in PodStatus are computed from cached probe results and container running state.
func TestContainerBoundManagerUpdatePodStatus(t *testing.T) {
	for _, tc := range []struct {
		name        string
		probeTypes  []probeType
		running     bool
		startup     *results.Result
		readiness   *results.Result
		wantStarted bool
		wantReady   bool
	}{{
		name:        "no probes: running is started and ready",
		running:     true,
		wantStarted: true,
		wantReady:   true,
	}, {
		name:        "no probes but not running",
		running:     false,
		wantStarted: false,
		wantReady:   false,
	}, {
		name:        "startup probe with no result yet",
		probeTypes:  []probeType{startup},
		running:     true,
		startup:     new(results.Unknown),
		wantStarted: false,
		wantReady:   false,
	}, {
		name:        "startup probe passed, no readiness probe",
		probeTypes:  []probeType{startup},
		running:     true,
		startup:     new(results.Success),
		wantStarted: true,
		wantReady:   true,
	}, {
		name:        "startup probe failed",
		probeTypes:  []probeType{startup},
		running:     true,
		startup:     new(results.Failure),
		wantStarted: false,
		wantReady:   false,
	}, {
		name:        "readiness probe not passing",
		probeTypes:  []probeType{readiness},
		running:     true,
		readiness:   new(results.Failure),
		wantStarted: true,
		wantReady:   false,
	}, {
		name:        "readiness probe passing",
		probeTypes:  []probeType{readiness},
		running:     true,
		readiness:   new(results.Success),
		wantStarted: true,
		wantReady:   true,
	}, {
		name:        "readiness passing but startup has not",
		probeTypes:  []probeType{startup, readiness},
		running:     true,
		startup:     new(results.Unknown),
		readiness:   new(results.Success),
		wantStarted: false,
		wantReady:   false,
	}, {
		name:        "startup probe passed, readiness probe not passing",
		probeTypes:  []probeType{startup, readiness},
		running:     true,
		startup:     new(results.Success),
		readiness:   new(results.Failure),
		wantStarted: true,
		wantReady:   false,
	}, {
		name:        "startup probe with no cached result",
		probeTypes:  []probeType{startup},
		running:     true,
		wantStarted: false,
		wantReady:   false,
	}, {
		name:        "probes present but container not running",
		probeTypes:  []probeType{startup, readiness},
		running:     false,
		startup:     new(results.Success),
		readiness:   new(results.Success),
		wantStarted: false,
		wantReady:   false,
	}, {
		name:       "readiness probe with no cached result",
		probeTypes: []probeType{readiness},
		running:    true,
		// Defensive check: workers seed the cache with Failure upon creation,
		// making an empty cache impossible in practice. Verify that an uncached
		// readiness probe still fails closed (Ready=false).
		wantStarted: true,
		wantReady:   false,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				pod := probedTestPod(tc.probeTypes...)

				if tc.startup != nil {
					m.startupManager.Seed(testID("a"), *tc.startup)
				}
				if tc.readiness != nil {
					m.readinessManager.Seed(testID("a"), *tc.readiness)
				}

				containerStatus := v1.ContainerStatus{
					Name:        testContainerName,
					ContainerID: "test://a",
					Ready:       !tc.wantReady,
				}
				if tc.running {
					containerStatus.State.Running = &v1.ContainerStateRunning{}
				} else {
					containerStatus.State.Terminated = &v1.ContainerStateTerminated{}
				}
				status := &v1.PodStatus{ContainerStatuses: []v1.ContainerStatus{containerStatus}}

				m.UpdatePodStatus(tCtx, pod, status)

				assertContainerStatus(tCtx, status.ContainerStatuses[0], tc.wantStarted, tc.wantReady)
			})
		})
	}
}

// TestContainerBoundManagerUpdatePodStatusInitContainers checks that a plain init
// container is reported Ready once it has exited successfully, independent of
// any probe.
func TestContainerBoundManagerUpdatePodStatusInitContainers(t *testing.T) {
	for _, tc := range []struct {
		name      string
		state     v1.ContainerState
		initReady bool
		wantReady bool
	}{{
		name:      "successful plain init container becomes ready",
		state:     v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0}},
		initReady: false,
		wantReady: true,
	}, {
		name:      "failed plain init container resets ready to false",
		state:     v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 1}},
		initReady: true,
		wantReady: false,
	}, {
		name:      "running plain init container resets ready to false",
		state:     v1.ContainerState{Running: &v1.ContainerStateRunning{}},
		initReady: true,
		wantReady: false,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				pod := getTestPod()
				pod.Spec.InitContainers = []v1.Container{{Name: "init"}}

				status := &v1.PodStatus{InitContainerStatuses: []v1.ContainerStatus{{
					Name:        "init",
					ContainerID: "test://init",
					State:       tc.state,
					Ready:       tc.initReady,
				}}}
				m.UpdatePodStatus(tCtx, pod, status)

				if got := status.InitContainerStatuses[0].Ready; got != tc.wantReady {
					tCtx.Errorf("Ready = %v, want %v", got, tc.wantReady)
				}
			})
		})
	}
}

// TestContainerBoundManagerRestartableInitContainers tests that restartable init
// containers (sidecars) have probes tracked and conditions reported like regular
// containers, while plain init containers do not.
func TestContainerBoundManagerRestartableInitContainers(t *testing.T) {
	restartPolicyAlways := v1.ContainerRestartPolicyAlways

	t.Run("EnsureProbes tracks restartable init container with probes", func(t *testing.T) {
		ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
			m := newTestContainerBoundManager(tCtx)
			defer m.CleanupPods(nil)

			pod := getTestPod()
			plainInit := v1.Container{Name: "plain-init", ReadinessProbe: testProbeSpec.DeepCopy()}
			sidecar := v1.Container{Name: "sidecar", RestartPolicy: &restartPolicyAlways, ReadinessProbe: testProbeSpec.DeepCopy()}
			pod.Spec.InitContainers = []v1.Container{plainInit, sidecar}

			status := &kubecontainer.PodStatus{
				IPs: []string{"1.2.3.4"},
				ContainerStatuses: []*kubecontainer.Status{
					{Name: "plain-init", ID: testID("plain"), State: kubecontainer.ContainerStateRunning, StartedAt: time.Now()},
					{Name: "sidecar", ID: testID("sidecar"), State: kubecontainer.ContainerStateRunning, StartedAt: time.Now()},
				},
			}

			m.EnsureProbes(tCtx, pod, status)

			// Plain init container must not be probed; sidecar must be probed.
			if _, ok := getContainerBoundWorker(m, pod.UID, "plain-init", readiness); ok {
				tCtx.Error("plain init container worker was created; only restartable init containers should be probed")
			}
			mustGetWorker(tCtx, m, pod.UID, "sidecar", readiness)
		})
	})

	t.Run("UpdatePodStatus uses probe results for restartable init container", func(t *testing.T) {
		ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
			m := newTestContainerBoundManager(tCtx)
			defer m.CleanupPods(nil)

			pod := getTestPod()
			sidecar := v1.Container{Name: "sidecar", RestartPolicy: &restartPolicyAlways, ReadinessProbe: testProbeSpec.DeepCopy()}
			pod.Spec.InitContainers = []v1.Container{sidecar}

			m.readinessManager.Seed(testID("sidecar"), results.Failure)

			status := &v1.PodStatus{InitContainerStatuses: []v1.ContainerStatus{{
				Name:        "sidecar",
				ContainerID: "test://sidecar",
				State:       v1.ContainerState{Running: &v1.ContainerStateRunning{}},
			}}}

			m.UpdatePodStatus(tCtx, pod, status)
			assertContainerStatus(tCtx, status.InitContainerStatuses[0], true, false)

			m.readinessManager.Seed(testID("sidecar"), results.Success)
			m.UpdatePodStatus(tCtx, pod, status)
			assertContainerStatus(tCtx, status.InitContainerStatuses[0], true, true)
		})
	})
}

// TestContainerBoundManagerDoesNotReprobeDoomedContainers checks that a
// container whose liveness or startup probe has already failed past its
// threshold is not probed again before the sync loop gets to kill it. Probing
// again could overwrite the very verdict the sync loop is about to act on.
func TestContainerBoundManagerDoesNotReprobeDoomedContainers(t *testing.T) {
	for _, probeType := range []probeType{liveness, startup} {
		t.Run(probeType.String(), func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				pod := probedTestPod(probeType)

				// The probe has just failed past its threshold, so its worker
				// recorded the verdict and exited.
				setTestProbe(pod, probeType, v1.Probe{})
				startTestProbes(tCtx, m, pod, testID("a"))
				tCtx.Wait()

				assertResult(tCtx, m, testID("a"), probeType, results.Failure)
				if got := m.workerCount(); got != 0 {
					tCtx.Fatalf("worker count = %d after the verdict, want 0", got)
				}

				// A sync lands before the container has been killed.
				m.EnsureProbes(tCtx, pod, runtimePodStatus(testContainerName, testID("a"), kubecontainer.ContainerStateRunning))
				testExecProber(m).set(probe.Success, nil)
				time.Sleep(5 * time.Second)
				tCtx.Wait()

				if got := m.workerCount(); got != 0 {
					tCtx.Errorf("worker count = %d, want 0: a doomed container was probed again", got)
				}
				assertResult(tCtx, m, testID("a"), probeType, results.Failure)
			})
		})
	}
}

// TestContainerBoundManagerCleansUpStartupOnlyContainerOnRestart verifies that
// restarting a container whose startup probe already succeeded still evicts the
// old instance's cached results, even though startup workers exit on success and
// leave no active workers in the manager.
func TestContainerBoundManagerCleansUpStartupOnlyContainerOnRestart(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(startup)

		testExecProber(m).set(probe.Success, nil)
		startTestProbes(tCtx, m, pod, testID("a"))
		tCtx.Wait()

		assertResult(tCtx, m, testID("a"), startup, results.Success)
		if got := m.workerCount(); got != 0 {
			tCtx.Fatalf("worker count = %d, want 0 after startup success", got)
		}

		// Container restarts as 'b'.
		startTestProbes(tCtx, m, pod, testID("b"))
		tCtx.Wait()

		assertNoResult(tCtx, m, testID("a"), startup)
		assertResult(tCtx, m, testID("b"), startup, results.Success)
		if got := trackedContainerCount(m); got != 1 {
			tCtx.Errorf("tracked container count = %d, want 1", got)
		}
	})
}

// TestContainerBoundManagerPodTeardownWhenNoWorkersSurvive verifies that pod teardown
// (both RemovePod and CleanupPods) clears cached results, tracked containers, and
// metric series even if all workers have already exited after passing startup.
func TestContainerBoundManagerPodTeardownWhenNoWorkersSurvive(t *testing.T) {
	for _, tc := range []struct {
		name     string
		teardown func(*containerBoundManager, *v1.Pod)
	}{{
		name:     "RemovePod",
		teardown: (*containerBoundManager).RemovePod,
	}, {
		name: "CleanupPods",
		teardown: func(m *containerBoundManager, pod *v1.Pod) {
			m.CleanupPods(map[types.UID]sets.Empty{"some-other-pod": {}})
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				pod := probedTestPod(startup)

				testExecProber(m).set(probe.Success, nil)
				startTestProbes(tCtx, m, pod, testID("a"))
				tCtx.Wait()

				assertResult(tCtx, m, testID("a"), startup, results.Success)
				if got := m.workerCount(); got != 0 {
					tCtx.Fatalf("worker count = %d, want 0 after startup success", got)
				}
				if got := probedMetricCount(m); got == 0 {
					tCtx.Fatal("expected probedMetrics to be recorded for startup probe")
				}
				if got := trackedContainerCount(m); got != 1 {
					tCtx.Fatalf("tracked container count = %d, want 1", got)
				}

				tc.teardown(m, pod)

				assertNoResult(tCtx, m, testID("a"), startup)
				if got := probedMetricCount(m); got != 0 {
					tCtx.Errorf("probedMetrics count = %d, want 0 after %s", got, tc.name)
				}
				if got := trackedContainerCount(m); got != 0 {
					tCtx.Errorf("tracked container count = %d, want 0 after %s", got, tc.name)
				}
			})
		})
	}
}
