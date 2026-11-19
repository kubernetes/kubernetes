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
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/utils/exec"
)

func enableMutableProbes(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerScopedProbes, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)
}

func TestMutableBoundProbeLifecycle(t *testing.T) {
	enableMutableProbes(t)
	for _, typ := range []probeType{readiness, liveness, startup} {
		t.Run(typ.String(), func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				pod := probedTestPod(typ)
				id := testID("mutable")
				status := runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning)
				startTestProbes(tCtx, m, pod, id)
				w := mustGetWorker(tCtx, m, pod.UID, testContainerName, typ)
				tCtx.Wait()
				updated := pod.DeepCopy()
				spec := probeSpec(typ, &updated.Spec.Containers[0])
				spec.TimeoutSeconds = 2
				spec.FailureThreshold = 2
				m.SyncPod(tCtx, updated, status)
				if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, typ); got != w {
					tCtx.Fatal("configuration update replaced the worker")
				}
				tCtx.Wait()
				assertResult(tCtx, m, id, typ, initialValue(typ))
				time.Sleep(time.Second)
				tCtx.Wait()
				assertResult(tCtx, m, id, typ, results.Failure)
				old := <-m.resultsManager(typ).Updates()
				removed := pod.DeepCopy()
				removed.Spec.Containers[0].ReadinessProbe = nil
				removed.Spec.Containers[0].LivenessProbe = nil
				removed.Spec.Containers[0].StartupProbe = nil
				m.SyncPod(tCtx, removed, status)
				assertNoResult(tCtx, m, id, typ)
				if m.resultsManager(typ).IsCurrent(old) {
					tCtx.Fatal("removed probe result remains current")
				}
				m.SyncPod(tCtx, pod, status)
				if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, typ); got != w {
					tCtx.Fatal("reenabling replaced the worker")
				}
				want := initialValue(typ)
				if typ == startup {
					want = results.Success
				}
				assertResult(tCtx, m, id, typ, want)
				replacement := testID("replacement")
				m.SyncPod(tCtx, pod, runtimePodStatus(testContainerName, replacement, kubecontainer.ContainerStateRunning))
				if got := mustGetWorker(tCtx, m, pod.UID, testContainerName, typ); got == w || got.containerID != replacement {
					tCtx.Fatal("replacement did not get its own worker")
				}
				assertNoResult(tCtx, m, id, typ)
			})
		})
	}
}

func TestMutableStopProbesRetainsResultsUntilReconciliation(t *testing.T) {
	enableMutableProbes(t)
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness, liveness)
		id := testID("stopped")
		startTestProbes(tCtx, m, pod, id)
		m.StopProbes(id, kubecontainer.AllProbes)
		tCtx.Wait()

		assertResult(tCtx, m, id, readiness, results.Failure)
		assertResult(tCtx, m, id, liveness, results.Success)
		key := containerKey{pod.UID, testContainerName}
		if target, ok := m.targets[key]; !ok || target.containerID != id {
			tCtx.Fatal("stopping probes discarded the tracked mutable target")
		}

		m.SyncPod(tCtx, pod, runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateExited))
		assertNoResult(tCtx, m, id, readiness, liveness)
		if _, ok := m.targets[key]; ok {
			tCtx.Fatal("reconciliation retained the exited container's mutable target")
		}
		if got := trackedContainerCount(m); got != 0 {
			tCtx.Fatalf("tracked container count after exit = %d, want 0", got)
		}
	})
}

func TestMutableStartupUsesLatestConfiguration(t *testing.T) {
	enableMutableProbes(t)
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(startup, readiness, liveness)
		id := testID("startup")
		status := runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning)
		startTestProbes(tCtx, m, pod, id)
		tCtx.Wait()
		updated := pod.DeepCopy()
		updated.Spec.Containers[0].ReadinessProbe.TimeoutSeconds = 23
		m.SyncPod(tCtx, updated, status)
		// A repeated runtime hook must not replace the reconciled configuration.
		startTestProbes(tCtx, m, pod, id)
		testExecProber(m).set(probe.Success, nil)
		time.Sleep(time.Second)
		tCtx.Wait()
		w := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)
		if snapshotMutable(w).spec.TimeoutSeconds != 23 {
			tCtx.Fatal("startup activated an obsolete readiness configuration")
		}
		completed := mustGetWorker(tCtx, m, pod.UID, testContainerName, startup)
		updated = updated.DeepCopy()
		updated.Spec.Containers[0].StartupProbe.TimeoutSeconds = 42
		m.SyncPod(tCtx, updated, status)
		if !snapshotMutable(completed).hold {
			tCtx.Fatal("configuration update reactivated completed startup")
		}
		m.StopLivenessAndStartup(updated)
		// A startup callback arriving after termination must not restart probing.
		m.onStartupSucceeded(completed)
		for _, typ := range []probeType{startup, liveness} {
			if _, ok := getContainerBoundWorker(m, pod.UID, testContainerName, typ); ok {
				tCtx.Fatalf("late startup callback left a %v worker after termination", typ)
			}
		}
		if got, ok := getContainerBoundWorker(m, pod.UID, testContainerName, readiness); !ok || got != w {
			tCtx.Fatal("termination stopped or replaced readiness worker")
		}
	})
}

type mutableHeldCall struct {
	canceled <-chan struct{}
	command  string
}
type mutableHeldRunner struct {
	calls   chan mutableHeldCall
	release chan struct{}
}

func (r *mutableHeldRunner) RunInContainer(ctx context.Context, _ kubecontainer.ContainerID, command []string, _ time.Duration) ([]byte, error) {
	r.calls <- mutableHeldCall{ctx.Done(), command[0]}
	<-r.release
	return []byte("ok"), nil
}

func TestMutableBoundCancellation(t *testing.T) {
	enableMutableProbes(t)
	for _, action := range []string{"handler", "remove", "replace", "stop"} {
		t.Run(action, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				runner := &mutableHeldRunner{make(chan mutableHeldCall, 10), make(chan struct{})}
				m.prober = newProber(runner, &record.FakeRecorder{})
				pod := probedTestPod(readiness)
				pod.Spec.Containers[0].ReadinessProbe.Exec.Command = []string{"old"}
				id := testID("held")
				status := runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning)
				startTestProbes(tCtx, m, pod, id)
				call := <-runner.calls
				updated := pod.DeepCopy()
				switch action {
				case "handler":
					updated.Spec.Containers[0].ReadinessProbe.Exec.Command = []string{"new"}
					m.SyncPod(tCtx, updated, status)
				case "remove":
					updated.Spec.Containers[0].ReadinessProbe = nil
					m.SyncPod(tCtx, updated, status)
					m.SyncPod(tCtx, pod, status)
				case "replace":
					m.SyncPod(tCtx, updated, runtimePodStatus(testContainerName, testID("new"), kubecontainer.ContainerStateRunning))
				case "stop":
					m.RemovePod(pod)
				}
				<-call.canceled
				tCtx.Wait()
				if action == "handler" || action == "remove" {
					if len(runner.calls) != 0 {
						tCtx.Fatal("configuration change overlapped a cancelled call")
					}
				}
				m.RemovePod(pod)
				close(runner.release)
				tCtx.Wait()
				assertNoResult(tCtx, m, id, readiness)
				assertNoResult(tCtx, m, testID("new"), readiness)
			})
		})
	}
}

func TestMutableBoundSchedule(t *testing.T) {
	enableMutableProbes(t)
	for _, change := range []string{"period", "initial delay", "handler"} {
		t.Run(change, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				pod := probedTestPod(readiness)
				spec := pod.Spec.Containers[0].ReadinessProbe
				spec.PeriodSeconds = 100
				if change == "initial delay" {
					spec.InitialDelaySeconds = 100
				}
				id := testID("schedule")
				startTestProbes(tCtx, m, pod, id)
				tCtx.Wait()
				w := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)
				before := snapshotMutable(w).completed
				time.Sleep(2 * time.Second)
				updated := pod.DeepCopy()
				spec = updated.Spec.Containers[0].ReadinessProbe
				switch change {
				case "period":
					spec.PeriodSeconds = 3
				case "initial delay":
					spec.InitialDelaySeconds = 3
				case "handler":
					spec.TimeoutSeconds = 2
					spec.PeriodSeconds = 200
				}
				m.SyncPod(tCtx, updated, runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning))
				if change != "handler" {
					time.Sleep(time.Second)
				}
				tCtx.Wait()
				if !snapshotMutable(w).completed.After(before) {
					tCtx.Fatalf("%s update did not reschedule probe", change)
				}
			})
		})
	}
}

func TestMutableBoundFeatureGate(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprint(enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerScopedProbes, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, enabled)
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				// Keep the container tracked when readiness is removed, even with mutable probes disabled.
				pod := probedTestPod(readiness, liveness)
				id := testID("gate")
				startTestProbes(tCtx, m, pod, id)
				updated := pod.DeepCopy()
				updated.Spec.Containers[0].ReadinessProbe = nil
				m.SyncPod(tCtx, updated, runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning))
				_, cached := m.readinessManager.Get(id)
				if cached == enabled {
					tCtx.Fatalf("cached result=%v with mutable gate=%v", cached, enabled)
				}
			})
		})
	}
}

func TestMutableBoundPreservesTerminalFailure(t *testing.T) {
	enableMutableProbes(t)
	for _, typ := range []probeType{startup, liveness} {
		t.Run(typ.String(), func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				m := newTestContainerBoundManager(tCtx)
				defer m.CleanupPods(nil)
				pod := probedTestPod(typ)
				probeSpec(typ, &pod.Spec.Containers[0]).FailureThreshold = 1
				id := testID("failed")
				startTestProbes(tCtx, m, pod, id)
				tCtx.Wait()
				update := <-m.resultsManager(typ).Updates()
				changed := pod.DeepCopy()
				probeSpec(typ, &changed.Spec.Containers[0]).TimeoutSeconds = 12
				m.SyncPod(tCtx, changed, runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning))
				tCtx.Wait()
				if !m.resultsManager(typ).IsCurrent(update) {
					tCtx.Fatal("handler update discarded the published terminal failure")
				}
				w := mustGetWorker(tCtx, m, pod.UID, testContainerName, typ)
				if !snapshotMutable(w).hold {
					tCtx.Fatal("handler update resumed a terminal probe")
				}
			})
		})
	}
}

func TestMutableBoundThresholdReset(t *testing.T) {
	enableMutableProbes(t)
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(readiness)
		pod.Spec.Containers[0].ReadinessProbe.FailureThreshold = 3
		id := testID("threshold")
		startTestProbes(tCtx, m, pod, id)
		tCtx.Wait()
		status := runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning)
		changed := pod.DeepCopy()
		changed.Spec.Containers[0].ReadinessProbe.FailureThreshold = 2
		m.SyncPod(tCtx, changed, status)
		m.SyncPod(tCtx, pod, status)
		w := mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)
		if snapshotMutable(w).consecutive != 0 {
			tCtx.Fatal("intermediate threshold update did not reset the streak")
		}
		time.Sleep(time.Second)
		tCtx.Wait()
		changed = pod.DeepCopy()
		changed.Spec.Containers[0].ReadinessProbe.InitialDelaySeconds = 1000
		m.SyncPod(tCtx, changed, status)
		time.Sleep(time.Second)
		tCtx.Wait()
		if snapshotMutable(w).consecutive != 2 {
			tCtx.Fatal("initial delay was reapplied after the first attempt")
		}
	})
}

func TestMutableBoundSidecarAdoption(t *testing.T) {
	enableMutableProbes(t)
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod(startup, readiness)
		sidecar := pod.Spec.Containers[0]
		restart := v1.ContainerRestartPolicyAlways
		sidecar.RestartPolicy = &restart
		pod.Spec.InitContainers = []v1.Container{sidecar}
		pod.Spec.Containers = nil
		id := testID("sidecar")
		started := true
		pod.Status.InitContainerStatuses = []v1.ContainerStatus{reportedStatus(id.String(), &started, true)}
		pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}
		status := runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning)
		m.SyncPod(tCtx, pod, status)
		assertResult(tCtx, m, id, startup, results.Success)
		assertResult(tCtx, m, id, readiness, results.Success)
		updated := pod.DeepCopy()
		updated.Spec.InitContainers[0].ReadinessProbe = nil
		m.SyncPod(tCtx, updated, status)
		assertNoResult(tCtx, m, id, readiness)
		updated = updated.DeepCopy()
		updated.Status.InitContainerStatuses[0].State.Running = &v1.ContainerStateRunning{}
		m.UpdatePodStatus(tCtx, updated, &pod.Status, &updated.Status)
		assertContainerStatus(tCtx, updated.Status.InitContainerStatuses[0], true, true)
		updated.Spec.InitContainers[0].ReadinessProbe = pod.Spec.InitContainers[0].ReadinessProbe.DeepCopy()
		m.SyncPod(tCtx, updated, status)
		assertResult(tCtx, m, id, readiness, results.Failure)
	})
}

type mutableTimedProber struct{ calls chan time.Time }

func (p *mutableTimedProber) Probe(exec.Cmd) (probe.Result, string, error) {
	p.calls <- time.Now()
	time.Sleep(3 * time.Second)
	return probe.Success, "", nil
}

func TestMutableBoundPeriodStartsAfterCompletion(t *testing.T) {
	enableMutableProbes(t)
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		p := &mutableTimedProber{calls: make(chan time.Time, 2)}
		m.prober.exec = p
		pod := probedTestPod(readiness)
		pod.Spec.Containers[0].ReadinessProbe.PeriodSeconds = 10
		startTestProbes(tCtx, m, pod, testID("timing"))
		first, second := <-p.calls, <-p.calls
		if got := second.Sub(first); got != 13*time.Second {
			tCtx.Fatalf("probe interval=%v, want 13s", got)
		}
		time.Sleep(3 * time.Second)
		tCtx.Wait()
	})
}

func TestMutableStartupAdditionAppliesOnNextContainer(t *testing.T) {
	enableMutableProbes(t)
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		m := newTestContainerBoundManager(tCtx)
		defer m.CleanupPods(nil)
		pod := probedTestPod()
		id := testID("started")
		startTestProbes(tCtx, m, pod, id)
		updated := pod.DeepCopy()
		setTestProbe(updated, startup, testProbeSpec)
		setTestProbe(updated, readiness, testProbeSpec)
		m.SyncPod(tCtx, updated, runtimePodStatus(testContainerName, id, kubecontainer.ContainerStateRunning))
		assertResult(tCtx, m, id, startup, results.Success)
		if _, ok := getContainerBoundWorker(m, pod.UID, testContainerName, startup); ok {
			tCtx.Fatal("adding startup probed an already started container")
		}
		mustGetWorker(tCtx, m, pod.UID, testContainerName, readiness)
		replacement := testID("replacement")
		m.SyncPod(tCtx, updated, runtimePodStatus(testContainerName, replacement, kubecontainer.ContainerStateRunning))
		assertResult(tCtx, m, replacement, startup, results.Unknown)
		mustGetWorker(tCtx, m, pod.UID, testContainerName, startup)
		if _, ok := getContainerBoundWorker(m, pod.UID, testContainerName, readiness); ok {
			tCtx.Fatal("replacement readiness bypassed startup")
		}
	})
}

func snapshotMutable(w *containerBoundWorker) struct {
	spec        *v1.Probe
	hold        bool
	completed   time.Time
	consecutive int
} {
	s := w.mutable
	s.mu.Lock()
	defer s.mu.Unlock()
	return struct {
		spec        *v1.Probe
		hold        bool
		completed   time.Time
		consecutive int
	}{s.spec, s.hold, s.completed, s.consecutive}
}
