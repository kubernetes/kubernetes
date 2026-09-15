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
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/test/utils/ktesting"
	utilexec "k8s.io/utils/exec"
)

type heldProbeCall struct {
	command  string
	id       kubecontainer.ContainerID
	canceled <-chan struct{}
}

type heldProbeRunner struct {
	calls   chan heldProbeCall
	release chan struct{}
}

func (r *heldProbeRunner) RunInContainer(ctx context.Context, id kubecontainer.ContainerID, command []string, _ time.Duration) ([]byte, error) {
	r.calls <- heldProbeCall{command: command[0], id: id, canceled: ctx.Done()}

	// Some runtimes complete a call after cancellation without returning a context error.
	<-r.release
	return []byte("ok"), nil
}

func TestMutableProbeReusesWorker(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	synctest.Test(t, func(t *testing.T) {
		logger, ctx := ktesting.NewTestContext(t)
		m := newTestManager()
		m.start = time.Now().Add(-time.Hour)
		runner := &heldProbeRunner{calls: make(chan heldProbeCall, 4), release: make(chan struct{})}
		m.prober = newProber(runner, &record.FakeRecorder{})
		pod := getTestPod()
		setTestProbe(pod, readiness, v1.Probe{PeriodSeconds: 10, SuccessThreshold: 1, FailureThreshold: 1})
		pod.Spec.Containers[0].ReadinessProbe.Exec.Command = []string{"old"}
		m.statusManager.SetPodStatus(logger, pod, getTestRunningStatus())
		defer func() {
			m.RemovePod(pod)
			close(runner.release)
			synctest.Wait()

			if m.workerCount() != 0 {
				t.Fatalf("workers remained after pod cleanup: %d", m.workerCount())
			}
		}()

		if _, err := m.ReconcilePod(ctx, pod); err != nil {
			t.Fatal(err)
		}

		oldCall := <-runner.calls
		w, ok := m.getWorker(pod.UID, testContainerName, readiness)
		if !ok {
			t.Fatal("readiness worker was not registered")
		}

		oldUpdate := <-m.readinessManager.Updates()
		removed := pod.DeepCopy()
		removed.Spec.Containers[0].ReadinessProbe = nil

		if changed, err := m.ReconcilePod(ctx, removed); err != nil || !changed {
			t.Fatalf("removing readiness did not change derived state: changed=%v err=%v", changed, err)
		}

		<-oldCall.canceled

		if _, enabled := m.getWorker(pod.UID, testContainerName, readiness); enabled {
			t.Fatal("disabled probe was reported as enabled")
		}
		if m.workerCount() != 1 {
			t.Fatal("removing a probe discarded its pending execution state")
		}

		for _, command := range []string{"intermediate", "latest"} {
			updated := pod.DeepCopy()
			updated.Spec.Containers[0].ReadinessProbe.Exec.Command = []string{command}

			if _, err := m.ReconcilePod(ctx, updated); err != nil {
				t.Fatal(err)
			}
		}

		synctest.Wait()
		current, ok := m.getWorker(pod.UID, testContainerName, readiness)
		if !ok || current != w {
			t.Fatal("re-enabling the probe replaced its worker")
		}
		if m.IsResultCurrent(oldUpdate, readiness) {
			t.Fatal("a previous enablement's publication became current again")
		}

		select {
		case call := <-runner.calls:
			t.Fatalf("started %q while the canceled call had not returned", call.command)
		default:
		}

		runner.release <- struct{}{}
		call := <-runner.calls
		if call.command != "latest" {
			t.Fatalf("next probe used command %q, want latest", call.command)
		}
		if result, _ := m.readinessManager.Get(testContainerID); result != results.Failure {
			t.Fatalf("the old call changed the re-enabled readiness result to %v", result)
		}
	})
}

func TestThresholdUpdatesDoNotLoseIntermediateChanges(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	w := newTestWorker(m, readiness, v1.Probe{SuccessThreshold: 1, FailureThreshold: 3})
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())
	w.doProbe(ctx)

	m.prober.exec = fakeExecProber{result: probe.Failure}
	w.doProbe(ctx)
	w.doProbe(ctx)

	blocking := &blockingExecProber{started: make(chan struct{}), release: make(chan struct{}), result: probe.Failure}
	m.prober.exec = blocking
	done := make(chan struct{})
	go func() { defer close(done); w.doProbe(ctx) }()
	<-blocking.started

	for _, threshold := range []int32{5, 3} {
		updated := w.configSnapshot().pod.DeepCopy()
		updated.Spec.Containers[0].ReadinessProbe.FailureThreshold = threshold
		w.updateConfig(updated, updated.Spec.Containers[0])
	}

	close(blocking.release)
	<-done

	if w.resultRun != 1 {
		t.Fatalf("in-flight result reused the pre-update streak: got %d, want 1", w.resultRun)
	}
	if result, _ := w.resultsManager.Get(testContainerID); result != results.Success {
		t.Fatalf("in-flight failure crossed the threshold using the old streak: %v", result)
	}
}

type countedProbe struct{ calls atomic.Int32 }

func (p *countedProbe) Probe(utilexec.Cmd) (probe.Result, string, error) {
	p.calls.Add(1)
	return probe.Success, "", nil
}

func TestMutableInitialDelay(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	synctest.Test(t, func(t *testing.T) {
		logger, ctx := ktesting.NewTestContext(t)
		m := newTestManager()
		counter := &countedProbe{}
		m.prober.exec = counter
		w := newTestWorker(m, readiness, v1.Probe{InitialDelaySeconds: 30, PeriodSeconds: 10, SuccessThreshold: 1})
		status := getTestRunningStatus()
		startedAt := time.Now()
		status.ContainerStatuses[0].State.Running.StartedAt = metav1.NewTime(startedAt)
		m.statusManager.SetPodStatus(logger, w.pod, status)
		w.doProbe(ctx)

		updated := w.pod.DeepCopy()
		updated.Spec.Containers[0].ReadinessProbe.InitialDelaySeconds = 10
		w.updateConfig(updated, updated.Spec.Containers[0])

		// Simulate an old timer firing after a new deadline has been installed.
		w.probe(ctx, true)

		if counter.calls.Load() != 0 || !w.nextProbeTime.Equal(startedAt.Add(10*time.Second)) {
			t.Fatal("probe ignored the revised initial delay")
		}

		time.Sleep(10 * time.Second)
		w.probe(ctx, true)

		if counter.calls.Load() != 1 {
			t.Fatal("probe did not run at the revised deadline")
		}

		updated = updated.DeepCopy()
		updated.Spec.Containers[0].ReadinessProbe.InitialDelaySeconds = 300
		w.updateConfig(updated, updated.Spec.Containers[0])
		time.Sleep(10 * time.Second)
		w.probe(ctx, true)

		if counter.calls.Load() != 2 {
			t.Fatal("initial delay was reintroduced after probing had begun")
		}

		status.ContainerStatuses[0].ContainerID = "test://replacement"
		status.ContainerStatuses[0].State.Running.StartedAt = metav1.Now()
		m.statusManager.SetPodStatus(logger, w.pod, status)
		w.doProbe(ctx)

		if counter.calls.Load() != 2 || w.firstProbeStarted {
			t.Fatal("replacement container did not use its own initial delay")
		}
	})
}

func TestMutablePeriodReschedulesWorker(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	synctest.Test(t, func(t *testing.T) {
		logger, ctx := ktesting.NewTestContext(t)
		m := newTestManager()
		m.start = time.Now().Add(-time.Hour)
		counter := &countedProbe{}
		m.prober.exec = counter
		pod := getTestPod()
		setTestProbe(pod, readiness, v1.Probe{PeriodSeconds: 10, SuccessThreshold: 1})
		m.statusManager.SetPodStatus(logger, pod, getTestRunningStatus())
		defer func() { m.RemovePod(pod); synctest.Wait() }()

		m.ReconcilePod(ctx, pod)
		synctest.Wait()

		if counter.calls.Load() != 1 {
			t.Fatal("initial probe did not execute")
		}

		time.Sleep(time.Second)
		updated := pod.DeepCopy()
		updated.Spec.Containers[0].ReadinessProbe.PeriodSeconds = 2
		m.ReconcilePod(ctx, updated)
		synctest.Wait()

		if counter.calls.Load() != 1 {
			t.Fatal("period update ran before the previous completion plus new period")
		}

		time.Sleep(time.Second)
		synctest.Wait()

		if counter.calls.Load() != 2 {
			t.Fatal("worker did not reschedule its old timer")
		}

		updated = updated.DeepCopy()
		updated.Spec.Containers[0].ReadinessProbe.PeriodSeconds = 20
		m.ReconcilePod(ctx, updated)
		time.Sleep(10 * time.Second)
		synctest.Wait()

		if counter.calls.Load() != 2 {
			t.Fatal("worker used a timer from the previous configuration")
		}
	})
}

func TestExecutionUpdateRunsBeforeUpdatedPeriod(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	for _, change := range []string{"handler", "timeout"} {
		t.Run(change, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				logger, ctx := ktesting.NewTestContext(t)
				m := newTestManager()
				counter := &countedProbe{}
				m.prober.exec = counter
				w := newTestWorker(m, readiness, v1.Probe{PeriodSeconds: 10, SuccessThreshold: 1})
				m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())
				w.doProbe(ctx)
				time.Sleep(time.Second)
				updated := w.pod.DeepCopy()
				if change == "handler" {
					updated.Spec.Containers[0].ReadinessProbe.Exec.Command = []string{"updated"}
				} else {
					updated.Spec.Containers[0].ReadinessProbe.TimeoutSeconds++
				}

				w.updateConfig(updated, updated.Spec.Containers[0])
				updated = updated.DeepCopy()
				updated.Spec.Containers[0].ReadinessProbe.PeriodSeconds = 3600
				w.updateConfig(updated, updated.Spec.Containers[0])
				w.probe(ctx, true)

				if counter.calls.Load() != 2 {
					t.Fatal("period update postponed the first attempt using the new execution configuration")
				}
			})
		})
	}
}

func TestHandlerUpdatePreservesPublishedFailure(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	m.prober.exec = fakeExecProber{result: probe.Failure}
	w := newTestWorker(m, liveness, v1.Probe{FailureThreshold: 1})
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())
	w.doProbe(ctx)
	var failure results.Update
	for failure.Result != results.Failure || !m.IsResultCurrent(failure, liveness) {
		select {
		case failure = <-m.livenessManager.Updates():
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatal("liveness failure was not published")
		}
	}

	updated := w.pod.DeepCopy()
	updated.Spec.Containers[0].LivenessProbe.Exec.Command = []string{"new"}
	w.updateConfig(updated, updated.Spec.Containers[0])

	if !m.IsResultCurrent(failure, liveness) {
		t.Fatal("replacing the execution configuration revoked an already published failure")
	}

	counter := &countedProbe{}
	m.prober.exec = counter
	w.doProbe(ctx)

	if counter.calls.Load() != 0 || !w.onHold {
		t.Fatal("configuration update resumed a probe awaiting container restart")
	}
}

func TestStartupRemovalPreservesCompletionBeforeStatusUpdate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	w := newTestWorker(m, startup, v1.Probe{SuccessThreshold: 1})
	status := getTestRunningStatusWithStarted(false)
	m.statusManager.SetPodStatus(logger, w.pod, status)
	w.doProbe(ctx)

	if !w.startupComplete {
		t.Fatal("startup success was not retained")
	}

	original := w.pod.DeepCopy()
	removed := original.DeepCopy()
	removed.Spec.Containers[0].StartupProbe = nil
	w.updateConfig(removed, removed.Spec.Containers[0])
	w.applyConfig(ctx, original, original.Spec.Containers[0])

	if result, _ := m.startupManager.Get(testContainerID); result != results.Success || !w.onHold {
		t.Fatalf("re-enabling startup lost completion while Started was still false: result=%v hold=%v", result, w.onHold)
	}
}

func TestContainerReplacementCancelsInFlightProbe(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	synctest.Test(t, func(t *testing.T) {
		logger, ctx := ktesting.NewTestContext(t)
		m := newTestManager()
		m.start = time.Now().Add(-time.Hour)
		runner := &heldProbeRunner{calls: make(chan heldProbeCall, 4), release: make(chan struct{})}
		m.prober = newProber(runner, &record.FakeRecorder{})
		pod := getTestPod()
		setTestProbe(pod, readiness, v1.Probe{PeriodSeconds: 10, SuccessThreshold: 1})
		pod.Spec.Containers[0].ReadinessProbe.Exec.Command = []string{"check"}
		status := getTestRunningStatus()
		m.statusManager.SetPodStatus(logger, pod, status)
		defer func() { m.RemovePod(pod); close(runner.release); synctest.Wait() }()

		m.ReconcilePod(ctx, pod)
		oldCall := <-runner.calls
		oldUpdate := <-m.readinessManager.Updates()

		id := kubecontainer.ContainerID{Type: "test", ID: "replacement"}
		status.ContainerStatuses[0].ContainerID = id.String()
		status.ContainerStatuses[0].State.Running.StartedAt = metav1.Now()
		m.statusManager.SetPodStatus(logger, pod, status)
		m.ReconcilePod(ctx, pod)
		<-oldCall.canceled

		if m.IsResultCurrent(oldUpdate, readiness) {
			t.Fatal("result from the previous container remained current")
		}
		if _, exists := m.readinessManager.Get(testContainerID); exists {
			t.Fatal("result for the previous container remained cached")
		}

		runner.release <- struct{}{}
		call := <-runner.calls
		if call.id != id {
			t.Fatalf("next attempt used container %v, want %v", call.id, id)
		}
		if result, ok := m.readinessManager.Get(id); !ok || result != results.Failure {
			t.Fatalf("late success replaced the new container's initial readiness: result=%v found=%v", result, ok)
		}
	})
}

func TestReconcilePreservesReadinessOnKubeletRestart(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ChangeContainerStatusOnKubeletRestart, false)

	synctest.Test(t, func(t *testing.T) {
		logger, ctx := ktesting.NewTestContext(t)
		m := newTestManager()
		pod := getTestPod()
		setTestProbe(pod, readiness, v1.Probe{InitialDelaySeconds: 7200, PeriodSeconds: 10, SuccessThreshold: 1})
		pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}
		status := getTestRunningStatus()
		status.ContainerStatuses[0].Ready = true
		status.ContainerStatuses[0].State.Running.StartedAt = metav1.NewTime(time.Now().Add(-time.Hour))
		m.statusManager.SetPodStatus(logger, pod, status)
		defer func() { m.RemovePod(pod); synctest.Wait() }()

		m.ReconcilePod(ctx, pod)
		synctest.Wait()
		derived := status.DeepCopy()
		m.UpdatePodStatus(ctx, pod, derived)

		if !derived.ContainerStatuses[0].Ready {
			t.Fatal("first reconciliation reset readiness retained across kubelet restart")
		}

		removed := pod.DeepCopy()
		removed.Spec.Containers[0].ReadinessProbe = nil
		m.ReconcilePod(ctx, removed)
		m.ReconcilePod(ctx, pod)
		derived = status.DeepCopy()
		m.UpdatePodStatus(ctx, pod, derived)

		if derived.ContainerStatuses[0].Ready {
			t.Fatal("re-added readiness probe reused the restart fallback instead of initializing")
		}
	})
}

func TestProbeInitializationPreservesStartupStagger(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	synctest.Test(t, func(t *testing.T) {
		logger, ctx := ktesting.NewTestContext(t)
		m := newTestManager()
		counter := &countedProbe{}
		m.prober.exec = counter
		w := newTestWorker(m, readiness, v1.Probe{PeriodSeconds: 10, SuccessThreshold: 1})
		m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())

		// Fix the randomized deadline so this test covers both sides of it deterministically.
		w.nextProbeTime = time.Now().Add(5 * time.Second)
		w.initializeProbe(ctx, false)
		w.probe(ctx, true)

		if counter.calls.Load() != 0 {
			t.Fatal("container discovery bypassed kubelet startup staggering")
		}

		time.Sleep(5 * time.Second)
		w.probe(ctx, true)

		if counter.calls.Load() != 1 {
			t.Fatal("probe did not execute at its staggered deadline")
		}
	})
}

func TestPodTerminationPreventsProbeReactivation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, true)

	for _, action := range []string{"remove", "stop liveness and startup", "cleanup"} {
		t.Run(action, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				logger, ctx := ktesting.NewTestContext(t)
				m := newTestManager()
				pod := getTestPod()
				for _, kind := range []probeType{startup, liveness, readiness} {
					setTestProbe(pod, kind, v1.Probe{InitialDelaySeconds: 3600})
				}

				m.statusManager.SetPodStatus(logger, pod, getTestRunningStatus())
				defer func() { m.RemovePod(pod); synctest.Wait() }()

				m.ReconcilePod(ctx, pod)
				synctest.Wait()
				switch action {
				case "remove":
					m.RemovePod(pod)
				case "stop liveness and startup":
					m.StopLivenessAndStartup(pod)
				case "cleanup":
					m.CleanupPods(nil)
				}

				synctest.Wait()
				m.ReconcilePod(ctx, pod)
				synctest.Wait()
				for _, kind := range []probeType{startup, liveness, readiness} {
					_, enabled := m.getWorker(pod.UID, testContainerName, kind)
					wantEnabled := action == "stop liveness and startup" && kind == readiness
					if enabled != wantEnabled {
						t.Errorf("%v enabled=%v, want %v after %s", kind, enabled, wantEnabled, action)
					}
				}
			})
		})
	}
}
