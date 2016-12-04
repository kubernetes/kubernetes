/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
)

func init() {
	runtime.ReallyCrash = true
}

func TestDoProbe(t *testing.T) {
	m := newTestManager()

	// Test statuses.
	runningStatus := getTestRunningStatus()
	pendingStatus := getTestRunningStatus()
	pendingStatus.ContainerStatuses[0].State.Running = nil
	terminatedStatus := getTestRunningStatus()
	terminatedStatus.ContainerStatuses[0].State.Running = nil
	terminatedStatus.ContainerStatuses[0].State.Terminated = &v1.ContainerStateTerminated{
		StartedAt: metav1.Now(),
	}
	otherStatus := getTestRunningStatus()
	otherStatus.ContainerStatuses[0].Name = "otherContainer"
	failedStatus := getTestRunningStatus()
	failedStatus.Phase = v1.PodFailed

	tests := []struct {
		probe          v1.Probe
		podStatus      *v1.PodStatus
		expectContinue bool
		expectSet      bool
		expectedResult results.Result
	}{
		{ // No status.
			expectContinue: true,
		},
		{ // Pod failed
			podStatus: &failedStatus,
		},
		{ // No container status
			podStatus:      &otherStatus,
			expectContinue: true,
		},
		{ // Container waiting
			podStatus:      &pendingStatus,
			expectContinue: true,
			expectSet:      true,
		},
		{ // Container terminated
			podStatus: &terminatedStatus,
			expectSet: true,
		},
		{ // Probe successful.
			podStatus:      &runningStatus,
			expectContinue: true,
			expectSet:      true,
			expectedResult: results.Success,
		},
		{ // Initial delay passed
			podStatus: &runningStatus,
			probe: v1.Probe{
				InitialDelaySeconds: -100,
			},
			expectContinue: true,
			expectSet:      true,
			expectedResult: results.Success,
		},
	}

	for _, probeType := range [...]probeType{liveness, readiness} {
		for i, test := range tests {
			w := newTestWorker(m, probeType, test.probe)
			if test.podStatus != nil {
				m.statusManager.SetPodStatus(w.pod, *test.podStatus)
			}
			if c := w.doProbe(); c != test.expectContinue {
				t.Errorf("[%s-%d] Expected continue to be %v but got %v", probeType, i, test.expectContinue, c)
			}
			result, ok := resultsManager(m, probeType).Get(testContainerID)
			if ok != test.expectSet {
				t.Errorf("[%s-%d] Expected to have result: %v but got %v", probeType, i, test.expectSet, ok)
			}
			if result != test.expectedResult {
				t.Errorf("[%s-%d] Expected result: %v but got %v", probeType, i, test.expectedResult, result)
			}

			// Clean up.
			m.statusManager = status.NewManager(&fake.Clientset{}, kubepod.NewBasicPodManager(nil))
			resultsManager(m, probeType).Remove(testContainerID)
		}
	}
}

func TestInitialDelay(t *testing.T) {
	m := newTestManager()

	for _, probeType := range [...]probeType{liveness, readiness} {
		w := newTestWorker(m, probeType, v1.Probe{
			InitialDelaySeconds: 10,
		})
		m.statusManager.SetPodStatus(w.pod, getTestRunningStatus())

		expectContinue(t, w, w.doProbe(), "during initial delay")
		expectResult(t, w, results.Result(probeType == liveness), "during initial delay")

		// 100 seconds later...
		laterStatus := getTestRunningStatus()
		laterStatus.ContainerStatuses[0].State.Running.StartedAt.Time =
			time.Now().Add(-100 * time.Second)
		m.statusManager.SetPodStatus(w.pod, laterStatus)

		// Second call should succeed (already waited).
		expectContinue(t, w, w.doProbe(), "after initial delay")
		expectResult(t, w, results.Success, "after initial delay")
	}
}

func TestFailureThreshold(t *testing.T) {
	m := newTestManager()
	w := newTestWorker(m, readiness, v1.Probe{SuccessThreshold: 1, FailureThreshold: 3})
	m.statusManager.SetPodStatus(w.pod, getTestRunningStatus())

	for i := 0; i < 2; i++ {
		// First probe should succeed.
		m.prober.exec = fakeExecProber{probe.Success, nil}

		for j := 0; j < 3; j++ {
			msg := fmt.Sprintf("%d success (%d)", j+1, i)
			expectContinue(t, w, w.doProbe(), msg)
			expectResult(t, w, results.Success, msg)
		}

		// Prober starts failing :(
		m.prober.exec = fakeExecProber{probe.Failure, nil}

		// Next 2 probes should still be "success".
		for j := 0; j < 2; j++ {
			msg := fmt.Sprintf("%d failing (%d)", j+1, i)
			expectContinue(t, w, w.doProbe(), msg)
			expectResult(t, w, results.Success, msg)
		}

		// Third & following fail.
		for j := 0; j < 3; j++ {
			msg := fmt.Sprintf("%d failure (%d)", j+3, i)
			expectContinue(t, w, w.doProbe(), msg)
			expectResult(t, w, results.Failure, msg)
		}
	}
}

func TestSuccessThreshold(t *testing.T) {
	m := newTestManager()
	w := newTestWorker(m, readiness, v1.Probe{SuccessThreshold: 3, FailureThreshold: 1})
	m.statusManager.SetPodStatus(w.pod, getTestRunningStatus())

	// Start out failure.
	w.resultsManager.Set(testContainerID, results.Failure, &v1.Pod{})

	for i := 0; i < 2; i++ {
		// Probe defaults to Failure.
		for j := 0; j < 2; j++ {
			msg := fmt.Sprintf("%d success (%d)", j+1, i)
			expectContinue(t, w, w.doProbe(), msg)
			expectResult(t, w, results.Failure, msg)
		}

		// Continuing success!
		for j := 0; j < 3; j++ {
			msg := fmt.Sprintf("%d success (%d)", j+3, i)
			expectContinue(t, w, w.doProbe(), msg)
			expectResult(t, w, results.Success, msg)
		}

		// Prober flakes :(
		m.prober.exec = fakeExecProber{probe.Failure, nil}
		msg := fmt.Sprintf("1 failure (%d)", i)
		expectContinue(t, w, w.doProbe(), msg)
		expectResult(t, w, results.Failure, msg)

		// Back to success.
		m.prober.exec = fakeExecProber{probe.Success, nil}
	}
}

func TestCleanUp(t *testing.T) {
	m := newTestManager()

	for _, probeType := range [...]probeType{liveness, readiness} {
		key := probeKey{testPodUID, testContainerName, probeType}
		w := newTestWorker(m, probeType, v1.Probe{})
		m.statusManager.SetPodStatus(w.pod, getTestRunningStatus())
		go w.run()
		m.workers[key] = w

		// Wait for worker to run.
		condition := func() (bool, error) {
			ready, _ := resultsManager(m, probeType).Get(testContainerID)
			return ready == results.Success, nil
		}
		if ready, _ := condition(); !ready {
			if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, condition); err != nil {
				t.Fatalf("[%s] Error waiting for worker ready: %v", probeType, err)
			}
		}

		for i := 0; i < 10; i++ {
			w.stop() // Stop should be callable multiple times without consequence.
		}
		if err := waitForWorkerExit(m, []probeKey{key}); err != nil {
			t.Fatalf("[%s] error waiting for worker exit: %v", probeType, err)
		}

		if _, ok := resultsManager(m, probeType).Get(testContainerID); ok {
			t.Errorf("[%s] Expected result to be cleared.", probeType)
		}
		if _, ok := m.workers[key]; ok {
			t.Errorf("[%s] Expected worker to be cleared.", probeType)
		}
	}
}

func TestHandleCrash(t *testing.T) {
	runtime.ReallyCrash = false // Test that we *don't* really crash.

	m := newTestManager()
	w := newTestWorker(m, readiness, v1.Probe{})
	m.statusManager.SetPodStatus(w.pod, getTestRunningStatus())

	expectContinue(t, w, w.doProbe(), "Initial successful probe.")
	expectResult(t, w, results.Success, "Initial successful probe.")

	// Prober starts crashing.
	m.prober = &prober{
		refManager: kubecontainer.NewRefManager(),
		recorder:   &record.FakeRecorder{},
		exec:       crashingExecProber{},
	}

	// doProbe should recover from the crash, and keep going.
	expectContinue(t, w, w.doProbe(), "Crashing probe.")
	expectResult(t, w, results.Success, "Crashing probe unchanged.")
}

func expectResult(t *testing.T, w *worker, expectedResult results.Result, msg string) {
	result, ok := resultsManager(w.probeManager, w.probeType).Get(w.containerID)
	if !ok {
		t.Errorf("[%s - %s] Expected result to be set, but was not set", w.probeType, msg)
	} else if result != expectedResult {
		t.Errorf("[%s - %s] Expected result to be %v, but was %v",
			w.probeType, msg, expectedResult, result)
	}
}

func expectContinue(t *testing.T, w *worker, c bool, msg string) {
	if !c {
		t.Errorf("[%s - %s] Expected to continue, but did not", w.probeType, msg)
	}
}

func resultsManager(m *manager, probeType probeType) results.Manager {
	switch probeType {
	case readiness:
		return m.readinessManager
	case liveness:
		return m.livenessManager
	}
	panic(fmt.Errorf("Unhandled case: %v", probeType))
}

type crashingExecProber struct{}

func (p crashingExecProber) Probe(_ exec.Cmd) (probe.Result, string, error) {
	panic("Intentional Probe crash.")
}

func TestOnHoldOnLivenessCheckFailure(t *testing.T) {
	m := newTestManager()
	w := newTestWorker(m, liveness, v1.Probe{SuccessThreshold: 1, FailureThreshold: 1})
	status := getTestRunningStatus()
	m.statusManager.SetPodStatus(w.pod, getTestRunningStatus())

	// First probe should fail.
	m.prober.exec = fakeExecProber{probe.Failure, nil}
	msg := "first probe"
	expectContinue(t, w, w.doProbe(), msg)
	expectResult(t, w, results.Failure, msg)
	if !w.onHold {
		t.Errorf("Prober should be on hold due to liveness check failure")
	}
	// Set fakeExecProber to return success. However, the result will remain
	// failure because the worker is on hold and won't probe.
	m.prober.exec = fakeExecProber{probe.Success, nil}
	msg = "while on hold"
	expectContinue(t, w, w.doProbe(), msg)
	expectResult(t, w, results.Failure, msg)
	if !w.onHold {
		t.Errorf("Prober should be on hold due to liveness check failure")
	}

	// Set a new container ID to lift the hold. The next probe will succeed.
	status.ContainerStatuses[0].ContainerID = "test://newCont_ID"
	m.statusManager.SetPodStatus(w.pod, status)
	msg = "hold lifted"
	expectContinue(t, w, w.doProbe(), msg)
	expectResult(t, w, results.Success, msg)
	if w.onHold {
		t.Errorf("Prober should not be on hold anymore")
	}
}
