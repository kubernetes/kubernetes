/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	containerName = "cOnTaInEr_NaMe"
	podUID        = "pOd_UiD"
)

var containerID = kubecontainer.ContainerID{"test", "cOnTaInEr_Id"}

func TestDoProbe(t *testing.T) {
	m := newTestManager()

	// Test statuses.
	runningStatus := getRunningStatus()
	pendingStatus := getRunningStatus()
	pendingStatus.ContainerStatuses[0].State.Running = nil
	terminatedStatus := getRunningStatus()
	terminatedStatus.ContainerStatuses[0].State.Running = nil
	terminatedStatus.ContainerStatuses[0].State.Terminated = &api.ContainerStateTerminated{
		StartedAt: unversioned.Now(),
	}
	otherStatus := getRunningStatus()
	otherStatus.ContainerStatuses[0].Name = "otherContainer"
	failedStatus := getRunningStatus()
	failedStatus.Phase = api.PodFailed

	tests := []struct {
		probe          api.Probe
		podStatus      *api.PodStatus
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
			probe: api.Probe{
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
			result, ok := resultsManager(m, probeType).Get(containerID)
			if ok != test.expectSet {
				t.Errorf("[%s-%d] Expected to have result: %v but got %v", probeType, i, test.expectSet, ok)
			}
			if result != test.expectedResult {
				t.Errorf("[%s-%d] Expected result: %v but got %v", probeType, i, test.expectedResult, result)
			}

			// Clean up.
			m.statusManager = status.NewManager(&testclient.Fake{}, kubepod.NewBasicPodManager(nil))
			resultsManager(m, probeType).Remove(containerID)
		}
	}
}

func TestInitialDelay(t *testing.T) {
	m := newTestManager()

	for _, probeType := range [...]probeType{liveness, readiness} {
		w := newTestWorker(m, probeType, api.Probe{
			InitialDelaySeconds: 10,
		})
		m.statusManager.SetPodStatus(w.pod, getRunningStatus())

		expectContinue(t, w, w.doProbe(), "during initial delay")
		expectResult(t, w, results.Result(probeType == liveness), "during initial delay")

		// 100 seconds later...
		laterStatus := getRunningStatus()
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
	w := newTestWorker(m, readiness, api.Probe{})
	m.statusManager.SetPodStatus(w.pod, getRunningStatus())

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
			msg := fmt.Sprintf("%d failure (%d)", j+1, i)
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
	w := newTestWorker(m, readiness, api.Probe{SuccessThreshold: 3, FailureThreshold: 1})
	m.statusManager.SetPodStatus(w.pod, getRunningStatus())

	// Start out failure.
	w.resultsManager.Set(containerID, results.Failure, nil)

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
		key := probeKey{podUID, containerName, probeType}
		w := newTestWorker(m, probeType, api.Probe{})
		m.statusManager.SetPodStatus(w.pod, getRunningStatus())
		go w.run()
		m.workers[key] = w

		// Wait for worker to run.
		condition := func() (bool, error) {
			ready, _ := resultsManager(m, probeType).Get(containerID)
			return ready == results.Success, nil
		}
		if ready, _ := condition(); !ready {
			if err := wait.Poll(100*time.Millisecond, util.ForeverTestTimeout, condition); err != nil {
				t.Fatalf("[%s] Error waiting for worker ready: %v", probeType, err)
			}
		}

		close(w.stop)
		if err := waitForWorkerExit(m, []probeKey{key}); err != nil {
			t.Fatalf("[%s] error waiting for worker exit: %v", probeType, err)
		}

		if _, ok := resultsManager(m, probeType).Get(containerID); ok {
			t.Errorf("[%s] Expected result to be cleared.", probeType)
		}
		if _, ok := m.workers[key]; ok {
			t.Errorf("[%s] Expected worker to be cleared.", probeType)
		}
	}
}

func TestHandleCrash(t *testing.T) {
	m := newTestManager()
	w := newTestWorker(m, readiness, api.Probe{})
	m.statusManager.SetPodStatus(w.pod, getRunningStatus())

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

func newTestWorker(m *manager, probeType probeType, probeSpec api.Probe) *worker {
	// All tests rely on the fake exec prober.
	probeSpec.Handler = api.Handler{
		Exec: &api.ExecAction{},
	}
	// Apply default values.
	defaults := map[string]int64{
		"TimeoutSeconds":   1,
		"PeriodSeconds":    10,
		"SuccessThreshold": 1,
		"FailureThreshold": 3,
	}
	for field, value := range defaults {
		f := reflect.ValueOf(&probeSpec).Elem().FieldByName(field)
		if f.Int() == 0 {
			f.SetInt(value)
		}
	}

	pod := getTestPod(probeType, probeSpec)
	return newWorker(m, probeType, &pod, pod.Spec.Containers[0])
}

func getRunningStatus() api.PodStatus {
	containerStatus := api.ContainerStatus{
		Name:        containerName,
		ContainerID: containerID.String(),
	}
	containerStatus.State.Running = &api.ContainerStateRunning{unversioned.Now()}
	podStatus := api.PodStatus{
		Phase:             api.PodRunning,
		ContainerStatuses: []api.ContainerStatus{containerStatus},
	}
	return podStatus
}

func getTestPod(probeType probeType, probeSpec api.Probe) api.Pod {
	container := api.Container{
		Name: containerName,
	}
	switch probeType {
	case readiness:
		container.ReadinessProbe = &probeSpec
	case liveness:
		container.LivenessProbe = &probeSpec
	}
	pod := api.Pod{
		Spec: api.PodSpec{
			Containers:    []api.Container{container},
			RestartPolicy: api.RestartPolicyNever,
		},
	}
	pod.UID = podUID
	return pod
}

func expectResult(t *testing.T, w *worker, expectedResult results.Result, msg string) {
	result, ok := resultsManager(w.probeManager, w.probeType).Get(containerID)
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
