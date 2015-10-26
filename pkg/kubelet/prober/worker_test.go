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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
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
			m.statusManager.DeletePodStatus(podUID)
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

		if !w.doProbe() {
			t.Errorf("[%s] Expected to continue, but did not", probeType)
		}

		expectedResult := results.Result(probeType == liveness)
		result, ok := resultsManager(m, probeType).Get(containerID)
		if !ok {
			t.Errorf("[%s] Expected result to be set during initial delay, but was not set", probeType)
		} else if result != expectedResult {
			t.Errorf("[%s] Expected result to be %v during initial delay, but was %v",
				probeType, expectedResult, result)
		}

		// 100 seconds later...
		laterStatus := getRunningStatus()
		laterStatus.ContainerStatuses[0].State.Running.StartedAt.Time =
			time.Now().Add(-100 * time.Second)
		m.statusManager.SetPodStatus(w.pod, laterStatus)

		// Second call should succeed (already waited).
		if !w.doProbe() {
			t.Errorf("[%s] Expected to continue, but did not", probeType)
		}

		result, ok = resultsManager(m, probeType).Get(containerID)
		if !ok {
			t.Errorf("[%s] Expected result to be true, but was not set", probeType)
		} else if !result {
			t.Errorf("[%s] Expected result to be true, but was false", probeType)
		}
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
	m.prober = &prober{
		refManager: kubecontainer.NewRefManager(),
		recorder:   &record.FakeRecorder{},
		exec:       crashingExecProber{},
	}

	w := newTestWorker(m, readiness, api.Probe{})
	m.statusManager.SetPodStatus(w.pod, getRunningStatus())

	// doProbe should recover from the crash, and keep going.
	if !w.doProbe() {
		t.Error("Expected to keep going, but terminated.")
	}
	if _, ok := m.readinessManager.Get(containerID); ok {
		t.Error("Expected readiness to be unchanged from crash.")
	}
}

func newTestWorker(m *manager, probeType probeType, probeSpec api.Probe) *worker {
	// All tests rely on the fake exec prober.
	probeSpec.Handler = api.Handler{
		Exec: &api.ExecAction{},
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
