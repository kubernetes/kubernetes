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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/probe"
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
		probe     api.Probe
		podStatus *api.PodStatus

		expectContinue    bool
		expectReadySet    bool
		expectedReadiness results.Result
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
			expectReadySet: true,
		},
		{ // Container terminated
			podStatus:      &terminatedStatus,
			expectReadySet: true,
		},
		{ // Probe successful.
			podStatus:         &runningStatus,
			expectContinue:    true,
			expectReadySet:    true,
			expectedReadiness: results.Success,
		},
		{ // Initial delay passed
			podStatus: &runningStatus,
			probe: api.Probe{
				InitialDelaySeconds: -100,
			},
			expectContinue:    true,
			expectReadySet:    true,
			expectedReadiness: results.Success,
		},
	}

	for i, test := range tests {
		w := newTestWorker(test.probe)
		if test.podStatus != nil {
			m.statusManager.SetPodStatus(w.pod, *test.podStatus)
		}
		if c := doProbe(m, w); c != test.expectContinue {
			t.Errorf("[%d] Expected continue to be %v but got %v", i, test.expectContinue, c)
		}
		ready, ok := m.readinessCache.Get(containerID)
		if ok != test.expectReadySet {
			t.Errorf("[%d] Expected to have readiness: %v but got %v", i, test.expectReadySet, ok)
		}
		if ready != test.expectedReadiness {
			t.Errorf("[%d] Expected readiness: %v but got %v", i, test.expectedReadiness, ready)
		}

		// Clean up.
		m.statusManager.DeletePodStatus(podUID)
		m.readinessCache.Remove(containerID)
	}
}

func TestInitialDelay(t *testing.T) {
	m := newTestManager()
	w := newTestWorker(api.Probe{
		InitialDelaySeconds: 10,
	})
	m.statusManager.SetPodStatus(w.pod, getRunningStatus())

	if !doProbe(m, w) {
		t.Errorf("Expected to continue, but did not")
	}

	ready, ok := m.readinessCache.Get(containerID)
	if !ok {
		t.Errorf("Expected readiness to be false, but was not set")
	} else if ready {
		t.Errorf("Expected readiness to be false, but was true")
	}

	// 100 seconds later...
	laterStatus := getRunningStatus()
	laterStatus.ContainerStatuses[0].State.Running.StartedAt.Time =
		time.Now().Add(-100 * time.Second)
	m.statusManager.SetPodStatus(w.pod, laterStatus)

	// Second call should succeed (already waited).
	if !doProbe(m, w) {
		t.Errorf("Expected to continue, but did not")
	}

	ready, ok = m.readinessCache.Get(containerID)
	if !ok {
		t.Errorf("Expected readiness to be true, but was not set")
	} else if !ready {
		t.Errorf("Expected readiness to be true, but was false")
	}
}

func TestCleanUp(t *testing.T) {
	m := newTestManager()
	pod := getTestPod(api.Probe{})
	m.statusManager.SetPodStatus(&pod, getRunningStatus())
	m.readinessCache.Set(containerID, results.Success)
	w := m.newWorker(&pod, pod.Spec.Containers[0])
	m.readinessProbes[containerPath{podUID, containerName}] = w

	if ready, _ := m.readinessCache.Get(containerID); !ready {
		t.Fatal("Expected readiness to be true.")
	}

	close(w.stop)
	if err := waitForWorkerExit(m, []containerPath{{podUID, containerName}}); err != nil {
		t.Fatal(err)
	}

	if _, ok := m.readinessCache.Get(containerID); ok {
		t.Error("Expected readiness to be cleared.")
	}
	if _, ok := m.readinessProbes[containerPath{podUID, containerName}]; ok {
		t.Error("Expected worker to be cleared.")
	}
}

func TestHandleCrash(t *testing.T) {
	m := newTestManager()
	m.prober = CrashingProber{}
	w := newTestWorker(api.Probe{})
	m.statusManager.SetPodStatus(w.pod, getRunningStatus())

	// doProbe should recover from the crash, and keep going.
	if !doProbe(m, w) {
		t.Error("Expected to keep going, but terminated.")
	}
	if _, ok := m.readinessCache.Get(containerID); ok {
		t.Error("Expected readiness to be unchanged from crash.")
	}
}

func newTestWorker(probeSpec api.Probe) *worker {
	pod := getTestPod(probeSpec)
	return &worker{
		stop:      make(chan struct{}),
		pod:       &pod,
		container: pod.Spec.Containers[0],
		spec:      &probeSpec,
	}
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

func getTestPod(probeSpec api.Probe) api.Pod {
	container := api.Container{
		Name:           containerName,
		ReadinessProbe: &probeSpec,
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

type CrashingProber struct{}

func (f CrashingProber) ProbeLiveness(_ *api.Pod, _ api.PodStatus, c api.Container, _ kubecontainer.ContainerID, _ int64) (probe.Result, error) {
	panic("Intentional ProbeLiveness crash.")
}

func (f CrashingProber) ProbeReadiness(_ *api.Pod, _ api.PodStatus, c api.Container, _ kubecontainer.ContainerID) (probe.Result, error) {
	panic("Intentional ProbeReadiness crash.")
}
