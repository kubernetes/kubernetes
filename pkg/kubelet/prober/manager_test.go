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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"
)

func TestAddRemovePods(t *testing.T) {
	noProbePod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID: "no_probe_pod",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{
				Name: "no_probe1",
			}, {
				Name: "no_probe2",
			}},
		},
	}

	probePod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID: "probe_pod",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{
				Name: "no_probe1",
			}, {
				Name:           "readiness",
				ReadinessProbe: &api.Probe{},
			}, {
				Name: "no_probe2",
			}, {
				Name:          "liveness",
				LivenessProbe: &api.Probe{},
			}},
		},
	}

	m := newTestManager()
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}

	// Adding a pod with no probes should be a no-op.
	m.AddPod(&noProbePod)
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}

	// Adding a pod with probes.
	m.AddPod(&probePod)
	probePaths := []probeKey{
		{"probe_pod", "readiness", readiness},
		{"probe_pod", "liveness", liveness},
	}
	if err := expectProbes(m, probePaths); err != nil {
		t.Error(err)
	}

	// Removing un-probed pod.
	m.RemovePod(&noProbePod)
	if err := expectProbes(m, probePaths); err != nil {
		t.Error(err)
	}

	// Removing probed pod.
	m.RemovePod(&probePod)
	if err := waitForWorkerExit(m, probePaths); err != nil {
		t.Fatal(err)
	}
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}

	// Removing already removed pods should be a no-op.
	m.RemovePod(&probePod)
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}
}

func TestCleanupPods(t *testing.T) {
	m := newTestManager()
	podToCleanup := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID: "pod_cleanup",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{
				Name:           "prober1",
				ReadinessProbe: &api.Probe{},
			}, {
				Name:          "prober2",
				LivenessProbe: &api.Probe{},
			}},
		},
	}
	podToKeep := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID: "pod_keep",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{
				Name:           "prober1",
				ReadinessProbe: &api.Probe{},
			}, {
				Name:          "prober2",
				LivenessProbe: &api.Probe{},
			}},
		},
	}
	m.AddPod(&podToCleanup)
	m.AddPod(&podToKeep)

	m.CleanupPods([]*api.Pod{&podToKeep})

	removedProbes := []probeKey{
		{"pod_cleanup", "prober1", readiness},
		{"pod_cleanup", "prober2", liveness},
	}
	expectedProbes := []probeKey{
		{"pod_keep", "prober1", readiness},
		{"pod_keep", "prober2", liveness},
	}
	if err := waitForWorkerExit(m, removedProbes); err != nil {
		t.Fatal(err)
	}
	if err := expectProbes(m, expectedProbes); err != nil {
		t.Error(err)
	}
}

func TestUpdatePodStatus(t *testing.T) {
	const podUID = "pod_uid"
	unprobed := api.ContainerStatus{
		Name:        "unprobed_container",
		ContainerID: "test://unprobed_container_id",
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	probedReady := api.ContainerStatus{
		Name:        "probed_container_ready",
		ContainerID: "test://probed_container_ready_id",
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	probedPending := api.ContainerStatus{
		Name:        "probed_container_pending",
		ContainerID: "test://probed_container_pending_id",
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	probedUnready := api.ContainerStatus{
		Name:        "probed_container_unready",
		ContainerID: "test://probed_container_unready_id",
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	terminated := api.ContainerStatus{
		Name:        "terminated_container",
		ContainerID: "test://terminated_container_id",
		State: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{},
		},
	}
	podStatus := api.PodStatus{
		Phase: api.PodRunning,
		ContainerStatuses: []api.ContainerStatus{
			unprobed, probedReady, probedPending, probedUnready, terminated,
		},
	}

	m := newTestManager()
	// Setup probe "workers" and cached results.
	m.workers = map[probeKey]*worker{
		probeKey{podUID, unprobed.Name, liveness}:       {},
		probeKey{podUID, probedReady.Name, readiness}:   {},
		probeKey{podUID, probedPending.Name, readiness}: {},
		probeKey{podUID, probedUnready.Name, readiness}: {},
		probeKey{podUID, terminated.Name, readiness}:    {},
	}
	m.readinessManager.Set(kubecontainer.ParseContainerID(probedReady.ContainerID), results.Success, nil)
	m.readinessManager.Set(kubecontainer.ParseContainerID(probedUnready.ContainerID), results.Failure, nil)
	m.readinessManager.Set(kubecontainer.ParseContainerID(terminated.ContainerID), results.Success, nil)

	m.UpdatePodStatus(podUID, &podStatus)

	expectedReadiness := map[probeKey]bool{
		probeKey{podUID, unprobed.Name, readiness}:      true,
		probeKey{podUID, probedReady.Name, readiness}:   true,
		probeKey{podUID, probedPending.Name, readiness}: false,
		probeKey{podUID, probedUnready.Name, readiness}: false,
		probeKey{podUID, terminated.Name, readiness}:    false,
	}
	for _, c := range podStatus.ContainerStatuses {
		expected, ok := expectedReadiness[probeKey{podUID, c.Name, readiness}]
		if !ok {
			t.Fatalf("Missing expectation for test case: %v", c.Name)
		}
		if expected != c.Ready {
			t.Errorf("Unexpected readiness for container %v: Expected %v but got %v",
				c.Name, expected, c.Ready)
		}
	}
}

func expectProbes(m *manager, expectedProbes []probeKey) error {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	var unexpected []probeKey
	missing := make([]probeKey, len(expectedProbes))
	copy(missing, expectedProbes)

outer:
	for probePath := range m.workers {
		for i, expectedPath := range missing {
			if probePath == expectedPath {
				missing = append(missing[:i], missing[i+1:]...)
				continue outer
			}
		}
		unexpected = append(unexpected, probePath)
	}

	if len(missing) == 0 && len(unexpected) == 0 {
		return nil // Yay!
	}

	return fmt.Errorf("Unexpected probes: %v; Missing probes: %v;", unexpected, missing)
}

func newTestManager() *manager {
	const probePeriod = 1
	m := NewManager(
		probePeriod,
		status.NewManager(&testclient.Fake{}),
		results.NewManager(),
		results.NewManager(),
		nil, // runner
		kubecontainer.NewRefManager(),
		&record.FakeRecorder{},
	).(*manager)
	// Don't actually execute probes.
	m.prober.exec = fakeExecProber{probe.Success, nil}
	return m
}

// Wait for the given workers to exit & clean up.
func waitForWorkerExit(m *manager, workerPaths []probeKey) error {
	const interval = 100 * time.Millisecond

	for _, w := range workerPaths {
		condition := func() (bool, error) {
			_, exists := m.getWorker(w.podUID, w.containerName, w.probeType)
			return !exists, nil
		}
		if exited, _ := condition(); exited {
			continue // Already exited, no need to poll.
		}
		glog.Infof("Polling %v", w)
		if err := wait.Poll(interval, util.ForeverTestTimeout, condition); err != nil {
			return err
		}
	}

	return nil
}
