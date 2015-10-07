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
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/probe"
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
				Name:           "prober1",
				ReadinessProbe: &api.Probe{},
			}, {
				Name: "no_probe2",
			}, {
				Name:           "prober2",
				ReadinessProbe: &api.Probe{},
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
	probePaths := []containerPath{{"probe_pod", "prober1"}, {"probe_pod", "prober2"}}
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
				Name:           "prober2",
				ReadinessProbe: &api.Probe{},
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
				Name:           "prober2",
				ReadinessProbe: &api.Probe{},
			}},
		},
	}
	m.AddPod(&podToCleanup)
	m.AddPod(&podToKeep)

	m.CleanupPods([]*api.Pod{&podToKeep})

	removedProbes := []containerPath{{"pod_cleanup", "prober1"}, {"pod_cleanup", "prober2"}}
	expectedProbes := []containerPath{{"pod_keep", "prober1"}, {"pod_keep", "prober2"}}
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
	m.readinessProbes = map[containerPath]*worker{
		containerPath{podUID, probedReady.Name}:   {},
		containerPath{podUID, probedPending.Name}: {},
		containerPath{podUID, probedUnready.Name}: {},
		containerPath{podUID, terminated.Name}:    {},
	}

	m.readinessCache.Set(kubecontainer.ParseContainerID(probedReady.ContainerID), results.Success)
	m.readinessCache.Set(kubecontainer.ParseContainerID(probedUnready.ContainerID), results.Failure)
	m.readinessCache.Set(kubecontainer.ParseContainerID(terminated.ContainerID), results.Success)

	m.UpdatePodStatus(podUID, &podStatus)

	expectedReadiness := map[containerPath]bool{
		containerPath{podUID, unprobed.Name}:      true,
		containerPath{podUID, probedReady.Name}:   true,
		containerPath{podUID, probedPending.Name}: false,
		containerPath{podUID, probedUnready.Name}: false,
		containerPath{podUID, terminated.Name}:    false,
	}
	for _, c := range podStatus.ContainerStatuses {
		expected, ok := expectedReadiness[containerPath{podUID, c.Name}]
		if !ok {
			t.Fatalf("Missing expectation for test case: %v", c.Name)
		}
		if expected != c.Ready {
			t.Errorf("Unexpected readiness for container %v: Expected %v but got %v",
				c.Name, expected, c.Ready)
		}
	}
}

func expectProbes(m *manager, expectedReadinessProbes []containerPath) error {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	var unexpected []containerPath
	missing := make([]containerPath, len(expectedReadinessProbes))
	copy(missing, expectedReadinessProbes)

outer:
	for probePath := range m.readinessProbes {
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
	statusManager := status.NewManager(&testclient.Fake{})
	prober := FakeProber{Readiness: probe.Success}
	return NewManager(probePeriod, statusManager, prober).(*manager)
}

// Wait for the given workers to exit & clean up.
func waitForWorkerExit(m *manager, workerPaths []containerPath) error {
	const interval = 100 * time.Millisecond
	const timeout = 30 * time.Second

	for _, w := range workerPaths {
		condition := func() (bool, error) {
			_, exists := m.getReadinessProbe(w.podUID, w.containerName)
			return !exists, nil
		}
		if exited, _ := condition(); exited {
			continue // Already exited, no need to poll.
		}
		glog.Infof("Polling %v", w)
		if err := wait.Poll(interval, timeout, condition); err != nil {
			return err
		}
	}

	return nil
}
