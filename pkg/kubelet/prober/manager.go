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
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
)

// Manager manages pod probing. It creates a probe "worker" for every container that specifies a
// probe (AddPod). The worker periodically probes its assigned container and caches the results. The
// manager usse the cached probe results to set the appropriate Ready state in the PodStatus when
// requested (UpdatePodStatus). Updating probe parameters is not currently supported.
// TODO: Move liveness probing out of the runtime, to here.
type Manager interface {
	// AddPod creates new probe workers for every container probe. This should be called for every
	// pod created.
	AddPod(pod *api.Pod)

	// RemovePod handles cleaning up the removed pod state, including terminating probe workers and
	// deleting cached results.
	RemovePod(pod *api.Pod)

	// CleanupPods handles cleaning up pods which should no longer be running.
	// It takes a list of "active pods" which should not be cleaned up.
	CleanupPods(activePods []*api.Pod)

	// UpdatePodStatus modifies the given PodStatus with the appropriate Ready state for each
	// container based on container running status, cached probe results and worker states.
	UpdatePodStatus(types.UID, *api.PodStatus)
}

type manager struct {
	// Caches the results of readiness probes.
	readinessCache results.Manager

	// Map of active workers for readiness
	readinessProbes map[containerPath]*worker
	// Lock for accessing & mutating readinessProbes
	workerLock sync.RWMutex

	// The statusManager cache provides pod IP and container IDs for probing.
	statusManager status.Manager

	// prober executes the probe actions.
	prober Prober

	// Default period for workers to execute a probe.
	defaultProbePeriod time.Duration
}

func NewManager(
	defaultProbePeriod time.Duration,
	statusManager status.Manager,
	prober Prober) Manager {
	return &manager{
		defaultProbePeriod: defaultProbePeriod,
		statusManager:      statusManager,
		prober:             prober,
		readinessCache:     results.NewManager(),
		readinessProbes:    make(map[containerPath]*worker),
	}
}

// Key uniquely identifying containers
type containerPath struct {
	podUID        types.UID
	containerName string
}

func (m *manager) AddPod(pod *api.Pod) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	key := containerPath{podUID: pod.UID}
	for _, c := range pod.Spec.Containers {
		key.containerName = c.Name
		if _, ok := m.readinessProbes[key]; ok {
			glog.Errorf("Readiness probe already exists! %v - %v",
				kubecontainer.GetPodFullName(pod), c.Name)
			return
		}
		if c.ReadinessProbe != nil {
			m.readinessProbes[key] = m.newWorker(pod, c)
		}
	}
}

func (m *manager) RemovePod(pod *api.Pod) {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	key := containerPath{podUID: pod.UID}
	for _, c := range pod.Spec.Containers {
		key.containerName = c.Name
		if worker, ok := m.readinessProbes[key]; ok {
			close(worker.stop)
		}
	}
}

func (m *manager) CleanupPods(activePods []*api.Pod) {
	desiredPods := make(map[types.UID]sets.Empty)
	for _, pod := range activePods {
		desiredPods[pod.UID] = sets.Empty{}
	}

	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	for path, worker := range m.readinessProbes {
		if _, ok := desiredPods[path.podUID]; !ok {
			close(worker.stop)
		}
	}
}

func (m *manager) UpdatePodStatus(podUID types.UID, podStatus *api.PodStatus) {
	for i, c := range podStatus.ContainerStatuses {
		var ready bool
		if c.State.Running == nil {
			ready = false
		} else if result, ok := m.readinessCache.Get(
			kubecontainer.ParseContainerID(c.ContainerID)); ok {
			ready = result == results.Success
		} else {
			// The check whether there is a probe which hasn't run yet.
			_, exists := m.getReadinessProbe(podUID, c.Name)
			ready = !exists
		}
		podStatus.ContainerStatuses[i].Ready = ready
	}
}

func (m *manager) getReadinessProbe(podUID types.UID, containerName string) (*worker, bool) {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()
	probe, ok := m.readinessProbes[containerPath{podUID, containerName}]
	return probe, ok
}

// Called by the worker after exiting.
func (m *manager) removeReadinessProbe(podUID types.UID, containerName string) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	delete(m.readinessProbes, containerPath{podUID, containerName})
}
