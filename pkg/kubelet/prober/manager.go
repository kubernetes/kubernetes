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
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubeutil "k8s.io/kubernetes/pkg/kubelet/util"
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
	// Map of active workers for probes
	workers map[probeKey]*worker
	// Lock for accessing & mutating workers
	workerLock sync.RWMutex

	// The statusManager cache provides pod IP and container IDs for probing.
	statusManager status.Manager

	// readinessManager manages the results of readiness probes
	readinessManager results.Manager

	// livenessManager manages the results of liveness probes
	livenessManager results.Manager

	// prober executes the probe actions.
	prober *prober

	// Default period for workers to execute a probe.
	defaultProbePeriod time.Duration
}

func NewManager(
	defaultProbePeriod time.Duration,
	statusManager status.Manager,
	readinessManager results.Manager,
	livenessManager results.Manager,
	runner kubecontainer.ContainerCommandRunner,
	refManager *kubecontainer.RefManager,
	recorder record.EventRecorder) Manager {
	prober := newProber(runner, refManager, recorder)
	return &manager{
		defaultProbePeriod: defaultProbePeriod,
		statusManager:      statusManager,
		prober:             prober,
		readinessManager:   readinessManager,
		livenessManager:    livenessManager,
		workers:            make(map[probeKey]*worker),
	}
}

// Key uniquely identifying container probes
type probeKey struct {
	podUID        types.UID
	containerName string
	probeType     probeType
}

// Type of probe (readiness or liveness)
type probeType int

const (
	liveness probeType = iota
	readiness
)

// For debugging.
func (t probeType) String() string {
	switch t {
	case readiness:
		return "Readiness"
	case liveness:
		return "Liveness"
	default:
		return "UNKNOWN"
	}
}

func (m *manager) AddPod(pod *api.Pod) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	key := probeKey{podUID: pod.UID}
	for _, c := range pod.Spec.Containers {
		key.containerName = c.Name

		if c.ReadinessProbe != nil {
			key.probeType = readiness
			if _, ok := m.workers[key]; ok {
				glog.Errorf("Readiness probe already exists! %v - %v",
					kubeutil.FormatPodName(pod), c.Name)
				return
			}
			w := newWorker(m, readiness, pod, c)
			m.workers[key] = w
			go w.run()
		}

		if c.LivenessProbe != nil {
			key.probeType = liveness
			if _, ok := m.workers[key]; ok {
				glog.Errorf("Liveness probe already exists! %v - %v",
					kubeutil.FormatPodName(pod), c.Name)
				return
			}
			w := newWorker(m, liveness, pod, c)
			m.workers[key] = w
			go w.run()
		}
	}
}

func (m *manager) RemovePod(pod *api.Pod) {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	key := probeKey{podUID: pod.UID}
	for _, c := range pod.Spec.Containers {
		key.containerName = c.Name
		for _, probeType := range [...]probeType{readiness, liveness} {
			key.probeType = probeType
			if worker, ok := m.workers[key]; ok {
				close(worker.stop)
			}
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

	for key, worker := range m.workers {
		if _, ok := desiredPods[key.podUID]; !ok {
			close(worker.stop)
		}
	}
}

func (m *manager) UpdatePodStatus(podUID types.UID, podStatus *api.PodStatus) {
	for i, c := range podStatus.ContainerStatuses {
		var ready bool
		if c.State.Running == nil {
			ready = false
		} else if result, ok := m.readinessManager.Get(kubecontainer.ParseContainerID(c.ContainerID)); ok {
			ready = result == results.Success
		} else {
			// The check whether there is a probe which hasn't run yet.
			_, exists := m.getWorker(podUID, c.Name, readiness)
			ready = !exists
		}
		podStatus.ContainerStatuses[i].Ready = ready
	}
}

func (m *manager) getWorker(podUID types.UID, containerName string, probeType probeType) (*worker, bool) {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()
	worker, ok := m.workers[probeKey{podUID, containerName, probeType}]
	return worker, ok
}

// Called by the worker after exiting.
func (m *manager) removeWorker(podUID types.UID, containerName string, probeType probeType) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	delete(m.workers, probeKey{podUID, containerName, probeType})
}
