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

package cpumanager

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/utils/cpuset"
)

var _ topologymanager.AdmitTransactionalHintProvider = &manager{}

// resizeTransaction is the resize of a pod that is being admitted.
// The policy allocates it on a scratch copy of the state.
type resizeTransaction struct {
	podUID  string
	scratch state.State
}

// admissionStateLocked returns the state the policy allocates on while admitting the pod.
// A resize gets the scratch state of its transaction, which is started on first use.
// The caller must hold m.Mutex.
func (m *manager) admissionStateLocked(logger klog.Logger, pod *v1.Pod, operation lifecycle.Operation) state.State {
	if operation != lifecycle.ResizeOperation {
		return m.state
	}
	podUID := string(pod.UID)
	if m.pendingResize != nil {
		if m.pendingResize.podUID == podUID {
			return m.pendingResize.scratch
		}
		// The allocation manager admits one pod at a time and completes each
		// admission, so this is a bug; the stale transaction must not leak into this one.
		logger.Info("Discarding a resize transaction that was not completed", "pod", klog.KObj(pod), "stalePodUID", m.pendingResize.podUID)
	}
	m.pendingResize = &resizeTransaction{
		podUID: podUID,
		// The scratch state logs every change at info level, while only the
		// changes that get committed are worth it.
		scratch: cloneState(logger.V(5), m.state),
	}
	return m.pendingResize.scratch
}

// CompleteAdmit ends the resize transaction of the pod. On success, it applies
// the resize prepared on the scratch state to the state, otherwise it only
// drops the scratch state. A failure is an implementation bug, so it is
// only logged.
func (m *manager) CompleteAdmit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation, result lifecycle.AdmitResult) {
	if operation != lifecycle.ResizeOperation {
		return
	}
	logger := klog.FromContext(ctx)
	m.Lock()
	defer m.Unlock()

	tx, err := m.getResizeTransactionLocked(pod)
	if err != nil {
		logger.Error(err, "Cannot complete the resize admission", "pod", klog.KObj(pod), "result", result)
		return
	}
	if tx == nil {
		return
	}
	m.pendingResize = nil
	if result != lifecycle.AdmitSuccess {
		return
	}

	added, released, err := m.resizeDeltaLocked(tx)
	if err != nil {
		logger.Error(err, "Cannot commit the resize, the CPU assignment stays unchanged", "pod", klog.KObj(pod))
		return
	}
	for containerName, cset := range tx.scratch.GetCPUAssignments()[tx.podUID] {
		if current, ok := m.state.GetCPUSet(tx.podUID, containerName); !ok || !current.Equals(cset) {
			m.state.SetCPUSet(tx.podUID, containerName, cset)
		}
	}
	if cset, ok := tx.scratch.GetPodCPUSet(tx.podUID); ok {
		if current, ok := m.state.GetPodCPUSet(tx.podUID); !ok || !current.Equals(cset) {
			m.state.SetPodCPUSet(tx.podUID, cset)
		}
	}
	// Other pods may have released CPUs since the resize was prepared, so the
	// default CPU set is updated by the difference instead of being copied.
	if !added.IsEmpty() || !released.IsEmpty() {
		m.state.SetDefaultCPUSet(m.state.GetDefaultCPUSet().Difference(added).Union(released))
	}
}

// getResizeTransactionLocked returns the resize transaction of the pod, or nil if
// nothing was prepared for it. The caller must hold m.Mutex.
func (m *manager) getResizeTransactionLocked(pod *v1.Pod) (*resizeTransaction, error) {
	if m.pendingResize == nil {
		return nil, nil
	}
	if m.pendingResize.podUID != string(pod.UID) {
		return nil, fmt.Errorf("the resize transaction in progress belongs to pod UID %s, not to pod %s (UID %s)", m.pendingResize.podUID, klog.KObj(pod), pod.UID)
	}
	return m.pendingResize, nil
}

// resizeDeltaLocked returns the CPUs that committing the transaction takes from the
// default CPU set and returns to it. It fails if a CPU to be taken is no
// longer free. The caller must hold m.Mutex.
func (m *manager) resizeDeltaLocked(tx *resizeTransaction) (added, released cpuset.CPUSet, err error) {
	before := podCPUs(m.state, tx.podUID)
	after := podCPUs(tx.scratch, tx.podUID)
	added = after.Difference(before)
	released = before.Difference(after)
	if free := m.state.GetDefaultCPUSet(); !added.IsSubsetOf(free) {
		return cpuset.New(), cpuset.New(), fmt.Errorf("cannot commit the resize of pod UID %s: CPUs %s are no longer free", tx.podUID, added.Difference(free))
	}
	return added, released, nil
}

// podCPUs returns all the CPUs assigned to the pod and its containers.
func podCPUs(s state.Reader, podUID string) cpuset.CPUSet {
	cpus, _ := s.GetPodCPUSet(podUID)
	for _, cset := range s.GetCPUAssignments()[podUID] {
		cpus = cpus.Union(cset)
	}
	return cpus
}

// cloneState returns an in-memory copy of the state.
func cloneState(logger klog.Logger, s state.Reader) state.State {
	c := state.NewMemoryState(logger)
	c.SetCPUAssignments(s.GetCPUAssignments())
	c.SetPodCPUAssignments(s.GetPodCPUAssignments())
	c.SetCPUBaselines(s.GetCPUBaselines())
	c.SetDefaultCPUSet(s.GetDefaultCPUSet())
	return c
}
