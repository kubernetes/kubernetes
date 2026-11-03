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

	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
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
// An update, such as a resize, gets the scratch state of its transaction, which is
// started on first use. A dry run of the admission only computes hints on it, and
// always ends with lifecycle.AdmitFailure, which drops it.
// The caller must hold m.Mutex.
func (m *manager) admissionStateLocked(logger klog.Logger, pod *v1.Pod, operation lifecycle.Operation) state.State {
	if operation != lifecycle.UpdateOperation {
		return m.state
	}
	podUID := string(pod.UID)
	if m.pendingResize != nil {
		if m.pendingResize.podUID == podUID {
			return m.pendingResize.scratch
		}
		// The allocation manager admits one pod at a time and completes each
		// admission, so this is a bug; the stale transaction must not leak into this one.
		logger.Error(nil, "Discarding a resize transaction that was not completed", "pod", klog.KObj(pod), "stalePodUID", m.pendingResize.podUID)
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
// the resize prepared on the scratch state to the state with a single write,
// otherwise it only drops the scratch state. A failure is an implementation
// bug, so it is only logged.
func (m *manager) CompleteAdmit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation, result lifecycle.AdmitResult) {
	if operation != lifecycle.UpdateOperation {
		return
	}
	logger := klog.FromContext(ctx)
	m.Lock()
	defer m.Unlock()

	tx, err := m.getResizeTransactionLocked(pod)
	if err != nil {
		// The allocation manager admits one pod at a time and completes each
		// admission, so this is a bug; the stale transaction must not leak into the next one.
		logger.Error(err, "Discarding a resize transaction that was not completed", "pod", klog.KObj(pod), "result", result)
		m.pendingResize = nil
		return
	}
	if tx == nil {
		return
	}
	m.pendingResize = nil
	if result != lifecycle.AdmitSuccess {
		return
	}

	pending, changed := m.buildPendingResizeLocked(tx, pod)
	if !changed {
		return
	}
	// Other pods may have released CPUs since the resize was prepared, which the
	// commit keeps, but none may have taken the CPUs the resize takes.
	if err := m.checkCPUsFree(pending); err != nil {
		logger.Error(err, "Cannot commit the resize, the CPU assignment stays unchanged", "pod", klog.KObj(pod))
		return
	}
	m.state.CommitPendingResize(pending)
}

// checkCPUsFree fails if a CPU the pending resize takes from the default CPU
// set is no longer free. The state validation would not catch it, since it
// does not check that the assignments of different pods are disjoint.
func (m *manager) checkCPUsFree(pending state.PendingResize) error {
	taken, _ := state.PendingResizeDelta(m.state.GetCPUAssignments(), m.state.GetPodCPUAssignments(), pending)
	if free := m.state.GetDefaultCPUSet(); !taken.IsSubsetOf(free) {
		return fmt.Errorf("cannot resize pod UID %s: CPUs %s are no longer free", pending.PodUID, taken.Difference(free))
	}
	return nil
}

// buildPendingResizeLocked returns the pending resize that commits the transaction,
// and false if the transaction changes no assignment.
// The caller must hold m.Mutex.
func (m *manager) buildPendingResizeLocked(tx *resizeTransaction, pod *v1.Pod) (state.PendingResize, bool) {
	pending := state.PendingResize{PodUID: tx.podUID}
	for container := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		containerTarget, ok := tx.scratch.GetCPUSet(tx.podUID, container.Name)
		if !ok {
			continue
		}
		if containerCurrent, _ := m.state.GetCPUSet(tx.podUID, container.Name); containerCurrent.Equals(containerTarget) {
			continue
		}
		if pending.ContainerCPUSets == nil {
			pending.ContainerCPUSets = make(map[string]cpuset.CPUSet)
		}
		pending.ContainerCPUSets[container.Name] = containerTarget
	}
	if podTarget, ok := tx.scratch.GetPodCPUSet(tx.podUID); ok {
		if podCurrent, _ := m.state.GetPodCPUSet(tx.podUID); !podCurrent.Equals(podTarget) {
			pending.PodCPUSet = &podTarget
		}
	}
	return pending, len(pending.ContainerCPUSets) > 0 || pending.PodCPUSet != nil
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

// cloneState returns an in-memory copy of the state.
func cloneState(logger klog.Logger, s state.Reader) state.State {
	c := state.NewMemoryState(logger)
	c.SetCPUAssignments(s.GetCPUAssignments())
	c.SetPodCPUAssignments(s.GetPodCPUAssignments())
	c.SetCPUBaselines(s.GetCPUBaselines())
	c.SetDefaultCPUSet(s.GetDefaultCPUSet())
	return c
}
