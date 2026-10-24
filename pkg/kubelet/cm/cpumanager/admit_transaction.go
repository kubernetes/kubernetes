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
	"k8s.io/apimachinery/pkg/api/resource"
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
// It becomes effective only when the admission is committed.
type resizeTransaction struct {
	podUID  string
	scratch state.State
}

// admissionState returns the state the policy allocates on while admitting the pod.
// A resize gets the scratch state of its transaction, which is started on first use.
// The caller must hold m.Mutex.
func (m *manager) admissionState(logger klog.Logger, pod *v1.Pod, operation lifecycle.Operation) state.State {
	if operation != lifecycle.ResizeOperation {
		// An added pod first gets its pending resize resolved.
		// This covers scenarion in which kubelet was restarted during resize and the pod is admitted again with Add after restart.
		// If the resize was staged and allocation manager commmited new resources, we should apply those resources here.
		// If kubelet was restarted before allocation manager committed new resources, pending resize must be cleared,
		// as the Resize operation will be processed again.
		if operation == lifecycle.AddOperation {
			m.resolvePendingResize(logger, pod)
		}
		return m.state
	}
	podUID := string(pod.UID)
	if m.pendingResize != nil {
		if m.pendingResize.podUID == podUID {
			return m.pendingResize.scratch
		}
		// The allocation manager admits one pod at a time and always commits or
		// aborts, so this is a bug; the stale transaction must not leak into this one.
		logger.Error(nil, "Discarding a resize transaction that was neither committed nor aborted", "pod", klog.KObj(pod), "stalePodUID", m.pendingResize.podUID)
	}
	m.pendingResize = &resizeTransaction{
		podUID: podUID,
		// The scratch state logs every change at info level, while only the
		// changes that get committed are worth it.
		scratch: cloneState(logger.V(5), m.state),
	}
	return m.pendingResize.scratch
}

// StageAdmit records the resize prepared for the pod in the state as a pending resize, without applying it.
func (m *manager) StageAdmit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation) error {
	if operation != lifecycle.ResizeOperation {
		return nil
	}
	m.Lock()
	defer m.Unlock()

	tx, err := m.getResizeTransaction(pod)
	if err != nil || tx == nil {
		return err
	}
	pending, changed := m.buildPendingResize(tx, pod)
	if !changed {
		return nil
	}
	if err := m.checkCPUsFree(pod, pending); err != nil {
		return err
	}
	m.state.SetPendingResize(tx.podUID, pending)
	return nil
}

// CommitAdmit applies the pending resize of the pod to the state.
func (m *manager) CommitAdmit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation) error {
	if operation != lifecycle.ResizeOperation {
		return nil
	}
	m.Lock()
	defer m.Unlock()

	tx, err := m.getResizeTransaction(pod)
	if err != nil || tx == nil {
		return err
	}
	m.pendingResize = nil

	pending, staged := m.state.GetPendingResize(tx.podUID)
	if !staged {
		if _, changed := m.buildPendingResize(tx, pod); changed {
			return fmt.Errorf("cannot commit the resize of pod %s (UID %s): it was not staged", klog.KObj(pod), pod.UID)
		}
		return nil
	}
	// Other pods may have released CPUs since the resize was staged, which the
	// commit keeps, but none may have taken the CPUs the resize takes.
	if err := m.checkCPUsFree(pod, pending); err != nil {
		m.state.DeletePendingResize(tx.podUID)
		return err
	}
	m.state.CommitPendingResize(tx.podUID)
	return nil
}

// AbortAdmit discards the resize prepared for the pod and its pending resize.
func (m *manager) AbortAdmit(ctx context.Context, pod *v1.Pod, operation lifecycle.Operation) {
	if operation != lifecycle.ResizeOperation {
		return
	}
	m.Lock()
	defer m.Unlock()

	if m.pendingResize != nil && m.pendingResize.podUID == string(pod.UID) {
		m.pendingResize = nil
	}
	if _, staged := m.state.GetPendingResize(string(pod.UID)); staged {
		m.state.DeletePendingResize(string(pod.UID))
	}
}

// resolvePendingResize completes or discards the resize of the pod staged
// before a kubelet restart. An added pod carries its persisted allocation,
// which includes the resize only if the resize was accepted.
// The caller must hold m.Mutex.
func (m *manager) resolvePendingResize(logger klog.Logger, pod *v1.Pod) {
	pending, ok := m.state.GetPendingResize(string(pod.UID))
	if !ok {
		return
	}
	if err := m.checkPendingResize(pod, pending); err != nil {
		logger.Info("Discarding the pending resize of the pod", "pod", klog.KObj(pod), "podUID", pod.UID, "reason", err)
		m.state.DeletePendingResize(string(pod.UID))
		return
	}
	logger.Info("Completing the pending resize of the pod, its allocation includes the resize", "pod", klog.KObj(pod), "podUID", pod.UID)
	m.state.CommitPendingResize(string(pod.UID))
}

// checkPendingResize returns why the pending resize cannot be committed for
// the pod with its allocated resources, or nil if it can.
//
// The static policy assigns a container as many exclusive CPUs as its integral
// CPU request, and a resize changes them only together with the request. So the
// size of a target CPU set tells whether the persisted allocation includes the
// resize. The same holds for the pod-level assignment and the pod CPU request.
// The caller must hold m.Mutex.
func (m *manager) checkPendingResize(pod *v1.Pod, pending state.PendingResize) error {
	for containerName, target := range pending.Containers {
		container := findContainer(pod, containerName)
		if container == nil {
			return fmt.Errorf("container %q is not part of the pod", containerName)
		}
		if request := container.Resources.Requests[v1.ResourceCPU]; !requestsCPUs(request, target) {
			return fmt.Errorf("container %q is allocated %s CPUs, the resize assigns it %d", containerName, request.String(), target.Size())
		}
	}
	if pending.Pod != nil {
		if request := podCPURequest(pod); !requestsCPUs(request, *pending.Pod) {
			return fmt.Errorf("the pod is allocated %s CPUs, the resize assigns it %d", request.String(), pending.Pod.Size())
		}
	}
	return m.checkCPUsFree(pod, pending)
}

// checkCPUsFree fails if a CPU the pending resize takes from the default CPU
// set is no longer free. The caller must hold m.Mutex.
func (m *manager) checkCPUsFree(pod *v1.Pod, pending state.PendingResize) error {
	taken, _ := state.PendingResizeDelta(m.state.GetCPUAssignments(), m.state.GetPodCPUAssignments(), string(pod.UID), pending)
	if free := m.state.GetDefaultCPUSet(); !taken.IsSubsetOf(free) {
		return fmt.Errorf("cannot resize pod %s (UID %s): CPUs %s are no longer free", klog.KObj(pod), pod.UID, taken.Difference(free))
	}
	return nil
}

// buildPendingResize returns the pending resize that commits the transaction,
// and false if the transaction changes no assignment.
// The caller must hold m.Mutex.
func (m *manager) buildPendingResize(tx *resizeTransaction, pod *v1.Pod) (state.PendingResize, bool) {
	var pending state.PendingResize
	for container := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		target, ok := tx.scratch.GetCPUSet(tx.podUID, container.Name)
		if !ok {
			continue
		}
		if current, _ := m.state.GetCPUSet(tx.podUID, container.Name); current.Equals(target) {
			continue
		}
		if pending.Containers == nil {
			pending.Containers = make(map[string]cpuset.CPUSet)
		}
		pending.Containers[container.Name] = target
	}
	if target, ok := tx.scratch.GetPodCPUSet(tx.podUID); ok {
		if current, _ := m.state.GetPodCPUSet(tx.podUID); !current.Equals(target) {
			pending.Pod = &target
		}
	}
	return pending, len(pending.Containers) > 0 || pending.Pod != nil
}

func findContainer(pod *v1.Pod, containerName string) *v1.Container {
	for container := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		if container.Name == containerName {
			return container
		}
	}
	return nil
}

func podCPURequest(pod *v1.Pod) resource.Quantity {
	if pod.Spec.Resources == nil {
		return resource.Quantity{}
	}
	return pod.Spec.Resources.Requests[v1.ResourceCPU]
}

// requestsCPUs returns whether the CPU request is as many whole CPUs as the CPU set has.
func requestsCPUs(request resource.Quantity, cset cpuset.CPUSet) bool {
	return request.MilliValue() == int64(cset.Size())*1000
}

// getResizeTransaction returns the resize transaction of the pod, or nil if
// nothing was prepared for it. The caller must hold m.Mutex.
func (m *manager) getResizeTransaction(pod *v1.Pod) (*resizeTransaction, error) {
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
