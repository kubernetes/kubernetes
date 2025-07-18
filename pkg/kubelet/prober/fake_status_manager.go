/*
Copyright 2015 The Kubernetes Authors.

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
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// fakeStatusManager implements status.Manager and status.PodStatusProvider for testing
type fakeStatusManager struct {
	podStatuses map[types.UID]v1.PodStatus
}

// GetPodStatus returns the cached status for the provided pod UID
func (m *fakeStatusManager) GetPodStatus(podUID types.UID) (v1.PodStatus, bool) {
	status, ok := m.podStatuses[podUID]
	return status, ok
}

// SetPodStatus updates the cached status for the given pod
func (m *fakeStatusManager) SetPodStatus(_ klog.Logger, pod *v1.Pod, status v1.PodStatus) {
	m.podStatuses[pod.UID] = status
}

// SetContainerStartup updates the cached container status with the given startup
func (m *fakeStatusManager) SetContainerStartup(_ klog.Logger, _ types.UID, _ kubecontainer.ContainerID, _ bool) {
}

// SetContainerReadiness updates the cached container status with the given readiness
func (m *fakeStatusManager) SetContainerReadiness(_ klog.Logger, _ types.UID, _ kubecontainer.ContainerID, _ bool) {
}

// GetPodResizeConditions returns cached PodStatus Resize conditions
func (m *fakeStatusManager) GetPodResizeConditions(_ types.UID) []*v1.PodCondition {
	return nil
}

// SetPodResizePendingCondition caches the last PodResizePending condition
func (m *fakeStatusManager) SetPodResizePendingCondition(_ types.UID, _ string, _ string, _ int64) {
}

// SetPodResizeInProgressCondition caches the last PodResizeInProgress condition
func (m *fakeStatusManager) SetPodResizeInProgressCondition(_ types.UID, _ string, _ string, _ int64) {
}

// ClearPodResizeInProgressCondition clears the PodResizeInProgress condition
func (m *fakeStatusManager) ClearPodResizeInProgressCondition(_ types.UID) bool {
	return false
}

// ClearPodResizePendingCondition clears the PodResizePending condition
func (m *fakeStatusManager) ClearPodResizePendingCondition(_ types.UID) {
}

// IsPodResizeDeferred returns true if the pod resize is deferred
func (m *fakeStatusManager) IsPodResizeDeferred(_ types.UID) bool {
	return false
}

// IsPodResizeInfeasible returns true if the pod resize is infeasible
func (m *fakeStatusManager) IsPodResizeInfeasible(_ types.UID) bool {
	return false
}

// RemoveOrphanedStatuses removes statuses for pods not in the provided podUIDs
func (m *fakeStatusManager) RemoveOrphanedStatuses(_ klog.Logger, _ map[types.UID]bool) {
}

// Start is a no-op for test purposes
func (m *fakeStatusManager) Start(_ context.Context) {
}

// TerminatePod resets the container status to terminated
func (m *fakeStatusManager) TerminatePod(_ klog.Logger, pod *v1.Pod) {
	delete(m.podStatuses, pod.UID)
}
