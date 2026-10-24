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
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"

	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/utils/cpuset"
)

// resizingPolicy assigns the CPUs in resized to the containers it allocates,
// and the CPUs in resizedPod to the pods it allocates, moving the difference
// between them and the default CPU set.
type resizingPolicy struct {
	mockPolicy
	resized    map[string]cpuset.CPUSet
	resizedPod cpuset.CPUSet
	// hintsSaw records the CPUs of the pod visible to each GetTopologyHints call.
	hintsSaw []cpuset.CPUSet
}

func (p *resizingPolicy) Allocate(_ klog.Logger, s state.State, pod *v1.Pod, container *v1.Container, _ lifecycle.Operation) error {
	target, ok := p.resized[container.Name]
	if !ok {
		return nil
	}
	current, _ := s.GetCPUSet(string(pod.UID), container.Name)
	s.SetCPUSet(string(pod.UID), container.Name, target)
	s.SetDefaultCPUSet(s.GetDefaultCPUSet().Union(current.Difference(target)).Difference(target))
	return nil
}

func (p *resizingPolicy) AllocatePod(_ klog.Logger, s state.State, pod *v1.Pod, _ lifecycle.Operation) error {
	if p.resizedPod.IsEmpty() {
		return nil
	}
	current, _ := s.GetPodCPUSet(string(pod.UID))
	s.SetPodCPUSet(string(pod.UID), p.resizedPod)
	s.SetDefaultCPUSet(s.GetDefaultCPUSet().Union(current.Difference(p.resizedPod)).Difference(p.resizedPod))
	return nil
}

func (p *resizingPolicy) GetTopologyHints(_ klog.Logger, s state.State, pod *v1.Pod, _ *v1.Container, _ lifecycle.Operation) map[string][]topologymanager.TopologyHint {
	p.hintsSaw = append(p.hintsSaw, assignedPodCPUs(s, string(pod.UID)))
	return nil
}

// assignedPodCPUs returns all the CPUs assigned to the pod and its containers.
func assignedPodCPUs(s state.Reader, podUID string) cpuset.CPUSet {
	cpus, _ := s.GetPodCPUSet(podUID)
	for _, cset := range s.GetCPUAssignments()[podUID] {
		cpus = cpus.Union(cset)
	}
	return cpus
}

const otherPodUID = "other-pod"

func newResizeTestPod(uid types.UID) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: uid, Name: string(uid), Namespace: "ns"},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "c1"}, {Name: "c2"}}},
	}
}

// newResizeTestManager returns a manager where the pod has c1={1,2} and
// c2={3}, another pod has {7} and {4,5,6} are free.
func newResizeTestManager(logger klog.Logger, policy Policy, pods ...*v1.Pod) *manager {
	s := state.NewMemoryState(logger)
	s.SetCPUSet(string(pods[0].UID), "c1", cpuset.New(1, 2))
	s.SetCPUSet(string(pods[0].UID), "c2", cpuset.New(3))
	s.SetCPUSet(otherPodUID, "c", cpuset.New(7))
	s.SetDefaultCPUSet(cpuset.New(4, 5, 6))
	return &manager{
		policy:          policy,
		state:           s,
		lastUpdateState: state.NewMemoryState(logger),
		containerMap:    containermap.NewContainerMap(),
		sourcesReady:    &sourcesReadyStub{},
		activePods:      func() []*v1.Pod { return pods },
	}
}

func TestResizeAdmitTransaction(t *testing.T) {

	tests := []struct {
		name    string
		resized map[string]cpuset.CPUSet
		// between changes the state after the resize is prepared, as other pods would.
		between         func(s state.State)
		abort           bool
		expectStageErr  bool
		expectCommitErr bool
		expectC1        cpuset.CPUSet
		expectC2        cpuset.CPUSet
		expectDefault   cpuset.CPUSet
	}{
		{
			name:          "scale up is committed",
			resized:       map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
			expectC1:      cpuset.New(1, 2, 4),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:          "scale down is committed",
			resized:       map[string]cpuset.CPUSet{"c1": cpuset.New(1)},
			expectC1:      cpuset.New(1),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(2, 4, 5, 6),
		},
		{
			name:          "abort leaves the state untouched",
			resized:       map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
			abort:         true,
			expectC1:      cpuset.New(1, 2),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name:    "commit keeps the CPUs released meanwhile",
			resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
			between: func(s state.State) {
				s.Delete(otherPodUID, "c")
				s.SetDefaultCPUSet(s.GetDefaultCPUSet().Union(cpuset.New(7)))
			},
			expectC1:      cpuset.New(1, 2, 4),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(5, 6, 7),
		},
		{
			name:    "a CPU taken meanwhile fails the stage and the commit",
			resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
			between: func(s state.State) {
				s.SetCPUSet(otherPodUID, "c", cpuset.New(4, 7))
				s.SetDefaultCPUSet(s.GetDefaultCPUSet().Difference(cpuset.New(4)))
			},
			expectStageErr:  true,
			expectCommitErr: true,
			expectC1:        cpuset.New(1, 2),
			expectC2:        cpuset.New(3),
			expectDefault:   cpuset.New(5, 6),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			pod := newResizeTestPod("pod")
			policy := &resizingPolicy{resized: tc.resized}
			m := newResizeTestManager(logger, policy, pod)

			for i := range pod.Spec.Containers {
				m.GetTopologyHints(logger, pod, &pod.Spec.Containers[i], lifecycle.ResizeOperation)
				require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[i], lifecycle.ResizeOperation))
			}
			c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
			require.True(t, c1.Equals(cpuset.New(1, 2)), "the resize must not change the state before it is committed, c1 is %s", c1)
			require.True(t, policy.hintsSaw[1].Equals(assignedPodCPUs(m.pendingResize.scratch, string(pod.UID))),
				"the hints of c2 must see what was prepared for c1, saw %s", policy.hintsSaw[1])

			if tc.between != nil {
				tc.between(m.state)
			}
			err := m.StageAdmit(ctx, pod, lifecycle.ResizeOperation)
			require.Equal(t, tc.expectStageErr, err != nil, "stage error: %v", err)
			_, staged := m.state.GetPendingResize(string(pod.UID))
			require.Equal(t, !tc.expectStageErr, staged, "a staged resize must be recorded as a pending resize")
			if tc.abort {
				m.AbortAdmit(ctx, pod, lifecycle.ResizeOperation)
			}
			err = m.CommitAdmit(ctx, pod, lifecycle.ResizeOperation)
			require.Equal(t, tc.expectCommitErr, err != nil, "commit error: %v", err)

			c1, _ = m.state.GetCPUSet(string(pod.UID), "c1")
			require.True(t, tc.expectC1.Equals(c1), "expected c1 %s, got %s", tc.expectC1, c1)
			c2, _ := m.state.GetCPUSet(string(pod.UID), "c2")
			require.True(t, tc.expectC2.Equals(c2), "expected c2 %s, got %s", tc.expectC2, c2)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			require.Nil(t, m.pendingResize, "the transaction must end with the commit or the abort")
			_, staged = m.state.GetPendingResize(string(pod.UID))
			require.False(t, staged, "the pending resize must end with the commit or the abort")
		})
	}
}

func TestResizeAdmitTransactionPodLevel(t *testing.T) {
	tests := []struct {
		name          string
		resizedPod    cpuset.CPUSet
		abort         bool
		expectPod     cpuset.CPUSet
		expectDefault cpuset.CPUSet
	}{
		{
			name:          "scale up is committed",
			resizedPod:    cpuset.New(1, 2, 3, 4, 5),
			expectPod:     cpuset.New(1, 2, 3, 4, 5),
			expectDefault: cpuset.New(6),
		},
		{
			name:          "scale down is committed",
			resizedPod:    cpuset.New(1, 2, 3),
			expectPod:     cpuset.New(1, 2, 3),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name:          "scale up is aborted",
			resizedPod:    cpuset.New(1, 2, 3, 4, 5),
			abort:         true,
			expectPod:     cpuset.New(1, 2, 3, 4),
			expectDefault: cpuset.New(5, 6),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			pod := newResizeTestPod("pod")
			m := newResizeTestManager(logger, &resizingPolicy{resizedPod: tc.resizedPod}, pod)
			// The pod-level assignment holds the CPUs of its containers and CPU 4,
			// which none of them uses.
			m.state.SetPodCPUSet(string(pod.UID), cpuset.New(1, 2, 3, 4))
			m.state.SetDefaultCPUSet(cpuset.New(5, 6))

			m.GetPodTopologyHints(logger, pod, lifecycle.ResizeOperation)
			require.NoError(t, m.AllocatePod(logger, pod, lifecycle.ResizeOperation))
			podSet, _ := m.state.GetPodCPUSet(string(pod.UID))
			require.True(t, podSet.Equals(cpuset.New(1, 2, 3, 4)), "the resize must not change the state before it is committed, the pod has %s", podSet)

			require.NoError(t, m.StageAdmit(ctx, pod, lifecycle.ResizeOperation))
			_, staged := m.state.GetPendingResize(string(pod.UID))
			require.True(t, staged, "a staged resize must be recorded as a pending resize")
			if tc.abort {
				m.AbortAdmit(ctx, pod, lifecycle.ResizeOperation)
			}
			require.NoError(t, m.CommitAdmit(ctx, pod, lifecycle.ResizeOperation))

			podSet, _ = m.state.GetPodCPUSet(string(pod.UID))
			require.True(t, tc.expectPod.Equals(podSet), "expected pod %s, got %s", tc.expectPod, podSet)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			require.Nil(t, m.pendingResize, "the transaction must end with the commit or the abort")
			_, staged = m.state.GetPendingResize(string(pod.UID))
			require.False(t, staged, "the pending resize must end with the commit or the abort")
		})
	}
}

func TestResizeAdmitTransactionOfAnotherPod(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	another := newResizeTestPod("another")
	m := newResizeTestManager(logger, &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}, pod, another)

	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.ResizeOperation))
	// The transaction of pod is never ended, so the one of another replaces it.
	require.NoError(t, m.Allocate(ctx, another, &another.Spec.Containers[1], lifecycle.ResizeOperation))
	require.Equal(t, string(another.UID), m.pendingResize.podUID)

	require.Error(t, m.StageAdmit(ctx, pod, lifecycle.ResizeOperation))
	require.Error(t, m.CommitAdmit(ctx, pod, lifecycle.ResizeOperation))
	m.AbortAdmit(ctx, pod, lifecycle.ResizeOperation)
	require.NotNil(t, m.pendingResize, "an abort for another pod must not end the transaction")

	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2)), "the discarded resize must not be committed, c1 is %s", c1)
}

func TestAddAdmitWithoutTransaction(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	m := newResizeTestManager(logger, &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}, pod)

	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
	require.Nil(t, m.pendingResize)
	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2, 4)), "the allocation must be applied to the state, c1 is %s", c1)

	require.NoError(t, m.StageAdmit(ctx, pod, lifecycle.AddOperation))
	require.NoError(t, m.CommitAdmit(ctx, pod, lifecycle.AddOperation))
	m.AbortAdmit(ctx, pod, lifecycle.AddOperation)
	c1, _ = m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2, 4)), "the transaction must not change the allocation, c1 is %s", c1)
}

func TestResolvePendingResize(t *testing.T) {
	// The resize staged before the restart grows c1 from {1,2} to {1,2,4}.
	pending := state.PendingResize{
		Containers: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
	}

	tests := []struct {
		name string
		// allocatedC1 is the CPU request of c1 in the persisted pod allocation.
		allocatedC1 string
		// before changes the restored state before the pod is added again.
		before        func(s state.State)
		expectC1      cpuset.CPUSet
		expectDefault cpuset.CPUSet
	}{
		{
			name:        "the resize is staged for a container the pod does not have",
			allocatedC1: "3",
			before: func(s state.State) {
				gone := pending.Clone()
				gone.Containers["gone"] = gone.Containers["c1"]
				delete(gone.Containers, "c1")
				s.SetPendingResize("pod", gone)
			},
			expectC1:      cpuset.New(1, 2),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name:          "the allocation includes the resize",
			allocatedC1:   "3",
			expectC1:      cpuset.New(1, 2, 4),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:          "the allocation does not include the resize",
			allocatedC1:   "2",
			expectC1:      cpuset.New(1, 2),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name:        "the CPUs taken are computed from the restored assignment",
			allocatedC1: "3",
			before: func(s state.State) {
				s.SetCPUSet("pod", "c1", cpuset.New(1))
				s.SetDefaultCPUSet(cpuset.New(2, 4, 5, 6))
			},
			expectC1:      cpuset.New(1, 2, 4),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:        "a CPU the resize takes is no longer free",
			allocatedC1: "3",
			before: func(s state.State) {
				s.SetCPUSet(otherPodUID, "c", cpuset.New(4, 7))
				s.SetDefaultCPUSet(cpuset.New(5, 6))
			},
			expectC1:      cpuset.New(1, 2),
			expectDefault: cpuset.New(5, 6),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			pod := newResizeTestPod("pod")
			pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: resource.MustParse(tc.allocatedC1)}
			pod.Spec.Containers[1].Resources.Requests = v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}
			m := newResizeTestManager(logger, &resizingPolicy{}, pod)
			m.state.SetPendingResize(string(pod.UID), pending)
			if tc.before != nil {
				tc.before(m.state)
			}

			// The pod is added again after the restart, with its allocated resources.
			m.GetTopologyHints(logger, pod, &pod.Spec.Containers[0], lifecycle.AddOperation)
			require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))

			c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
			require.True(t, tc.expectC1.Equals(c1), "expected c1 %s, got %s", tc.expectC1, c1)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			_, staged := m.state.GetPendingResize(string(pod.UID))
			require.False(t, staged, "adding the pod must resolve its pending resize")
		})
	}
}

func TestRemoveStaleStateRemovesPendingResize(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	m := newResizeTestManager(logger, &resizingPolicy{}, pod)
	m.state.SetPendingResize(string(pod.UID), state.PendingResize{Containers: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}})
	m.state.SetPendingResize("removed-pod", state.PendingResize{Containers: map[string]cpuset.CPUSet{"c": cpuset.New(5)}})

	m.removeStaleState(logger)

	_, staged := m.state.GetPendingResize(string(pod.UID))
	require.True(t, staged, "the pending resize of an active pod must be kept")
	_, staged = m.state.GetPendingResize("removed-pod")
	require.False(t, staged, "the pending resize of a removed pod must be removed")
}

// This test is artificial as pod-level resources do not scale exclusive CPUs.
func TestResolvePendingResizePodLevel(t *testing.T) {
	// The resize staged before the restart grows the pod from {1,2,3} to {1,2,3,4}.
	target := cpuset.New(1, 2, 3, 4)
	pending := state.PendingResize{Pod: &target}
	tests := []struct {
		name string
		// allocated is the pod-level CPU request in the persisted pod allocation.
		allocated string
		// podCPUs is the restored pod-level assignment.
		podCPUs       cpuset.CPUSet
		expectPod     cpuset.CPUSet
		expectDefault cpuset.CPUSet
	}{
		{
			name:          "the allocation includes the resize",
			allocated:     "4",
			podCPUs:       cpuset.New(1, 2, 3),
			expectPod:     cpuset.New(1, 2, 3, 4),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:          "the allocation does not include the resize",
			allocated:     "3",
			podCPUs:       cpuset.New(1, 2, 3),
			expectPod:     cpuset.New(1, 2, 3),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name:          "the CPUs taken are computed from the restored assignment",
			allocated:     "4",
			podCPUs:       cpuset.New(1, 2),
			expectPod:     cpuset.New(1, 2, 3, 4),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:          "a non-integral allocation does not include the resize",
			allocated:     "3500m",
			podCPUs:       cpuset.New(1, 2, 3),
			expectPod:     cpuset.New(1, 2, 3),
			expectDefault: cpuset.New(4, 5, 6),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			pod := newResizeTestPod("pod")
			pod.Spec.Resources = &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(tc.allocated)}}
			m := newResizeTestManager(logger, &resizingPolicy{}, pod)
			m.state.SetPodCPUSet(string(pod.UID), tc.podCPUs)
			m.state.SetPendingResize(string(pod.UID), pending)

			m.GetPodTopologyHints(logger, pod, lifecycle.AddOperation)
			require.NoError(t, m.AllocatePod(logger, pod, lifecycle.AddOperation))

			podSet, _ := m.state.GetPodCPUSet(string(pod.UID))
			require.True(t, tc.expectPod.Equals(podSet), "expected pod %s, got %s", tc.expectPod, podSet)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			_, staged := m.state.GetPendingResize(string(pod.UID))
			require.False(t, staged, "adding the pod must resolve its pending resize")
		})
	}
}

func TestCommitAdmitOfResizeNotStaged(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	m := newResizeTestManager(logger, &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}, pod)
	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.ResizeOperation))

	require.Error(t, m.CommitAdmit(ctx, pod, lifecycle.ResizeOperation))
	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2)), "a resize that was not staged must not be committed, c1 is %s", c1)
}

func TestCommitAdmitWithCPUTakenAfterStage(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	m := newResizeTestManager(logger, &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}, pod)
	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.ResizeOperation))
	require.NoError(t, m.StageAdmit(ctx, pod, lifecycle.ResizeOperation))
	m.state.SetCPUSet(otherPodUID, "c", cpuset.New(4, 7))
	m.state.SetDefaultCPUSet(cpuset.New(5, 6))

	require.Error(t, m.CommitAdmit(ctx, pod, lifecycle.ResizeOperation))
	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2)), "the resize must not be committed, c1 is %s", c1)
	_, staged := m.state.GetPendingResize(string(pod.UID))
	require.False(t, staged, "a pending resize that cannot be committed must be deleted")
}
