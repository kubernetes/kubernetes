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
	p.hintsSaw = append(p.hintsSaw, podCPUs(s, string(pod.UID)))
	return nil
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
			require.True(t, policy.hintsSaw[1].Equals(podCPUs(m.pendingResize.scratch, string(pod.UID))),
				"the hints of c2 must see what was prepared for c1, saw %s", policy.hintsSaw[1])

			if tc.between != nil {
				tc.between(m.state)
			}
			if tc.abort {
				m.AbortAdmit(ctx, pod, lifecycle.ResizeOperation)
			} else {
				err := m.StageAdmit(ctx, pod, lifecycle.ResizeOperation)
				require.Equal(t, tc.expectStageErr, err != nil, "stage error: %v", err)
			}
			err := m.CommitAdmit(ctx, pod, lifecycle.ResizeOperation)
			require.Equal(t, tc.expectCommitErr, err != nil, "commit error: %v", err)

			c1, _ = m.state.GetCPUSet(string(pod.UID), "c1")
			require.True(t, tc.expectC1.Equals(c1), "expected c1 %s, got %s", tc.expectC1, c1)
			c2, _ := m.state.GetCPUSet(string(pod.UID), "c2")
			require.True(t, tc.expectC2.Equals(c2), "expected c2 %s, got %s", tc.expectC2, c2)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			require.Nil(t, m.pendingResize, "the transaction must end with the commit or the abort")
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

			if tc.abort {
				m.AbortAdmit(ctx, pod, lifecycle.ResizeOperation)
			} else {
				require.NoError(t, m.StageAdmit(ctx, pod, lifecycle.ResizeOperation))
			}
			require.NoError(t, m.CommitAdmit(ctx, pod, lifecycle.ResizeOperation))

			podSet, _ = m.state.GetPodCPUSet(string(pod.UID))
			require.True(t, tc.expectPod.Equals(podSet), "expected pod %s, got %s", tc.expectPod, podSet)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			require.Nil(t, m.pendingResize, "the transaction must end with the commit or the abort")
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
