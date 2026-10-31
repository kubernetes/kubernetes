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
	"errors"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"

	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
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
	// hintStates records the state given to each GetTopologyHints and
	// GetPodTopologyHints call.
	hintStates []state.Reader
	// verified records the pending resizes given to VerifyPendingResize, and
	// verifiedWith the allocation reader of the last call.
	verified     []state.PendingResize
	verifiedWith AllocationReader
}

func (p *resizingPolicy) VerifyPendingResize(pending state.PendingResize, allocationReader AllocationReader) error {
	p.verified = append(p.verified, pending)
	p.verifiedWith = allocationReader
	return p.verifyPendingResizeErr
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
	p.hintStates = append(p.hintStates, s)
	return nil
}

func (p *resizingPolicy) GetPodTopologyHints(_ klog.Logger, s state.State, _ *v1.Pod, _ lifecycle.Operation) map[string][]topologymanager.TopologyHint {
	p.hintStates = append(p.hintStates, s)
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
		between       func(s state.State)
		result        lifecycle.AdmitResult
		expectC1      cpuset.CPUSet
		expectC2      cpuset.CPUSet
		expectDefault cpuset.CPUSet
	}{
		{
			name:          "scale up is committed",
			resized:       map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
			result:        lifecycle.AdmitSuccess,
			expectC1:      cpuset.New(1, 2, 4),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:          "scale down is committed",
			resized:       map[string]cpuset.CPUSet{"c1": cpuset.New(1)},
			result:        lifecycle.AdmitSuccess,
			expectC1:      cpuset.New(1),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(2, 4, 5, 6),
		},
		{
			name:          "a failure leaves the state untouched",
			resized:       map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
			result:        lifecycle.AdmitFailure,
			expectC1:      cpuset.New(1, 2),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name:    "the commit keeps the CPUs released meanwhile",
			resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
			between: func(s state.State) {
				s.Delete(otherPodUID, "c")
				s.SetDefaultCPUSet(s.GetDefaultCPUSet().Union(cpuset.New(7)))
			},
			result:        lifecycle.AdmitSuccess,
			expectC1:      cpuset.New(1, 2, 4),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(5, 6, 7),
		},
		{
			name:    "a CPU taken meanwhile leaves the assignment unchanged",
			resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
			between: func(s state.State) {
				s.SetCPUSet(otherPodUID, "c", cpuset.New(4, 7))
				s.SetDefaultCPUSet(s.GetDefaultCPUSet().Difference(cpuset.New(4)))
			},
			result:        lifecycle.AdmitSuccess,
			expectC1:      cpuset.New(1, 2),
			expectC2:      cpuset.New(3),
			expectDefault: cpuset.New(5, 6),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			pod := newResizeTestPod("pod")
			policy := &resizingPolicy{resized: tc.resized}
			m := newResizeTestManager(logger, policy, pod)

			for container := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
				m.GetTopologyHints(logger, pod, container, lifecycle.ResizeOperation)
				require.NoError(t, m.Allocate(ctx, pod, container, lifecycle.ResizeOperation))
			}
			c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
			require.True(t, c1.Equals(cpuset.New(1, 2)), "the resize must not change the state before it is committed, c1 is %s", c1)
			require.Len(t, policy.hintsSaw, len(pod.Spec.Containers), "the hints of every container must reach the policy")
			require.True(t, policy.hintsSaw[1].Equals(assignedPodCPUs(m.pendingResize.scratch, string(pod.UID))),
				"the hints of c2 must see what was prepared for c1, saw %s", policy.hintsSaw[1])
			for _, s := range policy.hintStates {
				require.Same(t, m.pendingResize.scratch, s, "the hints of a resize must be computed on the scratch state")
			}
			pending, recorded := m.state.GetPendingResize()
			require.True(t, recorded, "the prepared resize must be recorded as a pending resize")
			require.Equal(t, state.PendingResize{PodUID: string(pod.UID), ContainerCPUSets: tc.resized}, pending)

			if tc.between != nil {
				tc.between(m.state)
			}
			m.CompleteAdmit(ctx, pod, lifecycle.ResizeOperation, tc.result)

			c1, _ = m.state.GetCPUSet(string(pod.UID), "c1")
			require.True(t, tc.expectC1.Equals(c1), "expected c1 %s, got %s", tc.expectC1, c1)
			c2, _ := m.state.GetCPUSet(string(pod.UID), "c2")
			require.True(t, tc.expectC2.Equals(c2), "expected c2 %s, got %s", tc.expectC2, c2)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			require.Nil(t, m.pendingResize, "CompleteAdmit must end the transaction")
			_, recorded = m.state.GetPendingResize()
			require.False(t, recorded, "CompleteAdmit must delete the pending resize")
		})
	}
}

func TestResizeAdmitTransactionPodLevel(t *testing.T) {
	tests := []struct {
		name          string
		resizedPod    cpuset.CPUSet
		result        lifecycle.AdmitResult
		expectPod     cpuset.CPUSet
		expectDefault cpuset.CPUSet
	}{
		{
			name:          "scale up is committed",
			resizedPod:    cpuset.New(1, 2, 3, 4, 5),
			result:        lifecycle.AdmitSuccess,
			expectPod:     cpuset.New(1, 2, 3, 4, 5),
			expectDefault: cpuset.New(6),
		},
		{
			name:          "scale down is committed",
			resizedPod:    cpuset.New(1, 2, 3),
			result:        lifecycle.AdmitSuccess,
			expectPod:     cpuset.New(1, 2, 3),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name:          "a failed scale up leaves the state untouched",
			resizedPod:    cpuset.New(1, 2, 3, 4, 5),
			result:        lifecycle.AdmitFailure,
			expectPod:     cpuset.New(1, 2, 3, 4),
			expectDefault: cpuset.New(5, 6),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			pod := newResizeTestPod("pod")
			policy := &resizingPolicy{resizedPod: tc.resizedPod}
			m := newResizeTestManager(logger, policy, pod)
			// The pod-level assignment holds the CPUs of its containers and CPU 4,
			// which none of them uses.
			m.state.SetPodCPUSet(string(pod.UID), cpuset.New(1, 2, 3, 4))
			m.state.SetDefaultCPUSet(cpuset.New(5, 6))

			m.GetPodTopologyHints(logger, pod, lifecycle.ResizeOperation)
			require.NoError(t, m.AllocatePod(logger, pod, lifecycle.ResizeOperation))
			podSet, _ := m.state.GetPodCPUSet(string(pod.UID))
			require.True(t, podSet.Equals(cpuset.New(1, 2, 3, 4)), "the resize must not change the state before it is committed, the pod has %s", podSet)
			require.Len(t, policy.hintStates, 1, "GetPodTopologyHints must reach the policy")
			require.Same(t, m.pendingResize.scratch, policy.hintStates[0], "the pod hints of a resize must be computed on the scratch state")
			pending, recorded := m.state.GetPendingResize()
			require.True(t, recorded, "the prepared resize must be recorded as a pending resize")
			require.Equal(t, state.PendingResize{PodUID: string(pod.UID), PodCPUSet: &tc.resizedPod}, pending)

			m.CompleteAdmit(ctx, pod, lifecycle.ResizeOperation, tc.result)

			podSet, _ = m.state.GetPodCPUSet(string(pod.UID))
			require.True(t, tc.expectPod.Equals(podSet), "expected pod %s, got %s", tc.expectPod, podSet)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			require.Nil(t, m.pendingResize, "CompleteAdmit must end the transaction")
			_, recorded = m.state.GetPendingResize()
			require.False(t, recorded, "CompleteAdmit must delete the pending resize")
		})
	}
}

// TestResizeAdmitTransactionOfAnotherPod is a defensive test of conditions that
// should never happen: the allocation manager admits one pod at a time and
// completes every admission before it starts the next one. If a transaction is
// left uncompleted anyway, the admission of another pod discards it, and a late
// CompleteAdmit for the discarded pod neither ends the new transaction nor
// applies the discarded resize.
func TestResizeAdmitTransactionOfAnotherPod(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	another := newResizeTestPod("another")
	m := newResizeTestManager(logger, &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}, pod, another)

	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.ResizeOperation))
	// The transaction of pod is never completed, so the one of another replaces it.
	require.NoError(t, m.Allocate(ctx, another, &another.Spec.Containers[1], lifecycle.ResizeOperation))
	require.Equal(t, string(another.UID), m.pendingResize.podUID)
	// The resize of another changes nothing, so only the pending resize of the
	// discarded transaction could be left.
	pending, recorded := m.state.GetPendingResize()
	require.False(t, recorded, "the pending resize of the discarded transaction must be deleted, got %+v", pending)

	m.CompleteAdmit(ctx, pod, lifecycle.ResizeOperation, lifecycle.AdmitSuccess)
	require.NotNil(t, m.pendingResize, "committing the admission of another pod must not end the transaction")
	m.CompleteAdmit(ctx, pod, lifecycle.ResizeOperation, lifecycle.AdmitFailure)
	require.NotNil(t, m.pendingResize, "failing the admission of another pod must not end the transaction")

	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2)), "the discarded resize must not be committed, c1 is %s", c1)
}

func TestAddAdmitWithoutTransaction(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	policy := &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}
	m := newResizeTestManager(logger, policy, pod)

	m.GetTopologyHints(logger, pod, &pod.Spec.Containers[0], lifecycle.AddOperation)
	m.GetPodTopologyHints(logger, pod, lifecycle.AddOperation)
	require.Len(t, policy.hintStates, 2, "the hints must reach the policy")
	for _, s := range policy.hintStates {
		require.Same(t, m.state, s, "the hints of an added pod must be computed on the state")
	}
	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
	require.Nil(t, m.pendingResize)
	_, recorded := m.state.GetPendingResize()
	require.False(t, recorded, "an added pod must not record a pending resize")
	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2, 4)), "the allocation must be applied to the state, c1 is %s", c1)

	m.CompleteAdmit(ctx, pod, lifecycle.AddOperation, lifecycle.AdmitSuccess)
	m.CompleteAdmit(ctx, pod, lifecycle.AddOperation, lifecycle.AdmitFailure)
	c1, _ = m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2, 4)), "the transaction must not change the allocation, c1 is %s", c1)
}

// TestResizeWithStaticPolicyChangesNothing checks that this transaction alone
// does not change the behavior: the static policy handles only the add
// operation and skips a resize, so admitting a resize leaves the state exactly
// as before. The exclusive CPU resize of KEP-5554 will have to adapt this test.
func TestResizeWithStaticPolicyChangesNothing(t *testing.T) {
	if runtime.GOOS == "windows" {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.WindowsCPUAndMemoryAffinity, true)
	}

	for _, result := range []lifecycle.AdmitResult{lifecycle.AdmitSuccess, lifecycle.AdmitFailure} {
		t.Run(string(result), func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(), topologymanager.NewFakeManager(logger), nil)
			require.NoError(t, err)
			pod := makePod("pod", "c1", "2", "2")
			m := &manager{
				policy:          policy,
				state:           state.NewMemoryState(logger),
				lastUpdateState: state.NewMemoryState(logger),
				containerMap:    containermap.NewContainerMap(),
				sourcesReady:    &sourcesReadyStub{},
				activePods:      func() []*v1.Pod { return []*v1.Pod{pod} },
			}
			require.NoError(t, policy.Start(logger, m.state))
			require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
			c1, _ := m.state.GetCPUSet("pod", "c1")
			require.Equal(t, 2, c1.Size(), "the pod must get exclusive CPUs, c1 is %s", c1)
			before := cloneState(logger, m.state)

			resized := makePod("pod", "c1", "4", "4")
			m.GetTopologyHints(logger, resized, &resized.Spec.Containers[0], lifecycle.ResizeOperation)
			require.NoError(t, m.Allocate(ctx, resized, &resized.Spec.Containers[0], lifecycle.ResizeOperation))
			_, recorded := m.state.GetPendingResize()
			require.False(t, recorded, "a resize that changes nothing must not record a pending resize")
			m.CompleteAdmit(ctx, resized, lifecycle.ResizeOperation, result)

			require.Equal(t, before.GetCPUAssignments(), m.state.GetCPUAssignments(), "assignments")
			require.True(t, before.GetDefaultCPUSet().Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", before.GetDefaultCPUSet(), m.state.GetDefaultCPUSet())
			require.Nil(t, m.pendingResize, "CompleteAdmit must end the transaction")
		})
	}
}

func TestRecordPendingResize(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	m := newResizeTestManager(logger, &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4), "c2": cpuset.New(3)}}, pod)

	// c2 keeps its CPUs, so only c1 is recorded, and the entry is recorded
	// after the first allocation already.
	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.ResizeOperation))
	expected := state.PendingResize{PodUID: string(pod.UID), ContainerCPUSets: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}
	pending, recorded := m.state.GetPendingResize()
	require.True(t, recorded, "the resize must be recorded after the first allocation")
	require.Equal(t, expected, pending)

	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[1], lifecycle.ResizeOperation))
	pending, recorded = m.state.GetPendingResize()
	require.True(t, recorded)
	require.Equal(t, expected, pending, "a container that keeps its CPUs must not be recorded")

	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2)), "recording the resize must not apply it, c1 is %s", c1)
	require.True(t, m.state.GetDefaultCPUSet().Equals(cpuset.New(4, 5, 6)), "recording the resize must not change the default CPU set, got %s", m.state.GetDefaultCPUSet())
}

func TestRecordPendingResizeWithoutChanges(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	policy := &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}
	m := newResizeTestManager(logger, policy, pod)

	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.ResizeOperation))
	_, recorded := m.state.GetPendingResize()
	require.True(t, recorded)

	// The next allocation of the same admission gives c1 its CPUs back, so the
	// resize changes nothing and its pending resize is deleted.
	policy.resized["c1"] = cpuset.New(1, 2)
	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.ResizeOperation))
	_, recorded = m.state.GetPendingResize()
	require.False(t, recorded, "a resize that changes nothing must not leave a pending resize")

	m.CompleteAdmit(ctx, pod, lifecycle.ResizeOperation, lifecycle.AdmitSuccess)
	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2)), "expected c1 to keep its CPUs, got %s", c1)
}

func TestCompleteAdmitOfResizeNotRecorded(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := newResizeTestPod("pod")
	m := newResizeTestManager(logger, &resizingPolicy{resized: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)}}, pod)
	require.NoError(t, m.Allocate(ctx, pod, &pod.Spec.Containers[0], lifecycle.ResizeOperation))
	// Only a bug could lose the pending resize of a transaction in progress.
	m.state.DeletePendingResize()

	m.CompleteAdmit(ctx, pod, lifecycle.ResizeOperation, lifecycle.AdmitSuccess)
	c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
	require.True(t, c1.Equals(cpuset.New(1, 2)), "a resize that was not recorded must not be committed, c1 is %s", c1)
	require.True(t, m.state.GetDefaultCPUSet().Equals(cpuset.New(4, 5, 6)), "expected default 4-6, got %s", m.state.GetDefaultCPUSet())
	require.Nil(t, m.pendingResize, "CompleteAdmit must end the transaction")
}

// cpuRequest returns resources requesting the given quantity of CPU.
func cpuRequest(quantity string) v1.ResourceRequirements {
	return v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(quantity)}}
}

func TestResolvePendingResize(t *testing.T) {
	// The resize recorded before the restart grows c1 from {1,2} to {1,2,4}.
	pending := state.PendingResize{
		PodUID:           "pod",
		ContainerCPUSets: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 4)},
	}

	tests := []struct {
		name string
		// verifyErr is the verdict of the policy on the pending resize.
		verifyErr error
		// before changes the restored state before the pending resize is resolved.
		before        func(s state.State)
		expectC1      cpuset.CPUSet
		expectDefault cpuset.CPUSet
	}{
		{
			name:          "the policy finds the resize in the allocation",
			expectC1:      cpuset.New(1, 2, 4),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:          "the policy does not find the resize in the allocation",
			verifyErr:     errors.New("not allocated"),
			expectC1:      cpuset.New(1, 2),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name: "the CPUs taken are computed from the restored assignment",
			before: func(s state.State) {
				s.SetCPUSet("pod", "c1", cpuset.New(1))
				s.SetDefaultCPUSet(cpuset.New(2, 4, 5, 6))
			},
			expectC1:      cpuset.New(1, 2, 4),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name: "a CPU the resize takes is no longer free",
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
			logger, _ := ktesting.NewTestContext(t)
			pod := newResizeTestPod("pod")
			policy := &resizingPolicy{mockPolicy: mockPolicy{verifyPendingResizeErr: tc.verifyErr}}
			m := newResizeTestManager(logger, policy, pod)
			reader := &mockAllocationReader{}
			m.allocationReader = reader
			m.state.SetPendingResize(pending)
			if tc.before != nil {
				tc.before(m.state)
			}

			m.resolvePendingResize(logger)

			require.Equal(t, []state.PendingResize{pending}, policy.verified, "the policy must verify the pending resize")
			require.Same(t, reader, policy.verifiedWith, "the policy must verify against the allocation reader")
			c1, _ := m.state.GetCPUSet(string(pod.UID), "c1")
			require.True(t, tc.expectC1.Equals(c1), "expected c1 %s, got %s", tc.expectC1, c1)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			_, recorded := m.state.GetPendingResize()
			require.False(t, recorded, "the pending resize must be resolved")
		})
	}
}

// This test is artificial as pod-level resources do not scale exclusive CPUs.
func TestResolvePendingResizePodLevel(t *testing.T) {
	// The resize recorded before the restart grows the pod from {1,2,3} to {1,2,3,4}.
	pending := state.PendingResize{PodUID: "pod", PodCPUSet: new(cpuset.New(1, 2, 3, 4))}
	tests := []struct {
		name string
		// verifyErr is the verdict of the policy on the pending resize.
		verifyErr error
		// podCPUs is the restored pod-level assignment.
		podCPUs cpuset.CPUSet
		// taken makes another pod hold CPU 4, which the resize takes.
		taken         bool
		expectPod     cpuset.CPUSet
		expectDefault cpuset.CPUSet
	}{
		{
			name:          "the policy finds the resize in the allocation",
			podCPUs:       cpuset.New(1, 2, 3),
			expectPod:     cpuset.New(1, 2, 3, 4),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:          "the policy does not find the resize in the allocation",
			verifyErr:     errors.New("not allocated"),
			podCPUs:       cpuset.New(1, 2, 3),
			expectPod:     cpuset.New(1, 2, 3),
			expectDefault: cpuset.New(4, 5, 6),
		},
		{
			name:          "the CPUs taken are computed from the restored assignment",
			podCPUs:       cpuset.New(1, 2),
			expectPod:     cpuset.New(1, 2, 3, 4),
			expectDefault: cpuset.New(5, 6),
		},
		{
			name:          "a CPU the resize takes is no longer free",
			podCPUs:       cpuset.New(1, 2, 3),
			taken:         true,
			expectPod:     cpuset.New(1, 2, 3),
			expectDefault: cpuset.New(5, 6),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			pod := newResizeTestPod("pod")
			m := newResizeTestManager(logger, &resizingPolicy{mockPolicy: mockPolicy{verifyPendingResizeErr: tc.verifyErr}}, pod)
			m.allocationReader = mockAllocationReader{}
			m.state.SetPodCPUSet(string(pod.UID), tc.podCPUs)
			if tc.taken {
				m.state.SetCPUSet(otherPodUID, "c", cpuset.New(4, 7))
				m.state.SetDefaultCPUSet(cpuset.New(5, 6))
			}
			m.state.SetPendingResize(pending)

			m.resolvePendingResize(logger)

			podSet, _ := m.state.GetPodCPUSet(string(pod.UID))
			require.True(t, tc.expectPod.Equals(podSet), "expected pod %s, got %s", tc.expectPod, podSet)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			_, recorded := m.state.GetPendingResize()
			require.False(t, recorded, "the pending resize must be resolved")
		})
	}
}

// TestStartResolvesPendingResize checks that Start resolves a pending resize
// left by a kubelet restart against the persisted allocation, before the
// policy validates the state.
func TestStartResolvesPendingResize(t *testing.T) {
	if runtime.GOOS == "windows" {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.WindowsCPUAndMemoryAffinity, true)
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)

	tests := []struct {
		name      string
		allocated string
		// taken makes another pod hold CPU 2, which the resize takes.
		taken         bool
		expectC1      cpuset.CPUSet
		expectDefault cpuset.CPUSet
	}{
		{
			name:          "the allocation includes the resize",
			allocated:     "2",
			expectC1:      cpuset.New(1, 2),
			expectDefault: cpuset.New(0, 3),
		},
		{
			name:          "the allocation does not include the resize",
			allocated:     "1",
			expectC1:      cpuset.New(1),
			expectDefault: cpuset.New(0, 2, 3),
		},
		{
			name:          "a CPU the resize takes is no longer free",
			allocated:     "2",
			taken:         true,
			expectC1:      cpuset.New(1),
			expectDefault: cpuset.New(0, 3),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sDir := t.TempDir()
			logger, tCtx := ktesting.NewTestContext(t)

			// The checkpoint left by the restart: c1 has {1}, and the resize
			// recorded before the restart grows it to {1,2}.
			s, err := state.NewCheckpointState(logger, sDir, cpuManagerStateFileName, "static", containermap.NewContainerMap())
			require.NoError(t, err)
			s.SetCPUSet("pod", "c1", cpuset.New(1))
			s.SetDefaultCPUSet(cpuset.New(0, 2, 3))
			if tc.taken {
				s.SetCPUSet(otherPodUID, "c", cpuset.New(2))
				s.SetDefaultCPUSet(cpuset.New(0, 3))
			}
			s.SetPendingResize(state.PendingResize{PodUID: "pod", ContainerCPUSets: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2)}})

			mgr, err := newRestoreTestManager(logger, 4, sDir)
			require.NoError(t, err)
			reader := mockAllocationReader{containers: map[types.UID]map[string]v1.ResourceRequirements{"pod": {"c1": cpuRequest(tc.allocated)}}}
			// The pods are active, so that the reconcile loop started by Start keeps their CPUs.
			pod := newResizeTestPod("pod")
			other := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: otherPodUID, Name: otherPodUID, Namespace: "ns"},
				Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "c"}}},
			}
			err = mgr.Start(tCtx, func() []*v1.Pod { return []*v1.Pod{pod, other} }, &sourcesReadyStub{}, mockPodStatusProvider{}, reader, mockRuntimeService{}, containermap.NewContainerMap())
			require.NoError(t, err)

			m := mgr.(*manager)
			c1, _ := m.state.GetCPUSet("pod", "c1")
			require.True(t, tc.expectC1.Equals(c1), "expected c1 %s, got %s", tc.expectC1, c1)
			require.True(t, tc.expectDefault.Equals(m.state.GetDefaultCPUSet()), "expected default %s, got %s", tc.expectDefault, m.state.GetDefaultCPUSet())
			if tc.taken {
				otherCPUs, _ := m.state.GetCPUSet(otherPodUID, "c")
				require.True(t, otherCPUs.Equals(cpuset.New(2)), "the other pod must keep its CPUs, got %s", otherCPUs)
			}
			_, recorded := m.state.GetPendingResize()
			require.False(t, recorded, "Start must resolve the pending resize")
		})
	}
}
