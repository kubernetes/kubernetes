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

package manager

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/utils/ptr"
)

func newTestPodGroupInfo(pg *schedulingv1beta1.PodGroup, cpg *schedulingv1alpha3.CompositePodGroup, pods []*v1.Pod) fwk.PodGroupInfo {
	if cpg != nil {
		return &framework.PodGroupInfo{
			GenericPodGroup: fwk.NewGenericCompositePodGroup(cpg),
			UnscheduledPods: pods,
			Children: []*framework.PodGroupInfo{
				{
					GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{}),
					UnscheduledPods: pods,
				},
			},
		}
	}
	return &framework.PodGroupInfo{
		GenericPodGroup: fwk.NewGenericPodGroup(pg),
		UnscheduledPods: pods,
	}
}

func TestNewDefaultPreemptionManager(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	client := clientsetfake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	snapshot := internalcache.NewEmptySnapshot()

	registeredPlugins := []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
	}

	fh, err := tf.NewFramework(
		ctx,
		registeredPlugins,
		"",
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
	)
	if err != nil {
		t.Fatalf("failed to create framework handle: %v", err)
	}

	fts := feature.Features{}
	mgr := NewDefaultPreemptionManager(fh, fts)
	if mgr == nil {
		t.Fatalf("expected non-nil PreemptionManager from NewDefaultPreemptionManager")
	}
	if mgr.Executor() == nil {
		t.Fatalf("expected non-nil PreemptionExecutor from mgr.Executor()")
	}

	mgr2 := New(fh, fts)
	if mgr2 == nil {
		t.Fatalf("expected non-nil PreemptionManager from New")
	}
	if mgr2.Executor() == nil {
		t.Fatalf("expected non-nil PreemptionExecutor from mgr2.Executor()")
	}
}

func TestDefaultPreemptionManager_GenerateVictims(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Capacity(veryLargeRes).Obj()
	node2 := st.MakeNode().Name("node2").Capacity(veryLargeRes).Obj()

	lowPod := st.MakePod().Name("low-pod").Namespace("default").UID("low-pod").Node("node1").Priority(lowPriority).Obj()
	midPod := st.MakePod().Name("mid-pod").Namespace("default").UID("mid-pod").Node("node2").Priority(midPriority).Obj()
	highPod := st.MakePod().Name("high-pod").Namespace("default").UID("high-pod").Node("node2").Priority(highPriority).Obj()

	pdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pdb",
			Namespace: "default",
		},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: &intstr.IntOrString{Type: intstr.Int, IntVal: 1},
		},
		Status: policy.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 1,
		},
	}

	tests := []struct {
		name              string
		preemptorPriority int32
		preemptorPods     []*v1.Pod
		preemptorPG       *schedulingv1beta1.PodGroup
		preemptorCPG      *schedulingv1alpha3.CompositePodGroup
		expectedVictimLen int
	}{
		{
			name:              "Preemptor with high priority should consider lower priority victims",
			preemptorPriority: highPriority,
			preemptorPods:     []*v1.Pod{st.MakePod().Name("p1").Priority(highPriority).Obj()},
			preemptorPG:       st.MakePodGroup().Name("pg1").Priority(highPriority).Obj(),
			expectedVictimLen: 2, // lowPod and midPod
		},
		{
			name:              "Preemptor with mid priority should only consider low priority victims",
			preemptorPriority: midPriority,
			preemptorPods:     []*v1.Pod{st.MakePod().Name("p1").Priority(midPriority).Obj()},
			preemptorPG:       st.MakePodGroup().Name("pg1").Priority(midPriority).Obj(),
			expectedVictimLen: 1, // lowPod only
		},
		{
			name:              "Preemptor with low priority cannot preempt anything",
			preemptorPriority: lowPriority,
			preemptorPods:     []*v1.Pod{st.MakePod().Name("p1").Priority(lowPriority).Obj()},
			preemptorPG:       st.MakePodGroup().Name("pg1").Priority(lowPriority).Obj(),
			expectedVictimLen: 0,
		},
		{
			name:              "Preemptor with CompositePodGroup priority",
			preemptorPriority: highPriority,
			preemptorPods:     []*v1.Pod{st.MakePod().Name("p1").Priority(highPriority).Obj()},
			preemptorCPG:      &schedulingv1alpha3.CompositePodGroup{ObjectMeta: metav1.ObjectMeta{Name: "cpg1"}, Spec: schedulingv1alpha3.CompositePodGroupSpec{Priority: ptr.To(highPriority)}},
			expectedVictimLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			client := clientsetfake.NewSimpleClientset([]runtime.Object{pdb}...)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			_ = informerFactory.Policy().V1().PodDisruptionBudgets().Informer().GetStore().Add(pdb)

			snapshot := internalcache.NewSnapshot([]*v1.Pod{lowPod, midPod, highPod}, []*v1.Node{node1, node2})

			registeredPlugins := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}

			fh, err := tf.NewFramework(
				ctx,
				registeredPlugins,
				"",
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatalf("failed to create framework handle: %v", err)
			}

			mgr := NewDefaultPreemptionManager(fh, feature.Features{})
			pgInfo := newTestPodGroupInfo(tt.preemptorPG, tt.preemptorCPG, tt.preemptorPods)

			victims, err := mgr.GenerateVictims(ctx, pgInfo)
			if err != nil {
				t.Fatalf("unexpected error from GenerateVictims: %v", err)
			}

			if len(victims) != tt.expectedVictimLen {
				t.Errorf("expected %d victims, got %d", tt.expectedVictimLen, len(victims))
			}
		})
	}
}
