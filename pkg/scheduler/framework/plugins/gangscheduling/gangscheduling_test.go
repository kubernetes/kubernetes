/*
Copyright 2025 The Kubernetes Authors.

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

package gangscheduling

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

func init() {
	// This is required for tests where cache is initialized, and cache attempts to update metrics.
	metrics.Register()
}

func Test_isSchedulableAfterPodAdded(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		newPod       *v1.Pod
		pgs          []*schedulingapi.PodGroup
		cpgs         []*schedulingapi.CompositePodGroup
		expectedHint fwk.QueueingHint
	}{
		{
			name:         "add a newPod which matches the pod's scheduling group",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:       st.MakePod().PodGroupName("pg").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "add a newPod with NodeName set which matches the pod's scheduling group",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:       st.MakePod().PodGroupName("pg").Node("node1").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:   "add a newPod which doesn't match the pod's namespace",
			pod:    st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod: st.MakePod().Namespace("foo").PodGroupName("pg").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg").Obj(),
				st.MakePodGroup().Namespace("foo").Name("pg").Obj(),
			},
			expectedHint: fwk.QueueSkip,
		},
		{
			name:   "add a newPod which doesn't match the pod's pod group name",
			pod:    st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().PodGroupName("pg2").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Obj(),
				st.MakePodGroup().Name("pg2").Obj(),
			},
			expectedHint: fwk.QueueSkip,
		},
		{
			name:   "add a newPod which belongs to a different scheduling group but matches the root CPG",
			pod:    st.MakePod().Name("p1").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().Name("p2").PodGroupName("pg2").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root").Obj(),
				st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root").Obj(),
			},
			cpgs: []*schedulingapi.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Obj(),
			},
			expectedHint: fwk.Queue,
		},
		{
			name:   "add a newPod which belongs to a different scheduling group and does not match the root CPG",
			pod:    st.MakePod().Name("p1").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().Name("p2").PodGroupName("pg2").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root1").Obj(),
				st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root2").Obj(),
			},
			cpgs: []*schedulingapi.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-root2").Obj(),
			},
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.cpgs) > 0 {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)
			}
			logger, ctx := ktesting.NewTestContext(t)

			var objs []runtime.Object
			for _, pg := range tc.pgs {
				objs = append(objs, pg)
			}
			for _, cpg := range tc.cpgs {
				objs = append(objs, cpg)
			}
			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(objs...), 0)

			for _, pg := range tc.pgs {
				informerFactory.Scheduling().V1alpha3().PodGroups().Informer().GetStore().Add(pg)
			}
			for _, cpg := range tc.cpgs {
				informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer().GetStore().Add(cpg)
			}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			actualHint, err := p.(*GangScheduling).isSchedulableAfterPodAdded(logger, tc.pod, nil, tc.newPod)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("unexpected QueueingHint (-want, +got):\n%s", diff)
			}
		})
	}
}

func Test_isSchedulableAfterPodGroupAdded(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)
	tests := []struct {
		name         string
		pod          *v1.Pod
		newPodGroup  *schedulingapi.PodGroup
		pgs          []*schedulingapi.PodGroup
		cpgs         []*schedulingapi.CompositePodGroup
		expectedHint fwk.QueueingHint
	}{
		{
			name:         "add a pod group which matches the pod's pod group name",
			pod:          st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPodGroup:  st.MakePodGroup().Name("pg").MinCount(1).WorkloadRef("t", "w").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group name",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").MinCount(1).WorkloadRef("t", "w").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Obj(),
			},
			expectedHint: fwk.QueueSkip,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group namespace",
			pod:         st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			newPodGroup: st.MakePodGroup().Namespace("ns2").Name("pg").MinCount(1).WorkloadRef("t", "w").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Namespace("ns1").Name("pg").Obj(),
			},
			expectedHint: fwk.QueueSkip,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group but matches the root CPG",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root").MinCount(1).Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root").Obj(),
			},
			cpgs: []*schedulingapi.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Obj(),
			},
			expectedHint: fwk.Queue,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group and doesn't match the root CPG",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root2").MinCount(1).Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root1").Obj(),
			},
			cpgs: []*schedulingapi.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-root2").Obj(),
			},
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			var objs []runtime.Object
			for _, pg := range tc.pgs {
				objs = append(objs, pg)
			}
			for _, cpg := range tc.cpgs {
				objs = append(objs, cpg)
			}
			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(objs...), 0)

			for _, pg := range tc.pgs {
				informerFactory.Scheduling().V1alpha3().PodGroups().Informer().GetStore().Add(pg)
			}
			for _, cpg := range tc.cpgs {
				informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer().GetStore().Add(cpg)
			}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			actualHint, err := p.(*GangScheduling).isSchedulableAfterPodGroupAdded(logger, tc.pod, nil, tc.newPodGroup)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("Expected QueuingHint doesn't match (-want,+got):\n%s", diff)
			}
		})
	}
}

func Test_isSchedulableAfterPodGroupUpdated(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		oldPodGroup  *schedulingapi.PodGroup
		newPodGroup  *schedulingapi.PodGroup
		expectedHint fwk.QueueingHint
		expectErr    bool
	}{
		{
			name:         "minCount decreased matches target pod",
			pod:          st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "update Basic policy",
			pod:          st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").BasicPolicy().WorkloadRef("t", "w").Obj(),
			newPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").BasicPolicy().Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "minCount increased matches target pod",
			pod:          st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			newPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "minCount unchanged matches target pod",
			pod:          st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			newPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "minCount decreased but pod group name doesn't match target pod",
			pod:          st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg-other").Obj(),
			oldPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "minCount decreased but pod group namespace doesn't match target pod",
			pod:          st.MakePod().Namespace("ns-other").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		{
			name:         "pod without a scheduling group is skipped",
			pod:          st.MakePod().Namespace("ns1").Name("p").Obj(),
			oldPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:  st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			actualHint, err := p.(*GangScheduling).isSchedulableAfterPodGroupUpdated(logger, tc.pod, tc.oldPodGroup, tc.newPodGroup)
			if tc.expectErr {
				if err == nil {
					t.Errorf("Expected error but got nil")
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.expectedHint, actualHint); diff != "" {
				t.Errorf("Expected QueuingHint doesn't match (-want,+got):\n%s", diff)
			}
		})
	}
}

type podActivatorMock struct {
	activatedPods []*v1.Pod
}

func (pam *podActivatorMock) Activate(_ klog.Logger, pods map[string]*v1.Pod) {
	for _, pod := range pods {
		pam.activatedPods = append(pam.activatedPods, pod)
	}
}

type mockPodGroupState struct {
	fwk.PodGroupState
	scheduledPodsCount int
}

func (m *mockPodGroupState) ScheduledPodsCount() int { return m.scheduledPodsCount }

func (m *mockPodGroupState) GetParent() (string, bool) { return "", false }
func (m *mockPodGroupState) GetChildren() []string     { return nil }

type mockPodGroupStateLister struct {
	state *mockPodGroupState
	err   error
}

func (m *mockPodGroupStateLister) Get(groupType, namespace, podGroupName string) (fwk.PodGroupState, error) {
	return m.state, m.err
}

type mockSharedLister struct {
	fwk.SharedLister
	podGroupStateLister *mockPodGroupStateLister
}

func (m *mockSharedLister) PodGroupStates() fwk.PodGroupStateLister {
	return m.podGroupStateLister
}

func Test_isSchedulableAfterCompositePodGroupAdded(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)
	tests := []struct {
		name         string
		pod          *v1.Pod
		newCPG       *schedulingapi.CompositePodGroup
		cpgs         []*schedulingapi.CompositePodGroup
		pgs          []*schedulingapi.PodGroup
		expectedHint fwk.QueueingHint
	}{
		{
			name: "add a CPG which matches the pod's root CPG",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-root").Obj(),
			},
			newCPG:       st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "add a CPG which matches the pod's intermediate CPG but implies the same root",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-sub").Obj(),
			},
			cpgs: []*schedulingapi.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			},
			newCPG:       st.MakeCompositePodGroup().Namespace("default").Name("cpg-sub").ParentCompositePodGroup("cpg-root").Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "add a CPG which does not match the pod's CPG hierarchy",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-1").Obj(),
			},
			cpgs: []*schedulingapi.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-1").Obj(),
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-2").Obj(),
			},
			newCPG:       st.MakeCompositePodGroup().Namespace("default").Name("cpg-2").Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			// Must create clientset with objects
			var objs []runtime.Object
			for _, pg := range tc.pgs {
				objs = append(objs, pg)
			}
			for _, cpg := range tc.cpgs {
				objs = append(objs, cpg)
			}

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(objs...), 0)

			// We need to wait for informers to sync
			for _, pg := range tc.pgs {
				informerFactory.Scheduling().V1alpha3().PodGroups().Informer().GetStore().Add(pg)
			}
			for _, cpg := range tc.cpgs {
				informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer().GetStore().Add(cpg)
			}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			pl := p.(*GangScheduling)

			hint, err := pl.isSchedulableAfterCompositePodGroupAdded(logger, tc.pod, nil, tc.newCPG)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if hint != tc.expectedHint {
				t.Errorf("expected hint %v, got %v", tc.expectedHint, hint)
			}
		})
	}
}

func Test_isSchedulableAfterCompositePodGroupUpdated(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)
	tests := []struct {
		name         string
		pod          *v1.Pod
		oldCPG       *schedulingapi.CompositePodGroup
		newCPG       *schedulingapi.CompositePodGroup
		cpgs         []*schedulingapi.CompositePodGroup
		pgs          []*schedulingapi.PodGroup
		expectedHint fwk.QueueingHint
	}{
		{
			name: "update a CPG to decrease minGroupCount",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-root").Obj(),
			},
			cpgs: []*schedulingapi.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			},
			oldCPG:       st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").MinGroupCount(2).Obj(),
			newCPG:       st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").MinGroupCount(1).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name: "update a CPG where minGroupCount does not decrease",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingapi.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-root").Obj(),
			},
			cpgs: []*schedulingapi.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			},
			oldCPG:       st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").MinGroupCount(2).Obj(),
			newCPG:       st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").MinGroupCount(2).Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)

			// Must create clientset with objects
			var objs []runtime.Object
			for _, pg := range tc.pgs {
				objs = append(objs, pg)
			}
			for _, cpg := range tc.cpgs {
				objs = append(objs, cpg)
			}

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(objs...), 0)

			// We need to wait for informers to sync
			for _, pg := range tc.pgs {
				informerFactory.Scheduling().V1alpha3().PodGroups().Informer().GetStore().Add(pg)
			}
			for _, cpg := range tc.cpgs {
				informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer().GetStore().Add(cpg)
			}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			pl := p.(*GangScheduling)

			hint, err := pl.isSchedulableAfterCompositePodGroupUpdated(logger, tc.pod, tc.oldCPG, tc.newCPG)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if hint != tc.expectedHint {
				t.Errorf("expected hint %v, got %v", tc.expectedHint, hint)
			}
		})
	}
}

func TestGangSchedulingFlow(t *testing.T) {
	gangPodGroup1 := st.MakePodGroup().Namespace("ns1").Name("pg1").WorkloadRef("t1", "gang-wl").MinCount(3).Obj()
	gangPodGroup2 := st.MakePodGroup().Namespace("ns1").Name("pg2").WorkloadRef("t2", "gang-wl").MinCount(4).Obj()
	basicPodGroup := st.MakePodGroup().Namespace("ns1").Name("pg3").WorkloadRef("1", "basic-wl").BasicPolicy().Obj()

	p1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj()
	p2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj()
	p3 := st.MakePod().Namespace("ns1").Name("p3").UID("p3").PodGroupName("pg1").Obj()

	p4 := st.MakePod().Namespace("ns1").Name("p4").UID("p4").PodGroupName("pg2").Obj()
	p5 := st.MakePod().Namespace("ns1").Name("p5").UID("p5").PodGroupName("pg2").Obj()

	basicPolicyPod := st.MakePod().Namespace("ns1").Name("basic-pod").UID("basic-pod").PodGroupName("pg3").Obj()

	nonGangPod := st.MakePod().Namespace("ns1").Name("non-gang").UID("non-gang").Obj()

	tests := []struct {
		name                            string
		pod                             *v1.Pod
		initialPods                     []*v1.Pod
		initialPodGroups                []*schedulingapi.PodGroup
		podsWaitingOnPermit             []*v1.Pod
		isDuringPodGroupSchedulingCycle bool
		wantPreEnqueueStatus            *fwk.Status
		wantPermitStatus                *fwk.Status
		wantActivatedPods               []*v1.Pod
		wantAllowedPods                 []types.UID
	}{
		{
			name:                 "non-gang pod succeeds immediately",
			pod:                  nonGangPod,
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroup},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
		},
		{
			name:                 "basic policy pod succeeds immediately",
			pod:                  basicPolicyPod,
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroup},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
		},
		{
			name:                 "gang pod fails PreEnqueue when pod group is not yet created",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:     []*schedulingapi.PodGroup{},
			wantPreEnqueueStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for pods's pod group \"pg1\" to appear in scheduling queue"),
		},
		{
			name:                 "gang pod fails PreEnqueue when quorum is not met",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p4, p5}, // Only p1 and p2 exist from their gang, minCount is 3.
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2},
			wantPreEnqueueStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue"),
		},
		{
			name:                 "gang pod passes PreEnqueue, but waits at Permit",
			pod:                  p1,
			initialPods:          []*v1.Pod{p2, p3, p4, p5}, // All pods are available.
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2},
			podsWaitingOnPermit:  []*v1.Pod{p2, p4, p5},
			wantPreEnqueueStatus: nil,
			wantActivatedPods:    []*v1.Pod{p3},
			// At Permit, p1 will be assumed, but the count (2) is less than the quorum (3), so it must wait.
			wantPermitStatus: fwk.NewStatus(fwk.Wait, "waiting for minCount pods from a gang to be scheduled"),
		},
		{
			name:                 "final gang pod arrives at Permit and allows all waiting pods from a gang",
			pod:                  p1, // p3 is the pod being scheduled in this cycle.
			initialPods:          []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:     []*schedulingapi.PodGroup{gangPodGroup1, gangPodGroup2},
			podsWaitingOnPermit:  []*v1.Pod{p2, p3, p4, p5},
			wantPreEnqueueStatus: nil,
			wantPermitStatus:     nil,
			wantAllowedPods:      []types.UID{"p1", "p2", "p3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			logger, ctx := ktesting.NewTestContext(t)
			cache := internalcache.New(ctx, nil, true, true)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			podGroupInformer := informerFactory.Scheduling().V1alpha3().PodGroups()
			fakeActivator := &podActivatorMock{}
			snapshot := internalcache.NewEmptySnapshot()
			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithPodGroupManager(cache),
				frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
				frameworkruntime.WithPodActivator(fakeActivator),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			// Populate informers and manager state for the test case.
			for _, wl := range tt.initialPodGroups {
				err := podGroupInformer.Informer().GetStore().Add(wl)
				if err != nil {
					t.Fatalf("Failed to add podGroup %s to store: %v", wl.Name, err)
				}
				cache.AddPodGroup(wl)
			}

			for _, p := range tt.initialPods {
				cache.AddPodGroupMember(p)
			}
			cache.AddPodGroupMember(tt.pod)

			p, err := New(ctx, nil, fh, feature.Features{EnableGenericWorkload: true})
			if err != nil {
				t.Fatalf("Failed to create plugin: %v", err)
			}
			pl := p.(*GangScheduling)

			gotPreEnqueueStatus := pl.PreEnqueue(ctx, tt.pod)
			if diff := cmp.Diff(tt.wantPreEnqueueStatus, gotPreEnqueueStatus); diff != "" {
				t.Fatalf("Unexpected PreEnqueue status (-want,+got):\n%s", diff)
			}
			if !gotPreEnqueueStatus.IsSuccess() {
				// Pod is rejected.
				return
			}

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			// Simulate that other pods have already hit Permit and are now waiting.
			for _, p := range tt.podsWaitingOnPermit {
				pod := p.DeepCopy()
				pod.Spec.NodeName = "some-node"
				if err := cache.AssumePod(logger, pod); err != nil {
					t.Fatalf("Failed to assume pod %q: %v", pod.Name, err)
				}
				status, _ := pl.Permit(ctx, schedulerframework.NewCycleState(), pod, "some-node")
				if status.Code() != fwk.Wait {
					t.Fatalf("Expected Wait status while permitting a pod %q: %v", pod.Name, status)
				}
			}

			// Clear activated pods to assert those activated in tt.pod Permit.
			fakeActivator.activatedPods = nil

			cycleState := schedulerframework.NewCycleState()
			if tt.isDuringPodGroupSchedulingCycle {
				cycleState.SetPodGroupSchedulingCycle(cycleState)
			}

			pod := tt.pod.DeepCopy()
			pod.Spec.NodeName = "some-node"

			// In a pod group scheduling cycle, a snapshot is taken after all
			// waiting pods are assumed, so that Permit can read from it.
			if tt.isDuringPodGroupSchedulingCycle {
				if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}
				podInfo, err := schedulerframework.NewPodInfo(pod)
				if err != nil {
					t.Fatalf("Failed to create pod info for %q: %v", pod.Name, err)
				}
				// Assume pod in the snapshot, as in a pod group scheduling cycle.
				if err := snapshot.AssumePod(podInfo); err != nil {
					t.Fatalf("Failed to assume pod %q in snapshot: %v", pod.Name, err)
				}
			} else {
				// Assume pod in the cache, as in a pod-by-pod scheduling cycle, where Permit reads from cache.
				if err := cache.AssumePod(logger, pod); err != nil {
					t.Fatalf("Failed to assume pod %q in cache: %v", pod.Name, err)
				}
				if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
					t.Fatalf("Failed to update snapshot: %v", err)
				}
			}

			gotPermitStatus, _ := pl.Permit(ctx, cycleState, pod, "some-node")
			if diff := cmp.Diff(tt.wantPermitStatus, gotPermitStatus); diff != "" {
				t.Fatalf("Unexpected Permit status (-want, +got):\n%s", diff)
			}
			if gotPermitStatus.Code() == fwk.Wait {
				// Pod waits for others from a gang. Simulate its eventual forget.
				if tt.isDuringPodGroupSchedulingCycle {
					if err := snapshot.ForgetPod(logger, pod); err != nil {
						t.Fatalf("Failed to forget pod %q from snapshot: %v", pod.Name, err)
					}
				} else {
					if err := cache.ForgetPod(logger, pod); err != nil {
						t.Fatalf("Failed to forget pod %q from cache: %v", pod.Name, err)
					}
				}
				return
			}

			if diff := cmp.Diff(tt.wantActivatedPods, fakeActivator.activatedPods); diff != "" {
				t.Errorf("Unexpected activated pods (-want, +got):\n%s", diff)
			}
			for _, p := range tt.wantAllowedPods {
				if wp := fh.GetWaitingPod(p); wp != nil {
					t.Errorf("Expected pod %q to be allowed", p)
				}
			}
		})
	}
}

func TestPlacementFeasible(t *testing.T) {
	tests := []struct {
		name                  string
		minCount              int32
		unscheduledPods       []*v1.Pod
		podStatuses           []fwk.Code
		expectedStatuses      []fwk.Code
		initialScheduledCount int
	}{
		{
			name:     "All pods succeed, minCount met at end",
			minCount: 2,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Success,
				fwk.Success,
			},
			expectedStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Success,
			},
		},
		{
			name:     "First pod fails, minCount not satisfiable",
			minCount: 3,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
				st.MakePod().Name("p3").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Unschedulable,
			},
			expectedStatuses: []fwk.Code{
				fwk.Unschedulable,
			},
		},
		{
			name:     "Second pod fails, minCount not satisfiable",
			minCount: 2,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Success,
				fwk.Unschedulable,
			},
			expectedStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Unschedulable,
			},
		},
		{
			name:            "Non-gang pod group ignored",
			minCount:        0, // No gang policy
			unscheduledPods: []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses: []fwk.Code{
				fwk.Unschedulable,
			},
			expectedStatuses: []fwk.Code{
				fwk.Success,
			},
		},
		{
			name:     "More than minCount pods, all succeed",
			minCount: 2,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
				st.MakePod().Name("p3").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Success,
				fwk.Success,
				fwk.Success,
			},
			expectedStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Success,
				fwk.Success,
			},
		},
		{
			name:     "More than minCount pods, first fails",
			minCount: 2,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
				st.MakePod().Name("p3").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Unschedulable,
				fwk.Success,
				fwk.Success,
			},
			expectedStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Wait,
				fwk.Success,
			},
		},
		{
			name:     "More than minCount pods, minCount not satisfiable",
			minCount: 2,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
				st.MakePod().Name("p3").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Unschedulable,
				fwk.Unschedulable,
				fwk.Success,
			},
			expectedStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Unschedulable,
				fwk.Unschedulable,
			},
		},
		{
			name:     "1 pod scheduled, 2 unscheduled pods succeed, minCount 3 met",
			minCount: 3,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Success,
				fwk.Success,
			},
			expectedStatuses: []fwk.Code{
				fwk.Wait,
				fwk.Success,
			},
			initialScheduledCount: 1,
		},
		{
			name:     "minCount already met by scheduled pods",
			minCount: 2,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Unschedulable,
			},
			expectedStatuses: []fwk.Code{
				fwk.Success,
			},
			initialScheduledCount: 2,
		},
		{
			name:     "1 pod scheduled, minCount 3, first unscheduled fails, not enough remaining",
			minCount: 3,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Unschedulable,
			},
			expectedStatuses: []fwk.Code{
				fwk.Unschedulable,
			},
			initialScheduledCount: 1,
		},
		{
			name:     "1 pod scheduled, minCount 4, first unscheduled succeeds, not enough remaining",
			minCount: 4,
			unscheduledPods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
			},
			podStatuses: []fwk.Code{
				fwk.Success,
			},
			expectedStatuses: []fwk.Code{
				fwk.Unschedulable,
			},
			initialScheduledCount: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			pgName := "test-pg"
			namespace := "default"
			pg := st.MakePodGroup().Namespace(namespace).Name(pgName).Obj()
			if tc.minCount > 0 {
				pg.Spec.SchedulingPolicy.Gang = &schedulingapi.GangSchedulingPolicy{MinCount: tc.minCount}
			} else {
				pg.Spec.SchedulingPolicy.Basic = &schedulingapi.BasicSchedulingPolicy{}
			}

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(pg), 0)
			informerFactory.Scheduling().V1alpha3().PodGroups().Informer()
			informerFactory.StartWithContext(ctx)
			informerFactory.WaitForCacheSyncWithContext(ctx)

			mockState := &mockPodGroupState{scheduledPodsCount: tc.initialScheduledCount}
			mockLister := &mockSharedLister{
				podGroupStateLister: &mockPodGroupStateLister{state: mockState},
			}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{EnableGenericWorkload: true})
			if err != nil {
				t.Fatalf("Failed to create plugin: %v", err)
			}
			pl := p.(*GangScheduling)

			// Inject the mock lister
			pl.snapshotLister = mockLister

			pgInfo := &schedulerframework.PodGroupInfo{
				Namespace:       namespace,
				Name:            pgName,
				UnscheduledPods: tc.unscheduledPods,
				PodGroup:        pg,
				Type:            schedulerframework.PodGroupKeyType,
			}

			cycleState := schedulerframework.NewCycleState()
			cycleState.SetPodGroupSchedulingCycle(cycleState)

			for i, code := range tc.podStatuses {
				if code == fwk.Success {
					mockState.scheduledPodsCount++
				}

				gotStatus := pl.PlacementFeasible(ctx, cycleState, pgInfo)

				if gotCode := gotStatus.Code(); gotCode != tc.expectedStatuses[i] {
					t.Errorf("Step %d: expected status %v, got %v", i, tc.expectedStatuses[i], gotCode)
				}
			}
		})
	}
}

type testPodGroupInfo struct {
	namespace       string
	name            string
	groupType       string
	unscheduledPods []*v1.Pod
	cpg             *schedulingapi.CompositePodGroup
	podGroup        *schedulingapi.PodGroup
}

func (t *testPodGroupInfo) GetNamespace() string          { return t.namespace }
func (t *testPodGroupInfo) GetName() string               { return t.name }
func (t *testPodGroupInfo) GetUnscheduledPods() []*v1.Pod { return t.unscheduledPods }
func (t *testPodGroupInfo) GetType() string               { return t.groupType }
func (t *testPodGroupInfo) GetKey() string {
	return fmt.Sprintf("%s/%s/%s", t.groupType, t.namespace, t.name)
}
func (t *testPodGroupInfo) GetPodGroup() *schedulingapi.PodGroup {
	return t.podGroup
}
func (t *testPodGroupInfo) GetCompositePodGroup() *schedulingapi.CompositePodGroup {
	return t.cpg
}

type testPodGroupState struct {
	scheduledPodsCount int
	parent             string
	hasParent          bool
	children           []string
}

func (t *testPodGroupState) AllPods() sets.Set[types.UID]        { return nil }
func (t *testPodGroupState) AllPodsCount() int                   { return 0 }
func (t *testPodGroupState) UnscheduledPods() map[string]*v1.Pod { return nil }
func (t *testPodGroupState) AssumedPods() sets.Set[types.UID]    { return nil }
func (t *testPodGroupState) AssignedPods() sets.Set[types.UID]   { return nil }
func (t *testPodGroupState) ScheduledPods() []*v1.Pod            { return nil }
func (t *testPodGroupState) ScheduledPodsCount() int             { return t.scheduledPodsCount }
func (t *testPodGroupState) GetParent() (string, bool)           { return t.parent, t.hasParent }
func (t *testPodGroupState) GetChildren() []string               { return t.children }

type mapPodGroupStateLister struct {
	states map[string]*testPodGroupState
}

func (l *mapPodGroupStateLister) Get(groupType, namespace, podGroupName string) (fwk.PodGroupState, error) {
	key := fmt.Sprintf("%s/%s/%s", groupType, namespace, podGroupName)
	state, exists := l.states[key]
	if !exists {
		return nil, fmt.Errorf("state not found for key %s", key)
	}
	return state, nil
}

type mockSharedListerWithMap struct {
	fwk.SharedLister
	podGroupStateLister *mapPodGroupStateLister
}

func (m *mockSharedListerWithMap) PodGroupStates() fwk.PodGroupStateLister {
	return m.podGroupStateLister
}

func TestCPGHierarchicalScheduling(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	_, ctx := ktesting.NewTestContext(t)

	// Tree structure:
	//
	//	              cpg-root (Gang, MinGroupCount: 2)
	//	            /                 |                 \
	//	   cpg-sub1 (Gang, Min:2)  cpg-sub2 (Basic)    cpg-sub3 (Gang, Min:2)
	//	   /    |    \              /       \             /         \
	//	 pg1   pg2   pg3          pg4       pg5         pg6        pg7
	//	(S)   (S)   (F)          (S)       (S)         (F)        (F)

	namespace := "default"

	cpgRoot := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-root"},
		Spec: schedulingapi.CompositePodGroupSpec{
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}
	cpgSub1 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-sub1"},
		Spec: schedulingapi.CompositePodGroupSpec{
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}
	cpgSub2 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-sub2"},
		Spec: schedulingapi.CompositePodGroupSpec{
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicGroupSchedulingPolicy{},
			},
		},
	}
	cpgSub3 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-sub3"},
		Spec: schedulingapi.CompositePodGroupSpec{
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}

	pg1 := st.MakePodGroup().Namespace(namespace).Name("pg1").ParentCompositePodGroup("cpg-sub1").MinCount(2).Obj()
	pg2 := st.MakePodGroup().Namespace(namespace).Name("pg2").ParentCompositePodGroup("cpg-sub1").MinCount(2).Obj()
	pg3 := st.MakePodGroup().Namespace(namespace).Name("pg3").ParentCompositePodGroup("cpg-sub1").MinCount(2).Obj()
	pg4 := st.MakePodGroup().Namespace(namespace).Name("pg4").ParentCompositePodGroup("cpg-sub2").BasicPolicy().Obj()
	pg5 := st.MakePodGroup().Namespace(namespace).Name("pg5").ParentCompositePodGroup("cpg-sub2").MinCount(2).Obj()
	pg6 := st.MakePodGroup().Namespace(namespace).Name("pg6").ParentCompositePodGroup("cpg-sub3").MinCount(2).Obj()
	pg7 := st.MakePodGroup().Namespace(namespace).Name("pg7").ParentCompositePodGroup("cpg-sub3").MinCount(2).Obj()

	informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(
		cpgRoot, cpgSub1, cpgSub2, cpgSub3,
		pg1, pg2, pg3, pg4, pg5, pg6, pg7,
	), 0)

	fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
		frameworkruntime.WithInformerFactory(informerFactory),
	)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	p, err := New(ctx, nil, fh, feature.Features{EnableGenericWorkload: true})
	if err != nil {
		t.Fatalf("Failed to create plugin: %v", err)
	}
	pl := p.(*GangScheduling)

	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	mockLister := &mockSharedListerWithMap{
		podGroupStateLister: &mapPodGroupStateLister{
			states: map[string]*testPodGroupState{
				"podgroup/default/pg1": {scheduledPodsCount: 2},
				"podgroup/default/pg2": {scheduledPodsCount: 2},
				"podgroup/default/pg3": {scheduledPodsCount: 0},
				"podgroup/default/pg4": {scheduledPodsCount: 1},
				"podgroup/default/pg5": {scheduledPodsCount: 2},
				"podgroup/default/pg6": {scheduledPodsCount: 0},
				"podgroup/default/pg7": {scheduledPodsCount: 0},

				"compositepodgroup/default/cpg-sub1": {
					children: []string{"podgroup/default/pg1", "podgroup/default/pg2", "podgroup/default/pg3"},
				},
				"compositepodgroup/default/cpg-sub2": {
					children: []string{"podgroup/default/pg4", "podgroup/default/pg5"},
				},
				"compositepodgroup/default/cpg-sub3": {
					children: []string{"podgroup/default/pg6", "podgroup/default/pg7"},
				},
				"compositepodgroup/default/cpg-root": {
					children: []string{"compositepodgroup/default/cpg-sub1", "compositepodgroup/default/cpg-sub2", "compositepodgroup/default/cpg-sub3"},
				},
			},
		},
	}
	pl.snapshotLister = mockLister

	sharedPodGroupCycleState := schedulerframework.NewCycleState()
	newCycleState := func() *schedulerframework.CycleState {
		cs := schedulerframework.NewCycleState()
		cs.SetPodGroupSchedulingCycle(sharedPodGroupCycleState)
		return cs
	}

	// Step 1: Run PlacementFeasible for leaf PodGroups
	statusPG1 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "pg1", groupType: schedulerframework.PodGroupKeyType, unscheduledPods: []*v1.Pod{{}}, podGroup: pg1})
	if statusPG1 != nil {
		t.Errorf("Expected success for pg1, got code: %v", statusPG1.Code())
	}

	statusPG2 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "pg2", groupType: schedulerframework.PodGroupKeyType, unscheduledPods: []*v1.Pod{{}}, podGroup: pg2})
	if statusPG2 != nil {
		t.Errorf("Expected success for pg2, got code: %v", statusPG2.Code())
	}

	statusPG3 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "pg3", groupType: schedulerframework.PodGroupKeyType, unscheduledPods: []*v1.Pod{{}}, podGroup: pg3})
	if statusPG3 == nil || statusPG3.Code() != fwk.Unschedulable {
		t.Errorf("Expected pg3 to be Unschedulable, got: %v", statusPG3)
	}

	statusPG4 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "pg4", groupType: schedulerframework.PodGroupKeyType, unscheduledPods: []*v1.Pod{{}}, podGroup: pg4})
	if statusPG4 != nil {
		t.Errorf("Expected success for pg4, got code: %v", statusPG4.Code())
	}

	statusPG5 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "pg5", groupType: schedulerframework.PodGroupKeyType, unscheduledPods: []*v1.Pod{{}}, podGroup: pg5})
	if statusPG5 != nil {
		t.Errorf("Expected success for pg5, got code: %v", statusPG5.Code())
	}

	statusPG6 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "pg6", groupType: schedulerframework.PodGroupKeyType, unscheduledPods: []*v1.Pod{{}}, podGroup: pg6})
	if statusPG6 == nil || statusPG6.Code() != fwk.Unschedulable {
		t.Errorf("Expected pg6 to be Unschedulable, got: %v", statusPG6)
	}

	statusPG7 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "pg7", groupType: schedulerframework.PodGroupKeyType, unscheduledPods: []*v1.Pod{{}}, podGroup: pg7})
	if statusPG7 == nil || statusPG7.Code() != fwk.Unschedulable {
		t.Errorf("Expected pg7 to be Unschedulable, got: %v", statusPG7)
	}

	// Step 2: Run PlacementFeasible for intermediate CompositePodGroups
	statusSub1 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "cpg-sub1", groupType: schedulerframework.CompositePodGroupKeyType})
	if statusSub1 != nil {
		t.Errorf("Expected success for cpg-sub1, got code: %v", statusSub1.Code())
	}

	statusSub2 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "cpg-sub2", groupType: schedulerframework.CompositePodGroupKeyType})
	if statusSub2 != nil {
		t.Errorf("Expected success for cpg-sub2, got code: %v", statusSub2.Code())
	}

	statusSub3 := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "cpg-sub3", groupType: schedulerframework.CompositePodGroupKeyType})
	if statusSub3 == nil || statusSub3.Code() != fwk.Unschedulable {
		t.Errorf("Expected cpg-sub3 to be Unschedulable, got: %v", statusSub3)
	}

	// Step 3: Run PlacementFeasible for root CompositePodGroup
	statusRoot := pl.PlacementFeasible(ctx, newCycleState(), &testPodGroupInfo{namespace: namespace, name: "cpg-root", groupType: schedulerframework.CompositePodGroupKeyType})
	if statusRoot != nil {
		t.Errorf("Expected success for cpg-root, got code: %v", statusRoot.Code())
	}
}

// TestCPGPreEnqueue_Hierarchical verifies that PreEnqueue correctly evaluates a large nested
// hierarchy of CompositePodGroups and PodGroups based on AllPodsCount().
// It dynamically adds pods and validates PreEnqueue status for all pods at each stage.
//
// Tree structure:
//
//	                     cpg-root (Gang, Min: 2)
//	             /                 |                 \
//	    cpg-sub1 (Gang, Min:2)  cpg-sub2 (Basic)    cpg-sub3 (Gang, Min:2)
//	    /    |    \              /       \             /         \
//	  pg1   pg2   pg3          pg4       pg5         pg6        pg7
//	Min:1  Min:1 Min:1       Basic      Min:1       Min:1      Min:1
func TestCPGPreEnqueue_Hierarchical(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	_, ctx := ktesting.NewTestContext(t)
	namespace := "default"

	cpgRoot := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-root"},
		Spec: schedulingapi.CompositePodGroupSpec{
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}
	cpgSub1 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-sub1"},
		Spec: schedulingapi.CompositePodGroupSpec{
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}
	cpgSub2 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-sub2"},
		Spec: schedulingapi.CompositePodGroupSpec{
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicGroupSchedulingPolicy{},
			},
		},
	}
	cpgSub3 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-sub3"},
		Spec: schedulingapi.CompositePodGroupSpec{
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}

	pg1 := st.MakePodGroup().Namespace(namespace).Name("pg1").ParentCompositePodGroup("cpg-sub1").MinCount(1).Obj()
	pg2 := st.MakePodGroup().Namespace(namespace).Name("pg2").ParentCompositePodGroup("cpg-sub1").MinCount(1).Obj()
	pg3 := st.MakePodGroup().Namespace(namespace).Name("pg3").ParentCompositePodGroup("cpg-sub1").MinCount(1).Obj()
	pg4 := st.MakePodGroup().Namespace(namespace).Name("pg4").ParentCompositePodGroup("cpg-sub2").BasicPolicy().Obj()
	pg5 := st.MakePodGroup().Namespace(namespace).Name("pg5").ParentCompositePodGroup("cpg-sub2").MinCount(1).Obj()
	pg6 := st.MakePodGroup().Namespace(namespace).Name("pg6").ParentCompositePodGroup("cpg-sub3").MinCount(1).Obj()
	pg7 := st.MakePodGroup().Namespace(namespace).Name("pg7").ParentCompositePodGroup("cpg-sub3").MinCount(1).Obj()

	informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(
		cpgRoot, cpgSub1, cpgSub2, cpgSub3,
		pg1, pg2, pg3, pg4, pg5, pg6, pg7,
	), 0)

	cache := internalcache.New(ctx, nil, true, true)

	podGroupInformer := informerFactory.Scheduling().V1alpha3().PodGroups()
	cpgInformer := informerFactory.Scheduling().V1alpha3().CompositePodGroups()
	for _, cpg := range []*schedulingapi.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3} {
		cpgInformer.Informer().GetStore().Add(cpg)
		cache.AddCompositePodGroup(cpg)
	}
	for _, pg := range []*schedulingapi.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7} {
		podGroupInformer.Informer().GetStore().Add(pg)
		cache.AddPodGroup(pg)
	}

	p1 := st.MakePod().Namespace(namespace).Name("p1").PodGroupName("pg1").Obj()
	p2 := st.MakePod().Namespace(namespace).Name("p2").PodGroupName("pg2").Obj()
	p3 := st.MakePod().Namespace(namespace).Name("p3").PodGroupName("pg3").Obj()
	p4 := st.MakePod().Namespace(namespace).Name("p4").PodGroupName("pg4").Obj()
	p5 := st.MakePod().Namespace(namespace).Name("p5").PodGroupName("pg5").Obj()
	p6 := st.MakePod().Namespace(namespace).Name("p6").PodGroupName("pg6").Obj()
	p7 := st.MakePod().Namespace(namespace).Name("p7").PodGroupName("pg7").Obj()

	allPods := []*v1.Pod{p1, p2, p3, p4, p5, p6, p7}

	snapshot := internalcache.NewEmptySnapshot()
	fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithPodGroupManager(cache),
		frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
	)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	p, err := New(ctx, nil, fh, feature.Features{EnableGenericWorkload: true})
	if err != nil {
		t.Fatalf("Failed to create plugin: %v", err)
	}
	pl := p.(*GangScheduling)

	checkAllPods := func(stage string, wantSuccess bool) {
		for _, pod := range allPods {
			status := pl.PreEnqueue(ctx, pod)
			if wantSuccess && status != nil {
				t.Errorf("[%s] Expected PreEnqueue for %s to succeed, but got: %v", stage, pod.Name, status)
			} else if !wantSuccess && status == nil {
				t.Errorf("[%s] Expected PreEnqueue for %s to fail, but it succeeded", stage, pod.Name)
			} else if !wantSuccess && status != nil && status.Code() != fwk.UnschedulableAndUnresolvable {
				t.Errorf("[%s] Expected PreEnqueue for %s to return UnschedulableAndUnresolvable, got: %v", stage, pod.Name, status)
			}
		}
	}

	// Stage 1: No pods added. Tree is not ready.
	checkAllPods("Stage 1 - No pods", false)

	// Stage 2: Add p1 and p2.
	// cpg-sub1 (Min 2) is now ready.
	// cpg-root (Min 2) is NOT ready because it only has 1 ready child (cpg-sub1).
	cache.AddPodGroupMember(p1)
	cache.AddPodGroupMember(p2)
	checkAllPods("Stage 2 - Add p1, p2", false)

	// Stage 3: Add p6.
	// cpg-sub3 (Min 2) has 1 ready child, so it's NOT ready.
	// cpg-root (Min 2) still only has 1 ready child (cpg-sub1).
	cache.AddPodGroupMember(p6)
	checkAllPods("Stage 3 - Add p6", false)

	// Stage 4: Add p4.
	// cpg-sub2 (Basic) has 1 ready child, so it IS ready.
	// cpg-root (Min 2) now has 2 ready children (cpg-sub1, cpg-sub2), so it IS ready!
	cache.AddPodGroupMember(p4)
	checkAllPods("Stage 4 - Add p4", true)

	// Stage 5: Add p7.
	// cpg-sub3 (Min 2) now has 2 ready children, so it IS ready.
	// cpg-root (Min 2) now has 3 ready children. Still ready.
	cache.AddPodGroupMember(p7)
	checkAllPods("Stage 5 - Add p7", true)
}

// TestCPGPreEnqueue_BasicWithGangChildren verifies that PreEnqueue handles a Basic root CPG correctly.
//
// Tree structure:
//
//	   cpg-root (Basic)
//	  /            \
//	pg1 (Gang)    pg2 (Gang)
//	Min: 2         Min: 2
//	(R)            (N)
//
// (R) = Ready (AllPodsCount >= MinCount)
// (N) = Not Ready (AllPodsCount < MinCount)
func TestCPGPreEnqueue_BasicWithGangChildren(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, true)

	_, ctx := ktesting.NewTestContext(t)
	namespace := "default"

	cpgRoot := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "cpg-root"},
		Spec: schedulingapi.CompositePodGroupSpec{
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{},
		},
	}

	pg1 := st.MakePodGroup().Namespace(namespace).Name("pg1").ParentCompositePodGroup("cpg-root").MinCount(2).Obj()
	pg2 := st.MakePodGroup().Namespace(namespace).Name("pg2").ParentCompositePodGroup("cpg-root").MinCount(2).Obj()

	informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(cpgRoot, pg1, pg2), 0)
	cache := internalcache.New(ctx, nil, true, true)

	podGroupInformer := informerFactory.Scheduling().V1alpha3().PodGroups()
	cpgInformer := informerFactory.Scheduling().V1alpha3().CompositePodGroups()

	cpgInformer.Informer().GetStore().Add(cpgRoot)
	cache.AddCompositePodGroup(cpgRoot)
	podGroupInformer.Informer().GetStore().Add(pg1)
	podGroupInformer.Informer().GetStore().Add(pg2)
	cache.AddPodGroup(pg1)
	cache.AddPodGroup(pg2)

	// Make pg1 ready (add 2 pods)
	p1_1 := st.MakePod().Namespace(namespace).Name("p1-1").UID("uid-p1-1").PodGroupName("pg1").Obj()
	p1_2 := st.MakePod().Namespace(namespace).Name("p1-2").UID("uid-p1-2").PodGroupName("pg1").Obj()
	cache.AddPodGroupMember(p1_1)
	cache.AddPodGroupMember(p1_2)

	// Make pg2 not ready (add 1 pod)
	p2_1 := st.MakePod().Namespace(namespace).Name("p2-1").UID("uid-p2-1").PodGroupName("pg2").Obj()
	cache.AddPodGroupMember(p2_1)

	snapshot := internalcache.NewEmptySnapshot()
	fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithPodGroupManager(cache),
		frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
		frameworkruntime.WithSnapshotSharedLister(snapshot),
	)
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	p, err := New(ctx, nil, fh, feature.Features{EnableGenericWorkload: true})
	if err != nil {
		t.Fatalf("Failed to create plugin: %v", err)
	}
	pl := p.(*GangScheduling)

	// pg1 is ready, and root is Basic (always ready), so p1_1 succeeds
	if status := pl.PreEnqueue(ctx, p1_1); status != nil {
		t.Errorf("Expected PreEnqueue for p1_1 to succeed, got %v", status)
	}

	// GangScheduling evaluates readiness at the root of the CPG hierarchy.
	// Because the root is Basic and pg1 is ready, the tree satisfies quorum.
	// Consequently, p2_1 passes PreEnqueue despite pg2 lacking quorum, and
	// will be rejected downstream during PlacementFeasible.
	status2 := pl.PreEnqueue(ctx, p2_1)
	if status2 != nil {
		t.Errorf("Expected PreEnqueue for p2_1 to succeed because root is ready, but got: %v", status2)
	}
}
