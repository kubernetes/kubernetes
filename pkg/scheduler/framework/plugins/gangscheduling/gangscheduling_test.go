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
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
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
)

func init() {
	// This is required for tests where cache is initialized, and cache attempts to update metrics.
	metrics.Register()
}

func Test_isSchedulableAfterPodAdded(t *testing.T) {
	tests := []struct {
		name                       string
		pod                        *v1.Pod
		newPod                     *v1.Pod
		pgs                        []*schedulingv1beta1.PodGroup
		cpgs                       []*schedulingv1alpha3.CompositePodGroup
		expectedHint               fwk.QueueingHint
		isCompositePodGroupEnabled bool
	}{
		{
			name:                       "add a newPod which matches the pod's scheduling group",
			pod:                        st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:                     st.MakePod().PodGroupName("pg").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "add a newPod which matches the pod's scheduling group (CPG=false)",
			pod:                        st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:                     st.MakePod().PodGroupName("pg").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "add a newPod with NodeName set which matches the pod's scheduling group",
			pod:                        st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:                     st.MakePod().PodGroupName("pg").Node("node1").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "add a newPod with NodeName set which matches the pod's scheduling group (CPG=false)",
			pod:                        st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod:                     st.MakePod().PodGroupName("pg").Node("node1").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: false,
		},
		{
			name:   "add a newPod which doesn't match the pod's namespace",
			pod:    st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod: st.MakePod().Namespace("foo").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg").Obj(),
				st.MakePodGroup().Namespace("foo").Name("pg").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:   "add a newPod which doesn't match the pod's namespace (CPG=false)",
			pod:    st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPod: st.MakePod().Namespace("foo").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg").Obj(),
				st.MakePodGroup().Namespace("foo").Name("pg").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:   "add a newPod which doesn't match the pod's pod group name",
			pod:    st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().PodGroupName("pg2").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Obj(),
				st.MakePodGroup().Name("pg2").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:   "add a newPod which doesn't match the pod's pod group name (CPG=false)",
			pod:    st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().PodGroupName("pg2").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Obj(),
				st.MakePodGroup().Name("pg2").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:   "add a newPod which belongs to a different scheduling group but matches the root CPG",
			pod:    st.MakePod().Name("p1").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().Name("p2").PodGroupName("pg2").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root").Obj(),
				st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Obj(),
			},
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: true,
		},
		{
			name:   "add a newPod which belongs to a different scheduling group but matches the root CPG (CPG=false)",
			pod:    st.MakePod().Name("p1").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().Name("p2").PodGroupName("pg2").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root").Obj(),
				st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:   "add a newPod which belongs to a different scheduling group and does not match the root CPG",
			pod:    st.MakePod().Name("p1").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().Name("p2").PodGroupName("pg2").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root1").Obj(),
				st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root2").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-root2").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:   "add a newPod which belongs to a different scheduling group and does not match the root CPG (CPG=false)",
			pod:    st.MakePod().Name("p1").PodGroupName("pg1").Obj(),
			newPod: st.MakePod().Name("p2").PodGroupName("pg2").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root1").Obj(),
				st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root2").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-root2").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, tc.isCompositePodGroupEnabled)
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
				if err := informerFactory.Scheduling().V1beta1().PodGroups().Informer().GetStore().Add(pg); err != nil {
					t.Fatalf("Failed to add podGroup %s to store: %v", pg.Name, err)
				}
			}
			for _, cpg := range tc.cpgs {
				if err := informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer().GetStore().Add(cpg); err != nil {
					t.Fatalf("Failed to add cpg %s to store: %v", cpg.Name, err)
				}
			}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate))
			if err == nil {
				pgMap := make(map[string]*schedulingv1beta1.PodGroup)
				for _, pg := range tc.pgs {
					pgMap[pg.Name] = pg
				}
				cpgMap := make(map[string]*schedulingv1alpha3.CompositePodGroup)
				for _, cpg := range tc.cpgs {
					cpgMap[cpg.Name] = cpg
				}
				p.(*GangScheduling).podGroupManager = &mockPodGroupManager{pgs: pgMap, cpgs: cpgMap}
			}
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
	tests := []struct {
		name                       string
		pod                        *v1.Pod
		newPodGroup                *schedulingv1beta1.PodGroup
		pgs                        []*schedulingv1beta1.PodGroup
		cpgs                       []*schedulingv1alpha3.CompositePodGroup
		expectedHint               fwk.QueueingHint
		isCompositePodGroupEnabled bool
	}{
		{
			name:                       "add a pod group which matches the pod's pod group name",
			pod:                        st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPodGroup:                st.MakePodGroup().Name("pg").MinCount(1).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "add a pod group which matches the pod's pod group name (CPG=false)",
			pod:                        st.MakePod().Name("p").PodGroupName("pg").Obj(),
			newPodGroup:                st.MakePodGroup().Name("pg").MinCount(1).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: false,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group name",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").MinCount(1).WorkloadRef("t", "w").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group name (CPG=false)",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").MinCount(1).WorkloadRef("t", "w").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group namespace",
			pod:         st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			newPodGroup: st.MakePodGroup().Namespace("ns2").Name("pg").MinCount(1).WorkloadRef("t", "w").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("ns1").Name("pg").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group namespace (CPG=false)",
			pod:         st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			newPodGroup: st.MakePodGroup().Namespace("ns2").Name("pg").MinCount(1).WorkloadRef("t", "w").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("ns1").Name("pg").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group but matches the root CPG",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root").MinCount(1).Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Obj(),
			},
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: true,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group but matches the root CPG (CPG=false)",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root").MinCount(1).Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group and doesn't match the root CPG",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root2").MinCount(1).Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root1").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-root2").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:        "add a pod group which doesn't match the pod's scheduling group and doesn't match the root CPG (CPG=false)",
			pod:         st.MakePod().Name("p").PodGroupName("pg1").Obj(),
			newPodGroup: st.MakePodGroup().Name("pg2").ParentCompositePodGroup("cpg-root2").MinCount(1).Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("pg1").ParentCompositePodGroup("cpg-root1").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("cpg-root1").Obj(),
				st.MakeCompositePodGroup().Name("cpg-root2").Obj(),
			},
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, tc.isCompositePodGroupEnabled)
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
				if err := informerFactory.Scheduling().V1beta1().PodGroups().Informer().GetStore().Add(pg); err != nil {
					t.Fatalf("Failed to add podGroup %s to store: %v", pg.Name, err)
				}
			}
			for _, cpg := range tc.cpgs {
				if err := informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer().GetStore().Add(cpg); err != nil {
					t.Fatalf("Failed to add cpg %s to store: %v", cpg.Name, err)
				}
			}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate))
			if err == nil {
				pgMap := make(map[string]*schedulingv1beta1.PodGroup)
				for _, pg := range tc.pgs {
					pgMap[pg.Name] = pg
				}
				if tc.newPodGroup != nil {
					pgMap[tc.newPodGroup.Name] = tc.newPodGroup
				}
				cpgMap := make(map[string]*schedulingv1alpha3.CompositePodGroup)
				for _, cpg := range tc.cpgs {
					cpgMap[cpg.Name] = cpg
				}
				p.(*GangScheduling).podGroupManager = &mockPodGroupManager{pgs: pgMap, cpgs: cpgMap}
			}
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
		name                       string
		pod                        *v1.Pod
		oldPodGroup                *schedulingv1beta1.PodGroup
		newPodGroup                *schedulingv1beta1.PodGroup
		expectedHint               fwk.QueueingHint
		expectErr                  bool
		isCompositePodGroupEnabled bool
	}{
		{
			name:                       "minCount decreased matches target pod",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "minCount decreased matches target pod (CPG=false)",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "update Basic policy",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").BasicPolicy().WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").BasicPolicy().Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "update Basic policy (CPG=false)",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").BasicPolicy().WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").BasicPolicy().Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "minCount increased matches target pod",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "minCount increased matches target pod (CPG=false)",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "minCount unchanged matches target pod",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "minCount unchanged matches target pod (CPG=false)",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "minCount decreased but pod group name doesn't match target pod",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg-other").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "minCount decreased but pod group name doesn't match target pod (CPG=false)",
			pod:                        st.MakePod().Namespace("ns1").Name("p").PodGroupName("pg-other").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "minCount decreased but pod group namespace doesn't match target pod",
			pod:                        st.MakePod().Namespace("ns-other").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "minCount decreased but pod group namespace doesn't match target pod (CPG=false)",
			pod:                        st.MakePod().Namespace("ns-other").Name("p").PodGroupName("pg").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name:                       "pod without a scheduling group is skipped",
			pod:                        st.MakePod().Namespace("ns1").Name("p").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name:                       "pod without a scheduling group is skipped (CPG=false)",
			pod:                        st.MakePod().Namespace("ns1").Name("p").Obj(),
			oldPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(4).WorkloadRef("t", "w").Obj(),
			newPodGroup:                st.MakePodGroup().Namespace("ns1").Name("pg").MinCount(3).WorkloadRef("t", "w").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, tc.isCompositePodGroupEnabled)
			logger, ctx := ktesting.NewTestContext(t)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate))
			if err == nil {
				pgMap := make(map[string]*schedulingv1beta1.PodGroup)
				if tc.oldPodGroup != nil {
					pgMap[tc.oldPodGroup.Name] = tc.oldPodGroup
				}
				if tc.newPodGroup != nil {
					pgMap[tc.newPodGroup.Name] = tc.newPodGroup
				}
				cpgMap := make(map[string]*schedulingv1alpha3.CompositePodGroup)
				p.(*GangScheduling).podGroupManager = &mockPodGroupManager{pgs: pgMap, cpgs: cpgMap}
			}
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

func Test_isSchedulableAfterCompositePodGroupAdded(t *testing.T) {
	tests := []struct {
		name                       string
		pod                        *v1.Pod
		newCPG                     *schedulingv1alpha3.CompositePodGroup
		cpgs                       []*schedulingv1alpha3.CompositePodGroup
		pgs                        []*schedulingv1beta1.PodGroup
		expectedHint               fwk.QueueingHint
		isCompositePodGroupEnabled bool
	}{
		{
			name: "add a CPG which matches the pod's root CPG",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-root").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: true,
		},
		{
			name: "add a CPG which matches the pod's root CPG (CPG=false)",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-root").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name: "add a CPG which matches the pod's root CPG",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-root").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name: "add a CPG which matches the pod's intermediate CPG but implies the same root",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-sub").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-sub").ParentCompositePodGroup("cpg-root").Obj(),
			expectedHint:               fwk.Queue,
			isCompositePodGroupEnabled: true,
		},
		{
			name: "add a CPG which matches the pod's intermediate CPG but implies the same root (CPG=false)",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-sub").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-sub").ParentCompositePodGroup("cpg-root").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name: "add a CPG which matches the pod's intermediate CPG but implies the same root",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-sub").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-root").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-sub").ParentCompositePodGroup("cpg-root").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name: "add a CPG which does not match the pod's CPG hierarchy",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-1").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-1").Obj(),
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-2").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-2").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: true,
		},
		{
			name: "add a CPG which does not match the pod's CPG hierarchy (CPG=false)",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-1").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-1").Obj(),
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-2").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-2").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
		{
			name: "add a CPG which does not match the pod's CPG hierarchy",
			pod:  st.MakePod().Namespace("default").Name("p").PodGroupName("pg").Obj(),
			pgs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace("default").Name("pg").ParentCompositePodGroup("cpg-1").Obj(),
			},
			cpgs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-1").Obj(),
				st.MakeCompositePodGroup().Namespace("default").Name("cpg-2").Obj(),
			},
			newCPG:                     st.MakeCompositePodGroup().Namespace("default").Name("cpg-2").Obj(),
			expectedHint:               fwk.QueueSkip,
			isCompositePodGroupEnabled: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, tc.isCompositePodGroupEnabled)
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
				if err := informerFactory.Scheduling().V1beta1().PodGroups().Informer().GetStore().Add(pg); err != nil {
					t.Fatalf("Failed to add podGroup %s to store: %v", pg.Name, err)
				}
			}
			for _, cpg := range tc.cpgs {
				if err := informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer().GetStore().Add(cpg); err != nil {
					t.Fatalf("Failed to add cpg %s to store: %v", cpg.Name, err)
				}
			}

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate))
			if err == nil {
				pgMap := make(map[string]*schedulingv1beta1.PodGroup)
				for _, pg := range tc.pgs {
					pgMap[pg.Name] = pg
				}
				cpgMap := make(map[string]*schedulingv1alpha3.CompositePodGroup)
				for _, cpg := range tc.cpgs {
					cpgMap[cpg.Name] = cpg
				}
				if tc.newCPG != nil {
					cpgMap[tc.newCPG.Name] = tc.newCPG
				}
				p.(*GangScheduling).podGroupManager = &mockPodGroupManager{pgs: pgMap, cpgs: cpgMap}
			}
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

type mockPodGroupState struct {
	fwk.PodGroupState
	scheduledPodsCount int
}

func (m *mockPodGroupState) ScheduledPodsCount() int { return m.scheduledPodsCount }

type mockPodGroupStateLister struct {
	state *mockPodGroupState
	err   error
}

func (m *mockPodGroupStateLister) Get(namespace, podGroupName string) (fwk.PodGroupState, error) {
	return m.state, m.err
}

type mockSharedLister struct {
	fwk.SharedLister
	podGroupStateLister *mockPodGroupStateLister
}

func (m *mockSharedLister) PodGroupStates() fwk.PodGroupStateLister {
	return m.podGroupStateLister
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

	cpgRoot := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "cpg-root"},
		Spec: schedulingv1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.CompositePodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: 2}},
		},
	}
	cpgSub1 := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "cpg-sub1"},
		Spec: schedulingv1alpha3.CompositePodGroupSpec{
			ParentCompositePodGroupName: new("cpg-root"),
			SchedulingPolicy:            schedulingv1alpha3.CompositePodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: 2}},
		},
	}
	cpgSub2 := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "cpg-sub2"},
		Spec: schedulingv1alpha3.CompositePodGroupSpec{
			ParentCompositePodGroupName: new("cpg-root"),
			SchedulingPolicy:            schedulingv1alpha3.CompositePodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.CompositeBasicSchedulingPolicy{}},
		},
	}
	cpgSub3 := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "cpg-sub3"},
		Spec: schedulingv1alpha3.CompositePodGroupSpec{
			ParentCompositePodGroupName: new("cpg-root"),
			SchedulingPolicy:            schedulingv1alpha3.CompositePodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: 2}},
		},
	}

	pg1 := st.MakePodGroup().Namespace("ns1").Name("pg1-cpg").ParentCompositePodGroup("cpg-sub1").MinCount(1).Obj()
	pg2 := st.MakePodGroup().Namespace("ns1").Name("pg2-cpg").ParentCompositePodGroup("cpg-sub1").MinCount(1).Obj()
	pg3 := st.MakePodGroup().Namespace("ns1").Name("pg3-cpg").ParentCompositePodGroup("cpg-sub1").MinCount(1).Obj()
	pg4 := st.MakePodGroup().Namespace("ns1").Name("pg4-cpg").ParentCompositePodGroup("cpg-sub2").BasicPolicy().Obj()
	pg5 := st.MakePodGroup().Namespace("ns1").Name("pg5-cpg").ParentCompositePodGroup("cpg-sub2").MinCount(1).Obj()
	pg6 := st.MakePodGroup().Namespace("ns1").Name("pg6-cpg").ParentCompositePodGroup("cpg-sub3").MinCount(1).Obj()
	pg7 := st.MakePodGroup().Namespace("ns1").Name("pg7-cpg").ParentCompositePodGroup("cpg-sub3").MinCount(1).Obj()

	p1CPG := st.MakePod().Namespace("ns1").Name("p1-cpg").UID("p1-cpg").PodGroupName("pg1-cpg").Obj()
	p2CPG := st.MakePod().Namespace("ns1").Name("p2-cpg").UID("p2-cpg").PodGroupName("pg2-cpg").Obj()
	p4CPG := st.MakePod().Namespace("ns1").Name("p4-cpg").UID("p4-cpg").PodGroupName("pg4-cpg").Obj()
	p6CPG := st.MakePod().Namespace("ns1").Name("p6-cpg").UID("p6-cpg").PodGroupName("pg6-cpg").Obj()

	cpgBasicRoot := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns1", Name: "cpg-basic-root"},
		Spec: schedulingv1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: schedulingv1alpha3.CompositePodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.CompositeBasicSchedulingPolicy{}},
		},
	}
	pgBasic1 := st.MakePodGroup().Namespace("ns1").Name("pg-basic-1").ParentCompositePodGroup("cpg-basic-root").MinCount(2).Obj()
	pgBasic2 := st.MakePodGroup().Namespace("ns1").Name("pg-basic-2").ParentCompositePodGroup("cpg-basic-root").MinCount(2).Obj()
	p1_1 := st.MakePod().Namespace("ns1").Name("p1_1").UID("p1_1").PodGroupName("pg-basic-1").Obj()
	p1_2 := st.MakePod().Namespace("ns1").Name("p1_2").UID("p1_2").PodGroupName("pg-basic-1").Obj()
	p2_1 := st.MakePod().Namespace("ns1").Name("p2_1").UID("p2_1").PodGroupName("pg-basic-2").Obj()

	type testCase struct {
		name                            string
		pod                             *v1.Pod
		initialPods                     []*v1.Pod
		initialPodGroups                []*schedulingv1beta1.PodGroup
		initialCompositePodGroups       []*schedulingv1alpha3.CompositePodGroup
		podsWaitingOnPermit             []*v1.Pod
		isDuringPodGroupSchedulingCycle bool
		isCompositePodGroupEnabled      bool
		wantPreEnqueueStatus            *fwk.Status
		wantPermitStatus                *fwk.Status
		wantActivatedPods               []*v1.Pod
		wantAllowedPods                 []types.UID
	}
	baseTests := []testCase{
		{
			name:                       "non-gang pod succeeds immediately (CPG=false)",
			pod:                        nonGangPod,
			isCompositePodGroupEnabled: false,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroup},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
		},
		{
			name:                       "non-gang pod succeeds immediately (CPG=true)",
			pod:                        nonGangPod,
			isCompositePodGroupEnabled: true,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroup},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
		},
		{
			name:                       "basic policy pod succeeds immediately (CPG=false)",
			pod:                        basicPolicyPod,
			isCompositePodGroupEnabled: false,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroup},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
		},
		{
			name:                       "basic policy pod succeeds immediately (CPG=true)",
			pod:                        basicPolicyPod,
			isCompositePodGroupEnabled: true,
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2, basicPodGroup},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
		},
		{
			name:                       "gang pod fails PreEnqueue when pod group is not yet created (CPG=false)",
			pod:                        p1,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{},
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `waiting for pods's pod group "pg1" to appear in scheduling queue`),
		},
		{
			name:                       "gang pod fails PreEnqueue when pod group is not yet created (CPG=true)",
			pod:                        p1,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{},
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "failed to build hierarchy snapshot: pod group object not found in state for podgroup/ns1/pg1"),
		},
		{
			name:                       "gang pod fails PreEnqueue when quorum is not met (CPG=false)",
			pod:                        p1,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{p2, p4, p5},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2},
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue"),
		},
		{
			name:                       "gang pod fails PreEnqueue when quorum is not met (CPG=true)",
			pod:                        p1,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{p2, p4, p5},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2},
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue"),
		},
		{
			name:                       "gang pod passes PreEnqueue, but waits at Permit (CPG=false)",
			pod:                        p1,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2},
			podsWaitingOnPermit:        []*v1.Pod{p2, p4, p5},
			wantPreEnqueueStatus:       nil,
			wantActivatedPods:          []*v1.Pod{p3},
			wantPermitStatus:           fwk.NewStatus(fwk.Wait, "waiting for minCount pods from a gang to be scheduled"),
		},
		{
			name:                       "gang pod passes PreEnqueue, but waits at Permit (CPG=true)",
			pod:                        p1,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2},
			podsWaitingOnPermit:        []*v1.Pod{p2, p4, p5},
			wantPreEnqueueStatus:       nil,
			wantActivatedPods:          []*v1.Pod{p3},
			wantPermitStatus:           fwk.NewStatus(fwk.Wait, "waiting for minCount pods from a gang to be scheduled"),
		},
		{
			name:                       "final gang pod arrives at Permit and allows all waiting pods from a gang (CPG=false)",
			pod:                        p1,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2},
			podsWaitingOnPermit:        []*v1.Pod{p2, p3, p4, p5},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
			wantAllowedPods:            []types.UID{"p1", "p2", "p3"},
		},
		{
			name:                       "final gang pod arrives at Permit and allows all waiting pods from a gang (CPG=true)",
			pod:                        p1,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{p2, p3, p4, p5},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{gangPodGroup1, gangPodGroup2},
			podsWaitingOnPermit:        []*v1.Pod{p2, p3, p4, p5},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
			wantAllowedPods:            []types.UID{"p1", "p2", "p3"},
		},
		{
			name:                       "CPG Hierarchical Stage 1: No pods, tree not ready (CPG=true)",
			pod:                        p1CPG,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3},
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for composite pod group \"cpg-root\" tree to meet quorum"),
		},
		{
			name:                       "CPG Hierarchical Stage 1: No pods, tree not ready (CPG=false)",
			pod:                        p1CPG,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
		},
		{
			name:                       "CPG Hierarchical Stage 2: Add p6, cpg-sub1 ready but root not ready (CPG=true)",
			pod:                        p6CPG,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{p1CPG, p2CPG},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3},
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for composite pod group \"cpg-root\" tree to meet quorum"),
		},
		{
			name:                       "CPG Hierarchical Stage 2: Add p6 (CPG=false)",
			pod:                        p6CPG,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{p1CPG, p2CPG},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
		},
		{
			name:                       "CPG Hierarchical Stage 3: Add p4, root becomes ready (CPG=true)",
			pod:                        p4CPG,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{p1CPG, p2CPG, p6CPG},
			podsWaitingOnPermit:        []*v1.Pod{p1CPG, p2CPG, p6CPG},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
		},
		{
			name:                       "CPG Hierarchical Stage 3: Add p4 (CPG=false)",
			pod:                        p4CPG,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{p1CPG, p2CPG, p6CPG},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3},
			wantPreEnqueueStatus:       nil,
		},
		{
			name:                       "CPG Basic With Gang Stage 1: pg1 ready, root ready (CPG=true)",
			pod:                        p2_1,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{p1_1, p1_2},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pgBasic1, pgBasic2},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgBasicRoot},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           fwk.NewStatus(fwk.Wait, "waiting for composite pod group \"cpg-basic-root\" tree to meet quorum"),
		},
		{
			name:                       "CPG Basic With Gang Stage 1: pg1 ready, root ready (CPG=false)",
			pod:                        p2_1,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{p1_1, p1_2},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pgBasic1, pgBasic2},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgBasicRoot},
			wantPreEnqueueStatus:       fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue"),
		},
		{
			name:                       "CPG Permit Wait: Hierarchical gang waits at permit if gang isn't fully scheduled (CPG=true)",
			pod:                        p4CPG,
			isCompositePodGroupEnabled: true,
			initialPods:                []*v1.Pod{p1CPG, p2CPG, p6CPG},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           fwk.NewStatus(fwk.Wait, "waiting for composite pod group \"cpg-root\" tree to meet quorum"),
		},
		{
			name:                       "CPG Permit Wait: Hierarchical gang waits at permit if gang isn't fully scheduled (CPG=false)",
			pod:                        p4CPG,
			isCompositePodGroupEnabled: false,
			initialPods:                []*v1.Pod{p1CPG, p2CPG, p6CPG},
			initialPodGroups:           []*schedulingv1beta1.PodGroup{pg1, pg2, pg3, pg4, pg5, pg6, pg7},
			initialCompositePodGroups:  []*schedulingv1alpha3.CompositePodGroup{cpgRoot, cpgSub1, cpgSub2, cpgSub3},
			wantPreEnqueueStatus:       nil,
			wantPermitStatus:           nil,
		},
	}

	for _, tt := range baseTests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			if tt.isCompositePodGroupEnabled {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, tt.isCompositePodGroupEnabled)
			logger, ctx := ktesting.NewTestContext(t)
			cache := internalcache.New(ctx, nil, true, tt.isCompositePodGroupEnabled)

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			podGroupInformer := informerFactory.Scheduling().V1beta1().PodGroups()
			if tt.isCompositePodGroupEnabled {
				informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer()
			}
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
			for _, pg := range tt.initialPodGroups {
				err := podGroupInformer.Informer().GetStore().Add(pg)
				if err != nil {
					t.Fatalf("Failed to add podGroup %s to store: %v", pg.Name, err)
				}
				cache.AddPodGroup(pg)
			}
			if tt.isCompositePodGroupEnabled {
				for _, cpg := range tt.initialCompositePodGroups {
					err := informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer().GetStore().Add(cpg)
					if err != nil {
						t.Fatalf("Failed to add cpg %s to store: %v", cpg.Name, err)
					}
					cache.AddCompositePodGroup(logger, cpg)
				}
			}

			for _, p := range tt.initialPods {
				cache.AddPodGroupMember(p)
			}
			cache.AddPodGroupMember(tt.pod)

			p, err := New(ctx, nil, fh, feature.Features{EnableGenericWorkload: true, EnableCompositePodGroup: tt.isCompositePodGroupEnabled})
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
		cpgFeatureGate        bool
		isCPG                 bool
		minCount              int32
		childrenCount         int
		unscheduledPods       []*v1.Pod
		podStatuses           []fwk.Code
		expectedStatuses      []fwk.Code
		initialScheduledCount int
	}{
		{
			name:                  "All pods succeed, minCount met at end (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "All pods succeed, minCount met at end (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "All pods succeed, minCount met at end (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              2,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "First pod fails, minCount not satisfiable (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              3,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "First pod fails, minCount not satisfiable (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              3,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "First pod fails, minCount not satisfiable (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              3,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "Second pod fails, minCount not satisfiable (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "Second pod fails, minCount not satisfiable (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "Second pod fails, minCount not satisfiable (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              2,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with 0 scheduled pods fails (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with 0 scheduled pods fails (isCPG=true, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 true,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with 0 scheduled pods fails (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with 0 scheduled pods fails (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with succeeds once 1 pod is scheduled (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with succeeds once 1 pod is scheduled (isCPG=true, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 true,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with succeeds once 1 pod is scheduled (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with succeeds once 1 pod is scheduled (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "Non-gang pod group with 1 initially scheduled pod succeeds (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 1,
		},
		{
			name:                  "Non-gang pod group with 1 initially scheduled pod succeeds (isCPG=true, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 true,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 1,
		},
		{
			name:                  "Non-gang pod group with 1 initially scheduled pod succeeds (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 1,
		},
		{
			name:                  "Non-gang pod group with 1 initially scheduled pod succeeds (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              0,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 1,
		},
		{
			name:                  "More than minCount pods, all succeed (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "More than minCount pods, all succeed (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "More than minCount pods, all succeed (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "More than minCount pods, first fails (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable, fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Wait, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "More than minCount pods, first fails (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable, fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Wait, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "More than minCount pods, first fails (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable, fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Wait, fwk.Success},
			initialScheduledCount: 0,
		},
		{
			name:                  "More than minCount pods, minCount not satisfiable (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable, fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "More than minCount pods, minCount not satisfiable (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable, fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "More than minCount pods, minCount not satisfiable (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              2,
			childrenCount:         3,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj(), st.MakePod().Name("p3").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable, fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Unschedulable},
			initialScheduledCount: 0,
		},
		{
			name:                  "1 pod scheduled, 2 unscheduled pods succeed, minCount 3 met (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              3,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success},
			initialScheduledCount: 1,
		},
		{
			name:                  "1 pod scheduled, 2 unscheduled pods succeed, minCount 3 met (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              3,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success},
			initialScheduledCount: 1,
		},
		{
			name:                  "1 pod scheduled, 2 unscheduled pods succeed, minCount 3 met (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              3,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success, fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Wait, fwk.Success},
			initialScheduledCount: 1,
		},
		{
			name:                  "minCount already met by scheduled pods (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 2,
		},
		{
			name:                  "minCount already met by scheduled pods (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              2,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 2,
		},
		{
			name:                  "minCount already met by scheduled pods (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              2,
			childrenCount:         1,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Success},
			initialScheduledCount: 2,
		},
		{
			name:                  "1 pod scheduled, minCount 3, first unscheduled fails, not enough remaining (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              3,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 1,
		},
		{
			name:                  "1 pod scheduled, minCount 3, first unscheduled fails, not enough remaining (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              3,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 1,
		},
		{
			name:                  "1 pod scheduled, minCount 3, first unscheduled fails, not enough remaining (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              3,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Unschedulable},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 1,
		},
		{
			name:                  "1 pod scheduled, minCount 4, first unscheduled succeeds, not enough remaining (isCPG=false, cpgFeatureGate=false)",
			cpgFeatureGate:        false,
			isCPG:                 false,
			minCount:              4,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 1,
		},
		{
			name:                  "1 pod scheduled, minCount 4, first unscheduled succeeds, not enough remaining (isCPG=false, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 false,
			minCount:              4,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 1,
		},
		{
			name:                  "1 pod scheduled, minCount 4, first unscheduled succeeds, not enough remaining (isCPG=true, cpgFeatureGate=true)",
			cpgFeatureGate:        true,
			isCPG:                 true,
			minCount:              4,
			childrenCount:         2,
			unscheduledPods:       []*v1.Pod{st.MakePod().Name("p1").Obj(), st.MakePod().Name("p2").Obj()},
			podStatuses:           []fwk.Code{fwk.Success},
			expectedStatuses:      []fwk.Code{fwk.Unschedulable},
			initialScheduledCount: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			if tc.cpgFeatureGate {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CompositePodGroup, tc.cpgFeatureGate)
			_, ctx := ktesting.NewTestContext(t)

			pgName := "test-pg"
			namespace := "default"

			pgInfo := &testPodGroupInfo{
				namespace:       namespace,
				name:            pgName,
				unscheduledPods: tc.unscheduledPods,
			}

			var objs []runtime.Object

			if tc.isCPG {
				cpg := &schedulingv1alpha3.CompositePodGroup{
					ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: pgName},
					Spec:       schedulingv1alpha3.CompositePodGroupSpec{},
				}
				if tc.minCount > 0 {
					cpg.Spec.SchedulingPolicy.Gang = &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: tc.minCount}
				} else {
					cpg.Spec.SchedulingPolicy.Basic = &schedulingv1alpha3.CompositeBasicSchedulingPolicy{}
				}
				pgInfo.groupType = fwk.CompositePodGroupKeyType
				pgInfo.cpg = cpg
				objs = append(objs, cpg)
			} else {
				pg := st.MakePodGroup().Namespace(namespace).Name(pgName).ParentCompositePodGroup("cpg-root").Obj()
				if tc.minCount > 0 {
					pg.Spec.SchedulingPolicy.Gang = &schedulingv1beta1.GangSchedulingPolicy{MinCount: tc.minCount}
				} else {
					pg.Spec.SchedulingPolicy.Basic = &schedulingv1beta1.BasicSchedulingPolicy{}
				}
				pgInfo.groupType = fwk.PodGroupKeyType
				pgInfo.podGroup = pg
				objs = append(objs, pg)
			}

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(objs...), 0)
			informerFactory.Scheduling().V1beta1().PodGroups().Informer()
			if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
				informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer()
			}
			informerFactory.StartWithContext(ctx)
			informerFactory.WaitForCacheSyncWithContext(ctx)

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			p, err := New(ctx, nil, fh, feature.Features{EnableGenericWorkload: true, EnableCompositePodGroup: tc.cpgFeatureGate})
			if err != nil {
				t.Fatalf("Failed to create plugin: %v", err)
			}
			pl := p.(*GangScheduling)

			mockState := &mockPodGroupState{scheduledPodsCount: tc.initialScheduledCount}
			mockLister := &mockSharedLister{
				podGroupStateLister: &mockPodGroupStateLister{state: mockState},
			}
			pl.snapshotLister = mockLister

			cycleState := schedulerframework.NewCycleState()
			cycleState.SetPodGroupSchedulingCycle(cycleState)

			scheduled := tc.initialScheduledCount
			for i, code := range tc.podStatuses {
				if code == fwk.Success {
					scheduled++
					mockState.scheduledPodsCount++
				}

				args := schedulerframework.PlacementProgress{
					Remaining: tc.childrenCount - (i + 1),
					Scheduled: scheduled,
				}
				gotStatus := pl.PlacementFeasible(ctx, cycleState, pgInfo, args)

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
	groupType       fwk.EntityKeyType
	unscheduledPods []*v1.Pod
	cpg             *schedulingv1alpha3.CompositePodGroup
	podGroup        *schedulingv1beta1.PodGroup
}

func (t *testPodGroupInfo) GetNamespace() string                     { return t.namespace }
func (t *testPodGroupInfo) GetName() string                          { return t.name }
func (t *testPodGroupInfo) GetUnscheduledPods() []*v1.Pod            { return t.unscheduledPods }
func (t *testPodGroupInfo) GetPodGroup() *schedulingv1beta1.PodGroup { return t.podGroup }
func (t *testPodGroupInfo) GetType() fwk.EntityKeyType               { return t.groupType }
func (t *testPodGroupInfo) GetKey() string {
	return fmt.Sprintf("%s/%s/%s", t.groupType, t.namespace, t.name)
}
func (t *testPodGroupInfo) GetCompositePodGroup() *schedulingv1alpha3.CompositePodGroup {
	return t.cpg
}
func (t *testPodGroupInfo) GetChildren() []fwk.PodGroupInfo {
	return nil
}

type mockPodGroupManager struct {
	fwk.PodGroupManager
	pgs  map[string]*schedulingv1beta1.PodGroup
	cpgs map[string]*schedulingv1alpha3.CompositePodGroup
}

func (m *mockPodGroupManager) GetRootKeyForGroup(key fwk.EntityKey) (fwk.EntityKey, bool, error) {
	currentKey := key
	for {
		switch currentKey.Type {
		case fwk.PodKeyType:
			return currentKey, true, nil
		case fwk.PodGroupKeyType:
			pg, ok := m.pgs[currentKey.Name]
			if !ok {
				return currentKey, true, nil
			}
			if pg.Spec.ParentCompositePodGroupName == nil || !utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
				return currentKey, true, nil
			}
			currentKey = fwk.CompositePodGroupKey(currentKey.Namespace, *pg.Spec.ParentCompositePodGroupName)
		case fwk.CompositePodGroupKeyType:
			cpg, ok := m.cpgs[currentKey.Name]
			if !ok {
				return currentKey, true, nil
			}
			if cpg.Spec.ParentCompositePodGroupName == nil || !utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
				return currentKey, true, nil
			}
			currentKey = fwk.CompositePodGroupKey(currentKey.Namespace, *cpg.Spec.ParentCompositePodGroupName)
		default:
			return currentKey, true, nil
		}
	}
}
