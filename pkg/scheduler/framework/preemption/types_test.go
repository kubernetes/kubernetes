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

package preemption

import (
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestGetHighestAllAncestor(t *testing.T) {
	tests := []struct {
		name               string
		pod                *v1.Pod
		podGroups          map[string]*schedulingv1beta1.PodGroup
		compositePodGroups map[string]*schedulingv1alpha3.CompositePodGroup
		wantHighestAllKey  fwk.EntityKey
		wantHasAll         bool
	}{
		{
			name:       "pod without group",
			pod:        st.MakePod().Name("p1").Namespace("default").Obj(),
			wantHasAll: false,
		},
		{
			name:       "pod group missing from the lister",
			pod:        st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups:  nil,
			wantHasAll: false,
		},
		{
			name: "pod group disruption mode single",
			pod:  st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").Namespace("default").DisruptionModeSingle().Obj(),
			},
			wantHasAll: false,
		},
		{
			name: "pod group disruption mode All",
			pod:  st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").Namespace("default").DisruptionModeAll().Obj(),
			},
			wantHighestAllKey: fwk.PodGroupKey("default", "pg1"),
			wantHasAll:        true,
		},
		{
			name: "parent CPG disruption mode All, child PG disruption mode single",
			pod:  st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": {
					ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
					Spec: schedulingv1beta1.PodGroupSpec{
						DisruptionMode:              &schedulingv1beta1.DisruptionMode{Single: &schedulingv1beta1.SingleDisruptionMode{}},
						ParentCompositePodGroupName: new("cpg1"),
					},
				},
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": {
					ObjectMeta: metav1.ObjectMeta{Name: "cpg1", Namespace: "default"},
					Spec: schedulingv1alpha3.CompositePodGroupSpec{
						DisruptionMode: &schedulingv1alpha3.CompositeDisruptionMode{All: &schedulingv1alpha3.AllCompositeDisruptionMode{}},
					},
				},
			},
			wantHighestAllKey: fwk.CompositePodGroupKey("default", "cpg1"),
			wantHasAll:        true,
		},
		{
			name: "grandparent CPG disruption mode All, parent CPG disruption mode single, child PG disruption mode single",
			pod:  st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": {
					ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
					Spec: schedulingv1beta1.PodGroupSpec{
						DisruptionMode:              &schedulingv1beta1.DisruptionMode{Single: &schedulingv1beta1.SingleDisruptionMode{}},
						ParentCompositePodGroupName: new("cpg1"),
					},
				},
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": {
					ObjectMeta: metav1.ObjectMeta{Name: "cpg1", Namespace: "default"},
					Spec: schedulingv1alpha3.CompositePodGroupSpec{
						DisruptionMode:              &schedulingv1alpha3.CompositeDisruptionMode{Single: &schedulingv1alpha3.SingleCompositeDisruptionMode{}},
						ParentCompositePodGroupName: new("cpg2"),
					},
				},
				"cpg2": {
					ObjectMeta: metav1.ObjectMeta{Name: "cpg2", Namespace: "default"},
					Spec: schedulingv1alpha3.CompositePodGroupSpec{
						DisruptionMode: &schedulingv1alpha3.CompositeDisruptionMode{All: &schedulingv1alpha3.AllCompositeDisruptionMode{}},
					},
				},
			},
			wantHighestAllKey: fwk.CompositePodGroupKey("default", "cpg2"),
			wantHasAll:        true,
		},
		{
			name: "highest All parent wins (both have disruption mode All)",
			pod:  st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": {
					ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
					Spec: schedulingv1beta1.PodGroupSpec{
						DisruptionMode:              &schedulingv1beta1.DisruptionMode{All: &schedulingv1beta1.AllDisruptionMode{}},
						ParentCompositePodGroupName: new("cpg1"),
					},
				},
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": {
					ObjectMeta: metav1.ObjectMeta{Name: "cpg1", Namespace: "default"},
					Spec: schedulingv1alpha3.CompositePodGroupSpec{
						DisruptionMode: &schedulingv1alpha3.CompositeDisruptionMode{All: &schedulingv1alpha3.AllCompositeDisruptionMode{}},
					},
				},
			},
			wantHighestAllKey: fwk.CompositePodGroupKey("default", "cpg1"),
			wantHasAll:        true,
		},
		{
			name:       "pod group missing from cache lister",
			pod:        st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups:  nil,
			wantHasAll: false,
		},
		{
			name: "parent CPG missing from cache lister, child PG All returned",
			pod:  st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": {
					ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
					Spec: schedulingv1beta1.PodGroupSpec{
						DisruptionMode:              &schedulingv1beta1.DisruptionMode{All: &schedulingv1beta1.AllDisruptionMode{}},
						ParentCompositePodGroupName: new("cpg1"),
					},
				},
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{},
			wantHighestAllKey:  fwk.PodGroupKey("default", "pg1"),
			wantHasAll:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pgLister := &mockPodGroupLister{podGroups: tt.podGroups}
			var cpgLister fwk.CompositePodGroupLister
			if tt.compositePodGroups != nil {
				cpgLister = &mockCompositePodGroupLister{compositePodGroups: tt.compositePodGroups}
			}
			gotKey, gotHasAll := getHighestAllAncestor(tt.pod, pgLister, cpgLister)
			if gotHasAll != tt.wantHasAll {
				t.Errorf("hasAll mismatch: got %v, want %v", gotHasAll, tt.wantHasAll)
			}
			if gotHasAll && gotKey != tt.wantHighestAllKey {
				t.Errorf("highestAllKey mismatch: got %v, want %v", gotKey, tt.wantHighestAllKey)
			}
		})
	}
}

func TestNewPodGroupPreemptorResolvesPreemptionPolicy(t *testing.T) {
	preemptNeverPod := st.MakePod().Name("p-never").PreemptionPolicy(v1.PreemptNever).Obj()
	preemptLowerPriorityPod := st.MakePod().Name("p-lower").PreemptionPolicy(v1.PreemptLowerPriority).Obj()
	noPolicyPod := st.MakePod().Name("p-nil").Obj()

	tests := []struct {
		name                           string
		pg                             *schedulingv1beta1.PodGroup
		cpg                            *schedulingv1alpha3.CompositePodGroup
		pods                           []*v1.Pod
		enablePodGroupPreemptionPolicy bool
		wantPolicy                     schedulingv1beta1.PreemptionPolicy
	}{
		{
			name:                           "PreemptionPolicy PreemptNever is resolved from PodGroup, ignoring policy in pod, with PodGroupPreemptionPolicy enabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     schedulingv1beta1.PreemptNever,
		},
		{
			name:                           "PreemptionPolicy PreemptLowerPriority is resolved from PodGroup, ignoring different policies in pods, with PodGroupPreemptionPolicy enabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     schedulingv1beta1.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy is resolved from pods with PodGroupPreemptionPolicy disabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod},
			enablePodGroupPreemptionPolicy: false,
			wantPolicy:                     schedulingv1beta1.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy is resolved from pods when multiple pods have different policies, with PodGroupPreemptionPolicy disabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingv1beta1.PreemptLowerPriority).Obj(),
			pods:                           []*v1.Pod{preemptNeverPod, preemptLowerPriorityPod, noPolicyPod},
			enablePodGroupPreemptionPolicy: false,
			wantPolicy:                     schedulingv1beta1.PreemptNever,
		},
		{
			name:       "PreemptionPolicy is resolved from pods when CompositePodGroup is active: PreemptLowerPriority when no pod is PreemptNever",
			cpg:        st.MakeCompositePodGroup().Name("cpg1").Obj(),
			pods:       []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			wantPolicy: schedulingv1beta1.PreemptLowerPriority,
		},
		{
			name:       "PreemptionPolicy is resolved from pods when CompositePodGroup is active: PreemptNever when any pod is PreemptNever",
			cpg:        st.MakeCompositePodGroup().Name("cpg1").Obj(),
			pods:       []*v1.Pod{preemptNeverPod, preemptLowerPriorityPod},
			wantPolicy: schedulingv1beta1.PreemptNever,
		},
		{
			name:                           "PreemptionPolicy PreemptNever is resolved from CompositePodGroup, ignoring policy in pod, with PodGroupPreemptionPolicy enabled",
			cpg:                            st.MakeCompositePodGroup().Name("cpg1").PreemptionPolicy(schedulingv1alpha3.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     schedulingv1beta1.PreemptNever,
		},
		{
			name:                           "PreemptionPolicy PreemptLowerPriority is resolved from CompositePodGroup, ignoring different policies in pods, with PodGroupPreemptionPolicy enabled",
			cpg:                            st.MakeCompositePodGroup().Name("cpg1").PreemptionPolicy(schedulingv1alpha3.PreemptLowerPriority).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod, preemptNeverPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     schedulingv1beta1.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy is resolved from pods when CompositePodGroup has policy but PodGroupPreemptionPolicy is disabled",
			cpg:                            st.MakeCompositePodGroup().Name("cpg1").PreemptionPolicy(schedulingv1alpha3.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			enablePodGroupPreemptionPolicy: false,
			wantPolicy:                     schedulingv1beta1.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy defaults to PreemptLowerPriority when CompositePodGroup has no policy set with PodGroupPreemptionPolicy enabled, even if pods are PreemptNever",
			cpg:                            st.MakeCompositePodGroup().Name("cpg1").Obj(),
			pods:                           []*v1.Pod{preemptNeverPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     schedulingv1beta1.PreemptLowerPriority,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preemptor := newPodGroupPreemptor(&testPodGroupInfo{pg: tt.pg, cpg: tt.cpg, pods: tt.pods}, tt.enablePodGroupPreemptionPolicy)
			if preemptor.preemptionPolicy != tt.wantPolicy {
				t.Errorf("expected preemption policy %q, got %q", tt.wantPolicy, preemptor.preemptionPolicy)
			}
		})
	}
}

type expectedVictim struct {
	pods          sets.Set[string]
	affectedNodes sets.Set[string]
	priority      int32
}

func TestNewDomainForWorkloadPreemption(t *testing.T) {
	tests := []struct {
		name                    string
		nodes                   []*v1.Node
		pods                    []*v1.Pod
		podGroups               map[string]*schedulingv1beta1.PodGroup
		compositePodGroups      map[string]*schedulingv1alpha3.CompositePodGroup
		enableCompositePodGroup bool
		domainName              string
		wantVictims             []expectedVictim
	}{
		{
			name: "no pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
			},
			pods:        nil,
			podGroups:   nil,
			domainName:  "test-domain",
			wantVictims: nil,
		},
		{
			name: "pods without pod groups",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(10).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(20).Obj(),
			},
			podGroups:  nil,
			domainName: "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2"), affectedNodes: sets.New("node2"), priority: 20},
			},
		},
		{
			name: "pods with pod group (DisruptionModeAll)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(50).Obj(),
			},
			domainName: "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1", "p2"), affectedNodes: sets.New("node1", "node2"), priority: 50},
			},
		},
		{
			name: "pods with pod group (DisruptionModeSingle)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").PodGroupName("pg1").Priority(20).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().Priority(50).Obj(),
			},
			domainName: "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1"), affectedNodes: sets.New("node1"), priority: 50},
				{pods: sets.New("p2"), affectedNodes: sets.New("node2"), priority: 50},
			},
		},
		{
			name: "mix of pod groups and individual pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p3").UID("p3").Node("node1").PodGroupName("pg2").Priority(20).Obj(),
				st.MakePod().Name("p4").UID("p4").Node("node2").Priority(30).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(50).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeSingle().Priority(60).Obj(),
			},
			domainName: "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1", "p2"), affectedNodes: sets.New("node1", "node2"), priority: 50},
				{pods: sets.New("p3"), affectedNodes: sets.New("node1"), priority: 60},
				{pods: sets.New("p4"), affectedNodes: sets.New("node2"), priority: 30},
			},
		},
		{
			name: "pods with pod group (DisruptionModeAll) but missing pod group state in cache falls back to local pod preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
			},
			domainName: "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2"), affectedNodes: sets.New("node2"), priority: 10},
			},
		},
		{
			name: "CPG enabled: root CompositePodGroup with DisruptionModeAll groups descendant pods of all child PodGroups",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeAll().Priority(10).Obj(),
			},
			enableCompositePodGroup: true,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1", "p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG disabled: root CompositePodGroup with DisruptionModeAll does not group descendant pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeAll().Priority(10).Obj(),
			},
			enableCompositePodGroup: false,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG enabled: root CompositePodGroup with DisruptionModeSingle does not group descendants of child PodGroups but respects individual PodGroup DisruptionModes",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeSingle().Priority(10).Obj(),
			},
			enableCompositePodGroup: true,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg1"), affectedNodes: sets.New("node2"), priority: 10},
				{pods: sets.New("p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG disabled: root CompositePodGroup with DisruptionModeSingle does not group child PodGroups and resolves PodGroup level DisruptionModes",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeSingle().Priority(10).Obj(),
			},
			enableCompositePodGroup: false,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg1"), affectedNodes: sets.New("node2"), priority: 10},
				{pods: sets.New("p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG enabled: root CompositePodGroup with DisruptionModeAll groups all descendant pods under nested CompositePodGroups with DisruptionModeAll",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p1_pg3").UID("p1_pg3").Node("node1").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p2_pg3").UID("p2_pg3").Node("node2").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p1_pg4").UID("p1_pg4").Node("node1").PodGroupName("pg4").Priority(10).Obj(),
				st.MakePod().Name("p2_pg4").UID("p2_pg4").Node("node2").PodGroupName("pg4").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg2").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg2").Priority(10).Obj(),
				"pg3": st.MakePodGroup().Name("pg3").UID("pg3").DisruptionModeAll().ParentCompositePodGroup("cpg3").Priority(10).Obj(),
				"pg4": st.MakePodGroup().Name("pg4").UID("pg4").DisruptionModeAll().ParentCompositePodGroup("cpg3").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeAll().Priority(10).Obj(),
				"cpg2": st.MakeCompositePodGroup().Name("cpg2").UID("cpg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"cpg3": st.MakeCompositePodGroup().Name("cpg3").UID("cpg3").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			enableCompositePodGroup: true,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1", "p1_pg2", "p2_pg2", "p1_pg3", "p2_pg3", "p1_pg4", "p2_pg4"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG disabled: root CompositePodGroup with DisruptionModeAll with nested CPGs does not group descendant pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p1_pg3").UID("p1_pg3").Node("node1").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p2_pg3").UID("p2_pg3").Node("node2").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p1_pg4").UID("p1_pg4").Node("node1").PodGroupName("pg4").Priority(10).Obj(),
				st.MakePod().Name("p2_pg4").UID("p2_pg4").Node("node2").PodGroupName("pg4").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg2").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg2").Priority(10).Obj(),
				"pg3": st.MakePodGroup().Name("pg3").UID("pg3").DisruptionModeAll().ParentCompositePodGroup("cpg3").Priority(10).Obj(),
				"pg4": st.MakePodGroup().Name("pg4").UID("pg4").DisruptionModeAll().ParentCompositePodGroup("cpg3").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeAll().Priority(10).Obj(),
				"cpg2": st.MakeCompositePodGroup().Name("cpg2").UID("cpg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"cpg3": st.MakeCompositePodGroup().Name("cpg3").UID("cpg3").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			enableCompositePodGroup: false,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg3", "p2_pg3"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg4", "p2_pg4"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG enabled: root CompositePodGroup with DisruptionModeSingle groups descendant pods only under nested CompositePodGroup with DisruptionModeAll",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p1_pg3").UID("p1_pg3").Node("node1").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p2_pg3").UID("p2_pg3").Node("node2").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p1_pg4").UID("p1_pg4").Node("node1").PodGroupName("pg4").Priority(10).Obj(),
				st.MakePod().Name("p2_pg4").UID("p2_pg4").Node("node2").PodGroupName("pg4").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg2").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg2").Priority(10).Obj(),
				"pg3": st.MakePodGroup().Name("pg3").UID("pg3").DisruptionModeSingle().ParentCompositePodGroup("cpg3").Priority(10).Obj(),
				"pg4": st.MakePodGroup().Name("pg4").UID("pg4").DisruptionModeSingle().ParentCompositePodGroup("cpg3").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeSingle().Priority(10).Obj(),
				"cpg2": st.MakeCompositePodGroup().Name("cpg2").UID("cpg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"cpg3": st.MakeCompositePodGroup().Name("cpg3").UID("cpg3").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			enableCompositePodGroup: true,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1", "p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg3"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg3"), affectedNodes: sets.New("node2"), priority: 10},
				{pods: sets.New("p1_pg4"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg4"), affectedNodes: sets.New("node2"), priority: 10},
			},
		},
		{
			name: "CPG disabled: root CompositePodGroup with DisruptionModeSingle with nested CPG with DisruptionModeAll does not group descendant pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p1_pg3").UID("p1_pg3").Node("node1").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p2_pg3").UID("p2_pg3").Node("node2").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p1_pg4").UID("p1_pg4").Node("node1").PodGroupName("pg4").Priority(10).Obj(),
				st.MakePod().Name("p2_pg4").UID("p2_pg4").Node("node2").PodGroupName("pg4").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg2").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg2").Priority(10).Obj(),
				"pg3": st.MakePodGroup().Name("pg3").UID("pg3").DisruptionModeSingle().ParentCompositePodGroup("cpg3").Priority(10).Obj(),
				"pg4": st.MakePodGroup().Name("pg4").UID("pg4").DisruptionModeSingle().ParentCompositePodGroup("cpg3").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeSingle().Priority(10).Obj(),
				"cpg2": st.MakeCompositePodGroup().Name("cpg2").UID("cpg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"cpg3": st.MakeCompositePodGroup().Name("cpg3").UID("cpg3").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			enableCompositePodGroup: false,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg3"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg3"), affectedNodes: sets.New("node2"), priority: 10},
				{pods: sets.New("p1_pg4"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg4"), affectedNodes: sets.New("node2"), priority: 10},
			},
		},
		{
			name: "CPG enabled: gang CompositePodGroup with DisruptionModeAll groups pods from different child PodGroups",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeAll().MinGroupCount(2).Priority(10).Obj(),
			},
			enableCompositePodGroup: true,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1", "p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG disabled: gang CompositePodGroup with DisruptionModeAll does not group pods from different child PodGroups",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeAll().MinGroupCount(2).Priority(10).Obj(),
			},
			enableCompositePodGroup: false,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG enabled: gang CompositePodGroup with DisruptionModeSingle does not group pods from different child PodGroups",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeSingle().MinGroupCount(2).Priority(10).Obj(),
			},
			enableCompositePodGroup: true,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg1"), affectedNodes: sets.New("node2"), priority: 10},
				{pods: sets.New("p1_pg2"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg2"), affectedNodes: sets.New("node2"), priority: 10},
			},
		},
		{
			name: "CPG disabled: gang CompositePodGroup with DisruptionModeSingle does not group pods from different child PodGroups",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeSingle().MinGroupCount(2).Priority(10).Obj(),
			},
			enableCompositePodGroup: false,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg1"), affectedNodes: sets.New("node2"), priority: 10},
				{pods: sets.New("p1_pg2"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg2"), affectedNodes: sets.New("node2"), priority: 10},
			},
		},
		{
			name: "CPG enabled: root CompositePodGroup with DisruptionModeAll groups all descendants as a single victim even if descendants have DisruptionModeSingle",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeAll().Priority(10).Obj(),
			},
			enableCompositePodGroup: true,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1", "p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG disabled: root CompositePodGroup with DisruptionModeAll does not group descendants, only descendant PodGroup disruption modes are resolved",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg1").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg1": st.MakeCompositePodGroup().Name("cpg1").UID("cpg1").DisruptionModeAll().Priority(10).Obj(),
			},
			enableCompositePodGroup: false,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1"), affectedNodes: sets.New("node1"), priority: 10},
				{pods: sets.New("p2_pg1"), affectedNodes: sets.New("node2"), priority: 10},
				{pods: sets.New("p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
		{
			name: "CPG enabled: missing parent CompositePodGroup falls back to available ancestor disruption mode",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1_pg1").UID("p1_pg1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2_pg1").UID("p2_pg1").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p1_pg2").UID("p1_pg2").Node("node1").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p2_pg2").UID("p2_pg2").Node("node2").PodGroupName("pg2").Priority(10).Obj(),
				st.MakePod().Name("p1_pg3").UID("p1_pg3").Node("node1").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p2_pg3").UID("p2_pg3").Node("node2").PodGroupName("pg3").Priority(10).Obj(),
				st.MakePod().Name("p1_pg4").UID("p1_pg4").Node("node1").PodGroupName("pg4").Priority(10).Obj(),
				st.MakePod().Name("p2_pg4").UID("p2_pg4").Node("node2").PodGroupName("pg4").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingv1beta1.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().ParentCompositePodGroup("cpg-mid1").Priority(10).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeAll().ParentCompositePodGroup("cpg-mid1").Priority(10).Obj(),
				"pg3": st.MakePodGroup().Name("pg3").UID("pg3").DisruptionModeAll().ParentCompositePodGroup("cpg-mid2").Priority(10).Obj(),
				"pg4": st.MakePodGroup().Name("pg4").UID("pg4").DisruptionModeAll().ParentCompositePodGroup("cpg-mid2").Priority(10).Obj(),
			},
			compositePodGroups: map[string]*schedulingv1alpha3.CompositePodGroup{
				"cpg-root": st.MakeCompositePodGroup().Name("cpg-root").UID("cpg-root").DisruptionModeAll().Priority(10).Obj(),
				"cpg-mid2": st.MakeCompositePodGroup().Name("cpg-mid2").UID("cpg-mid2").DisruptionModeAll().ParentCompositePodGroup("cpg-root").Priority(10).Obj(),
			},
			enableCompositePodGroup: true,
			domainName:              "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1_pg1", "p2_pg1"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg2", "p2_pg2"), affectedNodes: sets.New("node1", "node2"), priority: 10},
				{pods: sets.New("p1_pg3", "p2_pg3", "p1_pg4", "p2_pg4"), affectedNodes: sets.New("node1", "node2"), priority: 10},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: tt.enableCompositePodGroup,
				features.CompositePodGroup:               tt.enableCompositePodGroup,
			})

			logger, ctx := ktesting.NewTestContext(t)
			var pgs []*schedulingv1beta1.PodGroup
			for _, pg := range tt.podGroups {
				pgs = append(pgs, pg)
			}
			var cpgs []*schedulingv1alpha3.CompositePodGroup
			for _, cpg := range tt.compositePodGroups {
				cpgs = append(cpgs, cpg)
			}

			var snapshot *internalcache.Snapshot
			if tt.enableCompositePodGroup {
				snapshot = internalcache.NewTestSnapshotWithCompositePodGroups(tt.pods, tt.nodes, pgs, cpgs)
			} else {
				snapshot = internalcache.NewTestSnapshotWithPodGroups(tt.pods, tt.nodes, pgs)
			}
			cache := internalcache.New(ctx, nil, true, tt.enableCompositePodGroup)

			nodeInfos := make(map[string]fwk.NodeInfo)
			for _, node := range tt.nodes {
				ni := framework.NewNodeInfo()
				ni.SetNode(node)
				nodeInfos[node.Name] = ni
			}
			for _, p := range tt.pods {
				if ni, ok := nodeInfos[p.Spec.NodeName]; ok {
					pi, _ := framework.NewPodInfo(p)
					ni.AddPodInfo(pi)
				}
			}

			for _, node := range tt.nodes {
				cache.AddNode(logger, node)
			}

			for _, pg := range pgs {
				cache.AddPodGroup(pg)
			}
			if tt.enableCompositePodGroup {
				for _, cpg := range cpgs {
					cache.AddCompositePodGroup(logger, cpg)
				}
			}

			for _, p := range tt.pods {
				if err := cache.AddPod(logger, p); err != nil {
					t.Fatalf("Failed to add pod: %v", err)
				}
				if p.Spec.SchedulingGroup != nil && p.Spec.SchedulingGroup.PodGroupName != nil {
					cache.AddPodGroupMember(p)
				}
			}

			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}
			pgLister := &mockPodGroupLister{podGroups: tt.podGroups}
			var cpgLister fwk.CompositePodGroupLister
			if tt.enableCompositePodGroup {
				cpgLister = &mockCompositePodGroupLister{compositePodGroups: tt.compositePodGroups}
			}
			domain, err := newDomainForWorkloadPreemption(logger, snapshot, pgLister, cpgLister, tt.domainName)
			if err != nil {
				t.Fatalf("Failed to create domain: %v", err)
			}

			if domain.GetName() != tt.domainName {
				t.Errorf("expected domain name %q, got %q", tt.domainName, domain.GetName())
			}

			gotNodeNames := sets.New[string]()
			for _, ni := range domain.Nodes() {
				gotNodeNames.Insert(ni.Node().Name)
			}
			wantNodeNames := sets.New[string]()
			for _, n := range tt.nodes {
				wantNodeNames.Insert(n.Name)
			}
			if diff := cmp.Diff(wantNodeNames, gotNodeNames); diff != "" {
				t.Errorf("Nodes() mismatch (-want +got):\n%s", diff)
			}

			victims := domain.GetAllPossibleVictims()

			var gotVictims []expectedVictim
			for _, v := range victims {
				ev := expectedVictim{
					pods:          sets.New[string](),
					affectedNodes: sets.New[string](),
					priority:      v.Priority(),
				}
				for _, p := range v.Pods() {
					ev.pods.Insert(p.GetPod().Name)
				}
				for n := range v.AffectedNodes() {
					ev.affectedNodes.Insert(n)
				}
				gotVictims = append(gotVictims, ev)
			}

			sortVictims := func(vs []expectedVictim) {
				sort.Slice(vs, func(i, j int) bool {
					if vs[i].pods.Len() == 0 || vs[j].pods.Len() == 0 {
						return vs[i].pods.Len() < vs[j].pods.Len()
					}
					return sets.List(vs[i].pods)[0] < sets.List(vs[j].pods)[0]
				})
			}

			wantVictims := tt.wantVictims
			sortVictims(gotVictims)
			sortVictims(wantVictims)

			if diff := cmp.Diff(wantVictims, gotVictims, cmp.AllowUnexported(expectedVictim{})); diff != "" {
				t.Errorf("victims mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewVictim(t *testing.T) {
	now := metav1.Now()
	earlier := metav1.NewTime(now.Add(-10 * time.Minute))
	later := metav1.NewTime(now.Add(10 * time.Minute))

	p1 := st.MakePod().Name("p1").StartTime(now).Obj()
	p2 := st.MakePod().Name("p2").StartTime(later).Obj()
	p3 := st.MakePod().Name("p3").StartTime(earlier).Obj()
	p4 := st.MakePod().Name("p4").Obj() // nil start time
	p5 := st.MakePod().Name("p5").StartTime(now).PodGroupName("pg1").Obj()

	pi1, _ := framework.NewPodInfo(p1)
	pi2, _ := framework.NewPodInfo(p2)
	pi3, _ := framework.NewPodInfo(p3)
	pi4, _ := framework.NewPodInfo(p4)
	pi5, _ := framework.NewPodInfo(p5)

	tests := []struct {
		name       string
		podInfos   []fwk.PodInfo
		priority   int32
		keyType    fwk.EntityKeyType
		wantErr    bool
		wantVictim *victim
	}{
		{
			name:       "empty pods slice returns error",
			podInfos:   nil,
			priority:   10,
			keyType:    fwk.PodKeyType,
			wantErr:    true,
			wantVictim: nil,
		},
		{
			name:       "empty keyType returns error",
			podInfos:   []fwk.PodInfo{pi1},
			priority:   20,
			keyType:    "",
			wantErr:    true,
			wantVictim: nil,
		},
		{
			name:     "single pod without scheduling group",
			podInfos: []fwk.PodInfo{pi1},
			priority: 20,
			keyType:  fwk.PodKeyType,
			wantErr:  false,
			wantVictim: &victim{
				priority:          20,
				earliestStartTime: &now,
				pods:              []fwk.PodInfo{pi1},
				keyType:           fwk.PodKeyType,
			},
		},
		{
			name:     "multiple pods with mixed start times (including nil)",
			podInfos: []fwk.PodInfo{pi4, pi2, pi3},
			priority: 30,
			keyType:  fwk.PodKeyType,
			wantErr:  false,
			wantVictim: &victim{
				priority:          30,
				earliestStartTime: &earlier,
				pods:              []fwk.PodInfo{pi4, pi2, pi3},
				keyType:           fwk.PodKeyType,
			},
		},
		{
			name:     "pod with scheduling group is constructed as pod group",
			podInfos: []fwk.PodInfo{pi5},
			priority: 40,
			keyType:  fwk.PodGroupKeyType,
			wantErr:  false,
			wantVictim: &victim{
				priority:          40,
				earliestStartTime: &now,
				pods:              []fwk.PodInfo{pi5},
				keyType:           fwk.PodGroupKeyType,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotVictim, err := NewVictim(tt.podInfos, tt.priority, tt.keyType)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewVictim() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}

			if diff := cmp.Diff(tt.wantVictim, gotVictim, cmp.AllowUnexported(victim{}, framework.PodInfo{})); diff != "" {
				t.Errorf("Victim mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewDomainVictim(t *testing.T) {
	tests := []struct {
		name              string
		nodes             []*v1.Node
		pods              []*v1.Pod
		priority          int32
		wantErr           bool
		wantAffectedNodes sets.Set[string]
	}{
		{
			name: "pod on missing node returns error",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Node("missing-node").Obj(),
			},
			priority:          10,
			wantErr:           true,
			wantAffectedNodes: nil,
		},
		{
			name: "multiple pods on same and different nodes deduplicates affectedNodes",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Obj(),
				st.MakePod().Name("p2").Node("node1").Obj(), // same node
				st.MakePod().Name("p3").Node("node2").Obj(), // different node
			},
			priority:          20,
			wantErr:           false,
			wantAffectedNodes: sets.New("node1", "node2"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			snapshot := internalcache.NewSnapshot(nil, tt.nodes)
			cache := internalcache.New(ctx, nil, true, false /* CompositePodGroup */)
			for _, node := range tt.nodes {
				cache.AddNode(logger, node)
			}
			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			var podInfos []fwk.PodInfo
			for _, p := range tt.pods {
				pi, _ := framework.NewPodInfo(p)
				podInfos = append(podInfos, pi)
			}

			dv, err := newDomainVictim(snapshot, podInfos, tt.priority, fwk.PodKeyType)
			if (err != nil) != tt.wantErr {
				t.Errorf("newDomainVictim() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}

			gotNodes := sets.New[string]()
			for nName := range dv.AffectedNodes() {
				gotNodes.Insert(nName)
			}

			if diff := cmp.Diff(tt.wantAffectedNodes, gotNodes); diff != "" {
				t.Errorf("AffectedNodes() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

type testPodGroupInfo struct {
	pg   *schedulingv1beta1.PodGroup
	cpg  *schedulingv1alpha3.CompositePodGroup
	pods []*v1.Pod
}

func (t *testPodGroupInfo) GetName() string {
	if t.cpg != nil {
		return t.cpg.Name
	}
	if t.pg != nil {
		return t.pg.Name
	}
	return ""
}

func (t *testPodGroupInfo) GetNamespace() string {
	if t.cpg != nil {
		return t.cpg.Namespace
	}
	if t.pg != nil {
		return t.pg.Namespace
	}
	return ""
}

func (t *testPodGroupInfo) GetType() fwk.EntityKeyType {
	if t.cpg != nil {
		return fwk.CompositePodGroupKeyType
	}
	return fwk.PodGroupKeyType
}

func (t *testPodGroupInfo) GetKey() string {
	if t.cpg != nil {
		return t.cpg.Name
	}
	if t.pg != nil {
		return t.pg.Name
	}
	return ""
}

func (t *testPodGroupInfo) GetUnscheduledPods() []*v1.Pod {
	return t.pods
}

func (t *testPodGroupInfo) GetPodGroup() *schedulingv1beta1.PodGroup {
	return t.pg
}

func (t *testPodGroupInfo) GetCompositePodGroup() *schedulingv1alpha3.CompositePodGroup {
	return t.cpg
}

func (t *testPodGroupInfo) GetChildren() []fwk.PodGroupInfo {
	return nil
}
