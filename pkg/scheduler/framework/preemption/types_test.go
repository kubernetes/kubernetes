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
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestGetPodGroup(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		podGroups    map[string]*schedulingapi.PodGroup
		wantPodGroup *schedulingapi.PodGroup
	}{
		{
			name:         "pod without scheduling group",
			pod:          st.MakePod().Name("p1").Namespace("default").Obj(),
			wantPodGroup: nil,
		},
		{
			name:         "pod group not found",
			pod:          st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups:    map[string]*schedulingapi.PodGroup{},
			wantPodGroup: nil,
		},
		{
			name: "pod group found",
			pod:  st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups: map[string]*schedulingapi.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").Namespace("default").Obj(),
			},
			wantPodGroup: st.MakePodGroup().Name("pg1").Namespace("default").Obj(),
		},
		{
			name:         "pod group found in pod spec but nil podGroupSnapshot (simulates disabled GenericWorkload)",
			pod:          st.MakePod().Name("p1").Namespace("default").PodGroupName("pg1").Obj(),
			podGroups:    nil, // nil snapshot when feature gate is disabled
			wantPodGroup: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pgLister := &mockPodGroupLister{podGroups: tt.podGroups}
			podGroup := getPodGroup(tt.pod, pgLister)
			if diff := cmp.Diff(tt.wantPodGroup, podGroup); diff != "" {
				t.Errorf("getPodGroup() mismatch (-want +got):\n%s", diff)
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
		pg                             *schedulingapi.PodGroup
		pods                           []*v1.Pod
		enablePodGroupPreemptionPolicy bool
		wantPolicy                     schedulingapi.PreemptionPolicy
	}{
		{
			name:                           "PreemptionPolicy PreemptNever is resolved from PodGroup, ignoring policy in pod, with PodGroupPreemptionPolicy enabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingapi.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     schedulingapi.PreemptNever,
		},
		{
			name:                           "PreemptionPolicy PreemptLowerPriority is resolved from PodGroup, ignoring different policies in pods, with PodGroupPreemptionPolicy enabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingapi.PreemptLowerPriority).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod, noPolicyPod},
			enablePodGroupPreemptionPolicy: true,
			wantPolicy:                     schedulingapi.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy is resolved from pods with PodGroupPreemptionPolicy disabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingapi.PreemptNever).Obj(),
			pods:                           []*v1.Pod{preemptLowerPriorityPod},
			enablePodGroupPreemptionPolicy: false,
			wantPolicy:                     schedulingapi.PreemptLowerPriority,
		},
		{
			name:                           "PreemptionPolicy is resolved from pods when multiple pods have different policies, with PodGroupPreemptionPolicy disabled",
			pg:                             st.MakePodGroup().Name("pg").PreemptionPolicy(schedulingapi.PreemptLowerPriority).Obj(),
			pods:                           []*v1.Pod{preemptNeverPod, preemptLowerPriorityPod, noPolicyPod},
			enablePodGroupPreemptionPolicy: false,
			wantPolicy:                     schedulingapi.PreemptNever,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preemptor := newPodGroupPreemptor(tt.pg, tt.pods, tt.enablePodGroupPreemptionPolicy)
			if preemptor.preemptionPolicy != tt.wantPolicy {
				t.Errorf("expected preemption policy %q, got %q", tt.wantPolicy, preemptor.preemptionPolicy)
			}
		})
	}
}

func TestIsDisruptionModeAll(t *testing.T) {
	tests := []struct {
		name        string
		pg          *schedulingapi.PodGroup
		wantModeAll bool
	}{
		{
			name:        "nil pod group",
			pg:          nil,
			wantModeAll: false,
		},
		{
			name:        "pod group with nil disruption mode",
			pg:          st.MakePodGroup().Name("pg1").Namespace("default").Obj(),
			wantModeAll: false,
		},
		{
			name:        "pod group with DisruptionModeSingle",
			pg:          st.MakePodGroup().Name("pg1").Namespace("default").DisruptionModeSingle().Obj(),
			wantModeAll: false,
		},
		{
			name:        "pod group with DisruptionModeAll",
			pg:          st.MakePodGroup().Name("pg1").Namespace("default").DisruptionModeAll().Obj(),
			wantModeAll: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotModeAll := isDisruptionModeAll(tt.pg); gotModeAll != tt.wantModeAll {
				t.Errorf("isDisruptionModeAll() = %v, want %v", gotModeAll, tt.wantModeAll)
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
		name        string
		nodes       []*v1.Node
		pods        []*v1.Pod
		podGroups   map[string]*schedulingapi.PodGroup
		domainName  string
		wantVictims []expectedVictim
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
			podGroups: map[string]*schedulingapi.PodGroup{
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
			podGroups: map[string]*schedulingapi.PodGroup{
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
			podGroups: map[string]*schedulingapi.PodGroup{
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.GenericWorkload, true)

			logger, ctx := ktesting.NewTestContext(t)
			snapshot := internalcache.NewSnapshot(tt.pods, tt.nodes)
			cache := internalcache.New(ctx, nil, true)

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
			domain, err := newDomainForWorkloadPreemption(snapshot, pgLister, tt.domainName)
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
			sortVictims(gotVictims)
			sortVictims(tt.wantVictims)

			if diff := cmp.Diff(tt.wantVictims, gotVictims, cmp.AllowUnexported(expectedVictim{})); diff != "" {
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
		wantErr    bool
		wantVictim *victim
	}{
		{
			name:       "empty pods slice returns error",
			podInfos:   nil,
			priority:   10,
			wantErr:    true,
			wantVictim: nil,
		},
		{
			name:     "single pod without scheduling group",
			podInfos: []fwk.PodInfo{pi1},
			priority: 20,
			wantErr:  false,
			wantVictim: &victim{
				priority:          20,
				earliestStartTime: &now,
				pods:              []fwk.PodInfo{pi1},
			},
		},
		{
			name:     "multiple pods with mixed start times (including nil)",
			podInfos: []fwk.PodInfo{pi4, pi2, pi3},
			priority: 30,
			wantErr:  false,
			wantVictim: &victim{
				priority:          30,
				earliestStartTime: &earlier,
				pods:              []fwk.PodInfo{pi4, pi2, pi3},
			},
		},
		{
			name:     "pod with scheduling group is identified as pod group",
			podInfos: []fwk.PodInfo{pi5},
			priority: 40,
			wantErr:  false,
			wantVictim: &victim{
				priority:          40,
				earliestStartTime: &now,
				pods:              []fwk.PodInfo{pi5},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotVictim, err := NewVictim(tt.podInfos, tt.priority)
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
			cache := internalcache.New(ctx, nil, true)
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

			dv, err := newDomainVictim(snapshot, podInfos, tt.priority)
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
