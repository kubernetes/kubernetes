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

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
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

func TestIsDisruptionModePodGroup(t *testing.T) {
	tests := []struct {
		name       string
		pg         *schedulingapi.PodGroup
		wantModePG bool
	}{
		{
			name:       "nil pod group",
			pg:         nil,
			wantModePG: false,
		},
		{
			name:       "pod group with nil disruption mode",
			pg:         st.MakePodGroup().Name("pg1").Namespace("default").Obj(),
			wantModePG: false,
		},
		{
			name:       "pod group with DisruptionModePod",
			pg:         st.MakePodGroup().Name("pg1").Namespace("default").DisruptionMode(schedulingapi.DisruptionModePod).Obj(),
			wantModePG: false,
		},
		{
			name:       "pod group with DisruptionModePodGroup",
			pg:         st.MakePodGroup().Name("pg1").Namespace("default").DisruptionMode(schedulingapi.DisruptionModePodGroup).Obj(),
			wantModePG: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotModePG := isDisruptionModePodGroup(tt.pg); gotModePG != tt.wantModePG {
				t.Errorf("isDisruptionModePodGroup() = %v, want %v", gotModePG, tt.wantModePG)
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
			name: "pods with pod group (DisruptionModePodGroup)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").PodGroupName("pg1").Priority(10).Obj(),
			},
			podGroups: map[string]*schedulingapi.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionMode(schedulingapi.DisruptionModePodGroup).Priority(50).Obj(),
			},
			domainName: "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1", "p2"), affectedNodes: sets.New("node1", "node2"), priority: 50},
			},
		},
		{
			name: "pods with pod group (DisruptionModePod)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").PodGroupName("pg1").Priority(10).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").PodGroupName("pg1").Priority(20).Obj(),
			},
			podGroups: map[string]*schedulingapi.PodGroup{
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionMode(schedulingapi.DisruptionModePod).Priority(50).Obj(),
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
				"pg1": st.MakePodGroup().Name("pg1").UID("pg1").DisruptionMode(schedulingapi.DisruptionModePodGroup).Priority(50).Obj(),
				"pg2": st.MakePodGroup().Name("pg2").UID("pg2").DisruptionMode(schedulingapi.DisruptionModePod).Priority(60).Obj(),
			},
			domainName: "test-domain",
			wantVictims: []expectedVictim{
				{pods: sets.New("p1", "p2"), affectedNodes: sets.New("node1", "node2"), priority: 50},
				{pods: sets.New("p3"), affectedNodes: sets.New("node1"), priority: 60},
				{pods: sets.New("p4"), affectedNodes: sets.New("node2"), priority: 30},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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

			var domainNodes []fwk.NodeInfo
			for _, node := range tt.nodes {
				if ni, ok := nodeInfos[node.Name]; ok {
					domainNodes = append(domainNodes, ni)
				}
			}

			pgLister := &mockPodGroupLister{podGroups: tt.podGroups}
			domain := newDomainForWorkloadPreemption(domainNodes, pgLister, tt.domainName)

			if domain.GetName() != tt.domainName {
				t.Errorf("expected domain name %q, got %q", tt.domainName, domain.GetName())
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
