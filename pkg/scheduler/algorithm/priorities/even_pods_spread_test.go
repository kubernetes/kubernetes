/*
Copyright 2019 The Kubernetes Authors.

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

package priorities

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func Test_topologySpreadConstraintsMap_initialize(t *testing.T) {
	tests := []struct {
		name                string
		pod                 *v1.Pod
		nodes               []*v1.Node
		wantNodeNameMap     map[string]int64
		wantTopologyPairMap map[topologyPair]*int64
	}{
		{
			name: "normal case",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			wantNodeNameMap: map[string]int64{
				"node-a": 0,
				"node-b": 0,
				"node-x": 0,
			},
			wantTopologyPairMap: map[topologyPair]*int64{
				{key: "zone", value: "zone1"}:  new(int64),
				{key: "zone", value: "zone2"}:  new(int64),
				{key: "node", value: "node-a"}: new(int64),
				{key: "node", value: "node-b"}: new(int64),
				{key: "node", value: "node-x"}: new(int64),
			},
		},
		{
			name: "node-x doesn't have label zone",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("node", "node-x").Obj(),
			},
			wantNodeNameMap: map[string]int64{
				"node-a": 0,
				"node-b": 0,
			},
			wantTopologyPairMap: map[topologyPair]*int64{
				{key: "zone", value: "zone1"}:  new(int64),
				{key: "node", value: "node-a"}: new(int64),
				{key: "node", value: "node-b"}: new(int64),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tMap := newTopologySpreadConstraintsMap()
			tMap.initialize(tt.pod, tt.nodes)
			if !reflect.DeepEqual(tMap.nodeNameToPodCounts, tt.wantNodeNameMap) {
				t.Errorf("initilize().nodeNameToPodCounts = %#v, want %#v", tMap.nodeNameToPodCounts, tt.wantNodeNameMap)
			}
			if !reflect.DeepEqual(tMap.topologyPairToPodCounts, tt.wantTopologyPairMap) {
				t.Errorf("initilize().topologyPairToPodCounts = %#v, want %#v", tMap.topologyPairToPodCounts, tt.wantTopologyPairMap)
			}
		})
	}
}

func TestCalculateEvenPodsSpreadPriority(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		existingPods []*v1.Pod
		nodes        []*v1.Node
		failedNodes  []*v1.Node // nodes + failedNodes = all nodes
		want         schedulerapi.HostPriorityList
	}{
		// Explanation on the Legend:
		// a) X/Y means there are X matching pods on node1 and Y on node2, both nodes are candidates
		//   (i.e. they have passed all predicates)
		// b) X/~Y~ means there are X matching pods on node1 and Y on node2, but node Y is NOT a candidate
		// c) X/?Y? means there are X matching pods on node1 and Y on node2, both nodes are candidates
		//    but node2 either i) doesn't have all required topologyKeys present, or ii) doesn't match
		//    incoming pod's nodeSelector/nodeAffinity
		{
			// if there is only one candidate node, it should be scored to 10
			name: "one constraint on node, no existing pods",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("node", "node-b").Obj(),
			},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 10},
				{Host: "node-b", Score: 10},
			},
		},
		{
			// if there is only one candidate node, it should be scored to 10
			name: "one constraint on node, only one node is candidate",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("node", "node-a").Obj(),
			},
			failedNodes: []*v1.Node{
				st.MakeNode().Name("node-b").Label("node", "node-b").Obj(),
			},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 10},
			},
		},
		{
			name: "one constraint on node, all nodes have the same number of matching pods",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("node", "node-b").Obj(),
			},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 10},
				{Host: "node-b", Score: 10},
			},
		},
		{
			// matching pods spread as 2/1/0/3, total = 6
			// after reversing, it's 4/5/6/3
			// so scores = 40/6, 50/6, 60/6, 30/6
			name: "one constraint on node, all 4 nodes are candidates",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-d1").Node("node-d").Label("foo", "").Obj(),
				st.MakePod().Name("p-d2").Node("node-d").Label("foo", "").Obj(),
				st.MakePod().Name("p-d3").Node("node-d").Label("foo", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-c").Label("node", "node-c").Obj(),
				st.MakeNode().Name("node-d").Label("node", "node-d").Obj(),
			},
			failedNodes: []*v1.Node{},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 6},
				{Host: "node-b", Score: 8},
				{Host: "node-c", Score: 10},
				{Host: "node-d", Score: 5},
			},
		},
		{
			// matching pods spread as 4/2/1/~3~, total = 4+2+1 = 7 (as node4 is not a candidate)
			// after reversing, it's 3/5/6
			// so scores = 30/6, 50/6, 60/6
			name: "one constraint on node, 3 out of 4 nodes are candidates",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a3").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a4").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-b2").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("node", "node-x").Obj(),
			},
			failedNodes: []*v1.Node{
				st.MakeNode().Name("node-y").Label("node", "node-y").Obj(),
			},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 5},
				{Host: "node-b", Score: 8},
				{Host: "node-x", Score: 10},
			},
		},
		{
			// matching pods spread as 4/?2?/1/~3~, total = 4+?+1 = 5 (as node2 is problematic)
			// after reversing, it's 1/?/4
			// so scores = 10/4, 0, 40/4
			name: "one constraint on node, 3 out of 4 nodes are candidates",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a3").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a4").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-b2").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("n", "node-b").Obj(), // label `n` doesn't match topologyKey
				st.MakeNode().Name("node-x").Label("node", "node-x").Obj(),
			},
			failedNodes: []*v1.Node{
				st.MakeNode().Name("node-y").Label("node", "node-y").Obj(),
			},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 2},
				{Host: "node-b", Score: 0},
				{Host: "node-x", Score: 10},
			},
		},
		{
			// matching pods spread as 4/2/1/~3~, total = 6+6+4 = 16 (as topologyKey is zone instead of node)
			// after reversing, it's 10/10/12
			// so scores = 100/12, 100/12, 120/12
			name: "one constraint on zone, 3 out of 4 nodes are candidates",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a3").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a4").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-b2").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			failedNodes: []*v1.Node{
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 8},
				{Host: "node-b", Score: 8},
				{Host: "node-x", Score: 10},
			},
		},
		{
			// matching pods spread as 2/~1~/2/~4~, total = 2+3 + 2+6 = 13 (zone and node should be both summed up)
			// after reversing, it's 8/5
			// so scores = 80/8, 50/8
			name: "two constraints on zone and node, 2 out of 4 nodes are candidates",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-a2").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-x2").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y3").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y4").Node("node-y").Label("foo", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			failedNodes: []*v1.Node{
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 10},
				{Host: "node-x", Score: 6},
			},
		},
		{
			// If constraints hold different labelSelectors, it's a little complex.
			// +----------------------+------------------------+
			// |         zone1        |          zone2         |
			// +----------------------+------------------------+
			// | node-a |    node-b   | node-x |     node-y    |
			// +--------+-------------+--------+---------------+
			// | P{foo} | P{foo, bar} |        | P{foo} P{bar} |
			// +--------+-------------+--------+---------------+
			// For the first constraint (zone): the matching pods spread as 2/2/1/1
			// For the second constraint (node): the matching pods spread as 0/1/0/1
			// sum them up gets: 2/3/1/2, and total number is 8.
			// after reversing, it's 6/5/7/6
			// so scores = 60/7, 50/7, 70/7, 60/7
			name: "two constraints on zone and node, with different labelSelectors",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("bar", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			failedNodes: []*v1.Node{},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 8},
				{Host: "node-b", Score: 7},
				{Host: "node-x", Score: 10},
				{Host: "node-y", Score: 8},
			},
		},
		{
			// For the first constraint (zone): the matching pods spread as 0/0/2/2
			// For the second constraint (node): the matching pods spread as 0/1/0/1
			// sum them up gets: 0/1/2/3, and total number is 6.
			// after reversing, it's 6/5/4/3.
			// so scores = 60/6, 50/6, 40/6, 30/6
			name: "two constraints on zone and node, with different labelSelectors, some nodes have 0 pods",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-b1").Node("node-b").Label("bar", "").Obj(),
				st.MakePod().Name("p-x1").Node("node-x").Label("foo", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Label("bar", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			failedNodes: []*v1.Node{},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 10},
				{Host: "node-b", Score: 8},
				{Host: "node-x", Score: 6},
				{Host: "node-y", Score: 5},
			},
		},
		{
			// For the first constraint (zone): the matching pods spread as 2/2/1/~1~
			// For the second constraint (node): the matching pods spread as 0/1/0/~1~
			// sum them up gets: 2/3/1, and total number is 6.
			// after reversing, it's 4/3/5
			// so scores = 40/5, 30/5, 50/5
			name: "two constraints on zone and node, with different labelSelectors, 3 out of 4 nodes are candidates",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p-a1").Node("node-a").Label("foo", "").Obj(),
				st.MakePod().Name("p-b1").Node("node-b").Label("foo", "").Label("bar", "").Obj(),
				st.MakePod().Name("p-y1").Node("node-y").Label("foo", "").Obj(),
				st.MakePod().Name("p-y2").Node("node-y").Label("bar", "").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-a").Label("zone", "zone1").Label("node", "node-a").Obj(),
				st.MakeNode().Name("node-b").Label("zone", "zone1").Label("node", "node-b").Obj(),
				st.MakeNode().Name("node-x").Label("zone", "zone2").Label("node", "node-x").Obj(),
			},
			failedNodes: []*v1.Node{
				st.MakeNode().Name("node-y").Label("zone", "zone2").Label("node", "node-y").Obj(),
			},
			want: []schedulerapi.HostPriority{
				{Host: "node-a", Score: 8},
				{Host: "node-b", Score: 6},
				{Host: "node-x", Score: 10},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allNodes := append([]*v1.Node{}, tt.nodes...)
			allNodes = append(allNodes, tt.failedNodes...)
			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(tt.existingPods, allNodes)

			got, _ := CalculateEvenPodsSpreadPriority(tt.pod, nodeNameToInfo, tt.nodes)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CalculateEvenPodsSpreadPriority() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func makeNodesAndPods(pod *v1.Pod, existingPodsNum, allNodesNum, filteredNodesNum int) (existingPods []*v1.Pod, allNodes []*v1.Node, filteredNodes []*v1.Node) {
	var topologyKeys []string
	var labels []string
	// regions := 3
	zones := 10
	for _, c := range pod.Spec.TopologySpreadConstraints {
		topologyKeys = append(topologyKeys, c.TopologyKey)
		labels = append(labels, c.LabelSelector.MatchExpressions[0].Key)
	}
	// build nodes
	for i := 0; i < allNodesNum; i++ {
		nodeWrapper := st.MakeNode().Name(fmt.Sprintf("node%d", i))
		for _, tpKey := range topologyKeys {
			if tpKey == "zone" {
				nodeWrapper = nodeWrapper.Label("zone", fmt.Sprintf("zone%d", i%zones))
			} else if tpKey == "node" {
				nodeWrapper = nodeWrapper.Label("node", fmt.Sprintf("node%d", i))
			}
		}
		node := nodeWrapper.Obj()
		allNodes = append(allNodes, node)
		if len(filteredNodes) < filteredNodesNum {
			filteredNodes = append(filteredNodes, node)
		}
	}
	// build pods
	for i := 0; i < existingPodsNum; i++ {
		podWrapper := st.MakePod().Name(fmt.Sprintf("pod%d", i)).Node(fmt.Sprintf("node%d", i%allNodesNum))
		// apply labels[0], labels[0,1], ..., labels[all] to each pod in turn
		for _, label := range labels[:i%len(labels)+1] {
			podWrapper = podWrapper.Label(label, "")
		}
		existingPods = append(existingPods, podWrapper.Obj())
	}
	return
}

func BenchmarkTestCalculateEvenPodsSpreadPriority(b *testing.B) {
	tests := []struct {
		name             string
		pod              *v1.Pod
		existingPodsNum  int
		allNodesNum      int
		filteredNodesNum int
	}{
		{
			name: "1000nodes/single-constraint-zone",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPodsNum:  10000,
			allNodesNum:      1000,
			filteredNodesNum: 500,
		},
		{
			name: "1000nodes/single-constraint-node",
			pod: st.MakePod().Name("p").Label("foo", "").
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPodsNum:  10000,
			allNodesNum:      1000,
			filteredNodesNum: 500,
		},
		{
			name: "1000nodes/two-constraints-zone-node",
			pod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("bar").Obj()).
				Obj(),
			existingPodsNum:  10000,
			allNodesNum:      1000,
			filteredNodesNum: 500,
		},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			existingPods, allNodes, filteredNodes := makeNodesAndPods(tt.pod, tt.existingPodsNum, tt.allNodesNum, tt.filteredNodesNum)
			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(existingPods, allNodes)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				CalculateEvenPodsSpreadPriority(tt.pod, nodeNameToInfo, filteredNodes)
			}
		})
	}
}

var (
	hardSpread = v1.DoNotSchedule
	softSpread = v1.ScheduleAnyway
)
