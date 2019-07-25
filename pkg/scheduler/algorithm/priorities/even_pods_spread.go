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
	"context"
	"math"
	"sync/atomic"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"

	"k8s.io/klog"
)

type topologyPair struct {
	key   string
	value string
}

type topologySpreadConstraintsMap struct {
	// nodeNameToPodCounts is keyed with node name, and valued with the number of matching pods.
	nodeNameToPodCounts map[string]int64
	// topologyPairToPodCounts is keyed with topologyPair, and valued with the number of matching pods.
	topologyPairToPodCounts map[topologyPair]*int64
}

func newTopologySpreadConstraintsMap() *topologySpreadConstraintsMap {
	return &topologySpreadConstraintsMap{
		nodeNameToPodCounts:     make(map[string]int64),
		topologyPairToPodCounts: make(map[topologyPair]*int64),
	}
}

// Note: the <nodes> passed in are the "filtered" nodes which have passed Predicates.
// This function iterates <nodes> to filter out the nodes which don't have required topologyKey(s),
// and initialize two maps:
// 1) t.topologyPairToPodCounts: keyed with both eligible topology pair and node names.
// 2) t.nodeNameToPodCounts: keyed with node name, and valued with a *int64 pointer for eligible node only.
func (t *topologySpreadConstraintsMap) initialize(pod *v1.Pod, nodes []*v1.Node) {
	constraints := getSoftTopologySpreadConstraints(pod)
	for _, node := range nodes {
		if !predicates.NodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
			continue
		}
		for _, constraint := range constraints {
			pair := topologyPair{key: constraint.TopologyKey, value: node.Labels[constraint.TopologyKey]}
			if t.topologyPairToPodCounts[pair] == nil {
				t.topologyPairToPodCounts[pair] = new(int64)
			}
		}
		t.nodeNameToPodCounts[node.Name] = 0
		// For those nodes which don't have all required topologyKeys present, it's intentional to keep
		// those entries absent in nodeNameToPodCounts, so that we're able to score them to 0 afterwards.
	}
}

// CalculateEvenPodsSpreadPriority computes a score by checking through the topologySpreadConstraints
// that are with WhenUnsatisfiable=ScheduleAnyway (a.k.a soft constraint).
// The function works as below:
// 1) In all nodes, calculate the number of pods which match <pod>'s soft topology spread constraints.
// 2) Group the number calculated in 1) by topologyPair, and sum up to corresponding candidate nodes.
// 3) Finally normalize the number to 0~10. The node with the highest score is the most preferred.
// Note: Symmetry is not applicable. We only weigh how incomingPod matches existingPod.
// Whether existingPod matches incomingPod doesn't contribute to the final score.
// This is different from the Affinity API.
func CalculateEvenPodsSpreadPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	result := make(schedulerapi.HostPriorityList, len(nodes))
	// return if incoming pod doesn't have soft topology spread constraints.
	constraints := getSoftTopologySpreadConstraints(pod)
	if len(constraints) == 0 {
		return result, nil
	}

	t := newTopologySpreadConstraintsMap()
	t.initialize(pod, nodes)

	allNodeNames := make([]string, 0, len(nodeNameToInfo))
	for name := range nodeNameToInfo {
		allNodeNames = append(allNodeNames, name)
	}

	errCh := schedutil.NewErrorChannel()
	ctx, cancel := context.WithCancel(context.Background())
	processAllNode := func(i int) {
		nodeInfo := nodeNameToInfo[allNodeNames[i]]
		node := nodeInfo.Node()
		if node == nil {
			return
		}
		// (1) `node` should satisfy incoming pod's NodeSelector/NodeAffinity
		// (2) All topologyKeys need to be present in `node`
		if !predicates.PodMatchesNodeSelectorAndAffinityTerms(pod, node) ||
			!predicates.NodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
			return
		}

		for _, constraint := range constraints {
			pair := topologyPair{key: constraint.TopologyKey, value: node.Labels[constraint.TopologyKey]}
			// If current topology pair is not associated with any candidate node,
			// continue to avoid unnecessary calculation.
			if t.topologyPairToPodCounts[pair] == nil {
				continue
			}

			// <matchSum> indicates how many pods (on current node) match the <constraint>.
			matchSum := int64(0)
			for _, existingPod := range nodeInfo.Pods() {
				match, err := predicates.PodMatchesSpreadConstraint(existingPod.Labels, constraint)
				if err != nil {
					errCh.SendErrorWithCancel(err, cancel)
					return
				}
				if match {
					matchSum++
				}
			}
			atomic.AddInt64(t.topologyPairToPodCounts[pair], matchSum)
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodeNames), processAllNode)
	if err := errCh.ReceiveError(); err != nil {
		return nil, err
	}

	var minCount int64 = math.MaxInt64
	// <total> sums up the number of matching pods on each qualified topology pair
	var total int64
	for _, node := range nodes {
		if _, ok := t.nodeNameToPodCounts[node.Name]; !ok {
			continue
		}

		// For each present <pair>, current node gets a credit of <matchSum>.
		// And we add <matchSum> to <t.total> to reverse the final score later.
		for _, constraint := range constraints {
			if tpVal, ok := node.Labels[constraint.TopologyKey]; ok {
				pair := topologyPair{key: constraint.TopologyKey, value: tpVal}
				matchSum := *t.topologyPairToPodCounts[pair]
				t.nodeNameToPodCounts[node.Name] += matchSum
				total += matchSum
			}
		}
		if t.nodeNameToPodCounts[node.Name] < minCount {
			minCount = t.nodeNameToPodCounts[node.Name]
		}
	}

	// calculate final priority score for each node
	// TODO(Huang-Wei): in alpha version, we keep the formula as simple as possible.
	// current version ranks the nodes properly, but it doesn't take MaxSkew into
	// consideration, we may come up with a better formula in the future.
	maxMinDiff := total - minCount
	for i := range nodes {
		node := nodes[i]
		result[i].Host = node.Name

		// debugging purpose: print the value for each node
		// score must be pointer here, otherwise it's always 0
		if klog.V(10) {
			defer func(score *int, nodeName string) {
				klog.Infof("%v -> %v: EvenPodsSpreadPriority, Score: (%d)", pod.Name, nodeName, *score)
			}(&result[i].Score, node.Name)
		}

		if _, ok := t.nodeNameToPodCounts[node.Name]; !ok {
			result[i].Score = 0
			continue
		}
		if maxMinDiff == 0 {
			result[i].Score = schedulerapi.MaxPriority
			continue
		}
		fScore := float64(schedulerapi.MaxPriority) * (float64(total-t.nodeNameToPodCounts[node.Name]) / float64(maxMinDiff))
		result[i].Score = int(fScore)
	}

	return result, nil
}

// TODO(Huang-Wei): combine this with getHardTopologySpreadConstraints() in predicates package
func getSoftTopologySpreadConstraints(pod *v1.Pod) (constraints []v1.TopologySpreadConstraint) {
	if pod != nil {
		for _, constraint := range pod.Spec.TopologySpreadConstraints {
			if constraint.WhenUnsatisfiable == v1.ScheduleAnyway {
				constraints = append(constraints, constraint)
			}
		}
	}
	return
}
