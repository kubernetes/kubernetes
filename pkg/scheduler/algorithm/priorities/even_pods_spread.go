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
	"sync/atomic"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
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
	// podCounts is keyed with node name, and valued with the number of matching pods.
	podCounts map[string]*int64
	// total number of matching pods on each qualified <topologyKey:value> pair
	total int64
	// topologyPairToNodeNames store the mapping from potential <topologyKey:value>
	// pair to node names
	topologyPairToNodeNames map[topologyPair][]string
}

func newTopologySpreadConstraintsMap(len int) *topologySpreadConstraintsMap {
	return &topologySpreadConstraintsMap{
		podCounts:               make(map[string]*int64, len),
		topologyPairToNodeNames: make(map[topologyPair][]string),
	}
}

func (t *topologySpreadConstraintsMap) initialize(pod *v1.Pod, nodes []*v1.Node) {
	constraints := getSoftTopologySpreadConstraints(pod)
	for _, node := range nodes {
		match := true
		var pairs []topologyPair
		for _, constraint := range constraints {
			tpKey := constraint.TopologyKey
			if _, ok := node.Labels[tpKey]; !ok {
				// Current node isn't qualified for the soft constraints,
				// so break here and the node will hold default value (nil).
				match = false
				break
			}
			pairs = append(pairs, topologyPair{key: tpKey, value: node.Labels[tpKey]})
		}
		if match {
			for _, pair := range pairs {
				t.topologyPairToNodeNames[pair] = append(t.topologyPairToNodeNames[pair], node.Name)
			}
			t.podCounts[node.Name] = new(int64)
		}
		// For those nodes which don't have all required topologyKeys present, it's intentional to
		// leave podCounts[nodeName] as nil, so that we're able to score these nodes to 0 afterwards.
	}
}

// CalculateEvenPodsSpreadPriority computes a score by checking through the topologySpreadConstraints
// that are with WhenUnsatisfiable=ScheduleAnyway (a.k.a soft constraint).
// The function works as below:
// 1) In all nodes, calculate the number of pods which match <pod>'s soft topology spread constraints.
// 2) Sum up the number to each node in <nodes> which has corresponding topologyPair present.
// 3) Finally normalize the number to 0~10. The node with the highest score is the most preferred.
// Note: Symmetry is not applicable. We only weigh how incomingPod matches existingPod.
// Whether existingPod matches incomingPod doesn't contribute to the final score.
// This is different with the Affinity API.
func CalculateEvenPodsSpreadPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	result := make(schedulerapi.HostPriorityList, len(nodes))
	// return if incoming pod doesn't have soft topology spread constraints.
	constraints := getSoftTopologySpreadConstraints(pod)
	if len(constraints) == 0 {
		return result, nil
	}

	t := newTopologySpreadConstraintsMap(len(nodes))
	t.initialize(pod, nodes)

	allNodeNames := make([]string, 0, len(nodeNameToInfo))
	for name := range nodeNameToInfo {
		allNodeNames = append(allNodeNames, name)
	}

	errCh := schedutil.NewErrorChannel()
	ctx, cancel := context.WithCancel(context.Background())
	processNode := func(i int) {
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
		// It's enough to use topologyKey as the "key" of the map.
		matchCount := make(map[string]int64)
		for _, existingPod := range nodeInfo.Pods() {
			podLabelSet := labels.Set(existingPod.Labels)
			// Matching on constraints is calculated independently.
			for _, constraint := range constraints {
				match, err := predicates.PodMatchesSpreadConstraint(podLabelSet, constraint)
				if err != nil {
					errCh.SendErrorWithCancel(err, cancel)
					return
				}
				if match {
					matchCount[constraint.TopologyKey]++
				}
			}
		}
		// Keys in t.podCounts have been ensured to contain "filtered" nodes only.
		for _, constraint := range constraints {
			tpKey := constraint.TopologyKey
			pair := topologyPair{key: tpKey, value: node.Labels[tpKey]}
			// For each <pair>, all matched nodes get the credit of summed matchCount.
			// And we add matchCount to <t.total> to reverse the final score later.
			for _, nodeName := range t.topologyPairToNodeNames[pair] {
				atomic.AddInt64(t.podCounts[nodeName], matchCount[tpKey])
				atomic.AddInt64(&t.total, matchCount[tpKey])
			}
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodeNames), processNode)
	if err := errCh.ReceiveError(); err != nil {
		return nil, err
	}

	var maxCount, minCount int64
	for _, node := range nodes {
		if t.podCounts[node.Name] == nil {
			continue
		}
		// reverse
		count := t.total - *t.podCounts[node.Name]
		if count > maxCount {
			maxCount = count
		} else if count < minCount {
			minCount = count
		}
		t.podCounts[node.Name] = &count
	}
	// calculate final priority score for each node
	// TODO(Huang-Wei): in alpha version, we keep the formula as simple as possible.
	// current version ranks the nodes properly, but it doesn't take MaxSkew into
	// consideration, we may come up with a better formula in the future.
	maxMinDiff := maxCount - minCount
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

		if t.podCounts[node.Name] == nil {
			result[i].Score = 0
			continue
		}
		if maxMinDiff == 0 {
			result[i].Score = schedulerapi.MaxPriority
			continue
		}
		fScore := float64(schedulerapi.MaxPriority) * (float64(*t.podCounts[node.Name]-minCount) / float64(maxMinDiff))
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
