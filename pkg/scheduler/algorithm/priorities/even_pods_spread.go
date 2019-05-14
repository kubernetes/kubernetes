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
	"sync"
	"sync/atomic"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"

	"k8s.io/klog"
)

type topologyPair struct {
	key   string
	value string
}

type topologySpreadConstrantsMap struct {
	// The first error that we faced.
	firstError error
	sync.Mutex

	// counts store the mapping from node name to so-far computed score of
	// the node.
	counts map[string]*int64
	// total number of matching pods on each qualified <topologyKey:value> pair
	total int64
	// topologyPairToNodeNames store the mapping from potential <topologyKey:value>
	// pair to node names
	topologyPairToNodeNames map[topologyPair][]string
}

func newTopologySpreadConstrantsMap(len int) *topologySpreadConstrantsMap {
	return &topologySpreadConstrantsMap{
		counts:                  make(map[string]*int64, len),
		topologyPairToNodeNames: make(map[topologyPair][]string),
	}
}

func (t *topologySpreadConstrantsMap) setError(err error) {
	t.Lock()
	if t.firstError == nil {
		t.firstError = err
	}
	t.Unlock()
}

func (t *topologySpreadConstrantsMap) initialize(pod *v1.Pod, nodes []*v1.Node) {
	constraints := getSoftTopologySpreadConstraints(pod)
	for _, node := range nodes {
		labelSet := labels.Set(node.Labels)
		allMatch := true
		var pairs []topologyPair
		for _, constraint := range constraints {
			tpKey := constraint.TopologyKey
			if !labelSet.Has(tpKey) {
				allMatch = false
				break
			}
			pairs = append(pairs, topologyPair{key: tpKey, value: node.Labels[tpKey]})
		}
		if allMatch {
			for _, pair := range pairs {
				t.topologyPairToNodeNames[pair] = append(t.topologyPairToNodeNames[pair], node.Name)
			}
			t.counts[node.Name] = new(int64)
		}
		// for those nodes which don't have all required topologyKeys present, it's intentional to
		// leave counts[nodeName] as nil, so that we're able to score these nodes to 0 afterwards
	}
}

// CalculateEvenPodsSpreadPriority computes a score by checking through the topologySpreadConstraints
// that are with WhenUnsatifiable=ScheduleAnyway (a.k.a soft constraint).
// For each node (not only "filtered" nodes by Predicates), it adds the number of matching pods
// (all topologySpreadConstraints must be satified) as a "weight" to any "filtered" node
// which has the <topologyKey:value> pair present.
// Then the sumed "weight" are normalized to 0~10, and the node(s) with the highest score are
// the most preferred.
// Symmetry is not considered.
func CalculateEvenPodsSpreadPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	nodesLen := len(nodes)
	result := make(schedulerapi.HostPriorityList, nodesLen)
	// if incoming pod doesn't have soft topology spread constraints, return
	constraints := getSoftTopologySpreadConstraints(pod)
	if len(constraints) == 0 {
		return result, nil
	}

	t := newTopologySpreadConstrantsMap(len(nodes))
	t.initialize(pod, nodes)

	allNodeNames := make([]string, 0, len(nodeNameToInfo))
	for name := range nodeNameToInfo {
		allNodeNames = append(allNodeNames, name)
	}

	ctx, cancel := context.WithCancel(context.Background())
	processNode := func(i int) {
		nodeInfo := nodeNameToInfo[allNodeNames[i]]
		if node := nodeInfo.Node(); node != nil {
			// (1) `node` should satisfy incoming pod's NodeSelector/NodeAffinity
			// (2) All topologyKeys need to be present in `node`
			if !predicates.PodMatchesNodeSelectorAndAffinityTerms(pod, node) ||
				!predicates.NodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
				return
			}
			matchCount := 0
			for _, existingPod := range nodeInfo.Pods() {
				match, err := predicates.PodMatchesAllSpreadConstraints(existingPod, pod.Namespace, constraints)
				if err != nil {
					t.setError(err)
					cancel()
					return
				}
				if match {
					matchCount++
				}
			}
			// add matchCount up to EACH node which is at least in one topology domain
			// with current node
			for _, constraint := range constraints {
				tpKey := constraint.TopologyKey
				pair := topologyPair{key: tpKey, value: node.Labels[tpKey]}
				for _, nodeName := range t.topologyPairToNodeNames[pair] {
					atomic.AddInt64(t.counts[nodeName], int64(matchCount))
					atomic.AddInt64(&t.total, int64(matchCount))
				}
			}
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodeNames), processNode)
	if t.firstError != nil {
		return nil, t.firstError
	}

	var maxCount, minCount int64
	for _, node := range nodes {
		if t.counts[node.Name] == nil {
			continue
		}
		// reverse
		count := t.total - *t.counts[node.Name]
		if count > maxCount {
			maxCount = count
		} else if count < minCount {
			minCount = count
		}
		t.counts[node.Name] = &count
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

		if t.counts[node.Name] == nil {
			result[i].Score = 0
			continue
		}
		if maxMinDiff == 0 {
			result[i].Score = schedulerapi.MaxPriority
			continue
		}
		fScore := float64(schedulerapi.MaxPriority) * (float64(*t.counts[node.Name]-minCount) / float64(maxMinDiff))
		// need to reverse b/c the more matching pods it has, the less qualified it is
		// result[i].Score = schedulerapi.MaxPriority - int(fScore)
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
