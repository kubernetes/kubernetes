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
	"fmt"
	"math"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"

	"k8s.io/klog"
)

type topologyPair struct {
	key   string
	value string
}

type podTopologySpreadMap struct {
	constraints []topologySpreadConstraint
	// nodeNameSet is a string set holding all node names which have all constraints[*].topologyKey present.
	nodeNameSet map[string]struct{}
	// topologyPairToPodCounts is keyed with topologyPair, and valued with the number of matching pods.
	topologyPairToPodCounts map[topologyPair]*int64
}

// topologySpreadConstraint is an internal version for a soft (ScheduleAnyway
// unsatisfiable constraint action) v1.TopologySpreadConstraint and where the
// selector is parsed.
type topologySpreadConstraint struct {
	topologyKey string
	selector    labels.Selector
}

func newTopologySpreadConstraintsMap() *podTopologySpreadMap {
	return &podTopologySpreadMap{
		nodeNameSet:             make(map[string]struct{}),
		topologyPairToPodCounts: make(map[topologyPair]*int64),
	}
}

// buildPodTopologySpreadMap prepares necessary data (podTopologySpreadMap) for incoming pod on the filteredNodes.
// Later Priority function will use 'podTopologySpreadMap' to perform the Scoring calculations.
func buildPodTopologySpreadMap(pod *v1.Pod, filteredNodes []*v1.Node, allNodes []*schedulernodeinfo.NodeInfo) (*podTopologySpreadMap, error) {
	if len(filteredNodes) == 0 || len(allNodes) == 0 {
		return nil, nil
	}

	// initialize podTopologySpreadMap which will be used in Score plugin.
	m := newTopologySpreadConstraintsMap()
	err := m.initialize(pod, filteredNodes)
	if err != nil {
		return nil, err
	}
	// return if incoming pod doesn't have soft topology spread constraints.
	if m.constraints == nil {
		return nil, nil
	}

	processAllNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			return
		}
		// (1) `node` should satisfy incoming pod's NodeSelector/NodeAffinity
		// (2) All topologyKeys need to be present in `node`
		if !predicates.PodMatchesNodeSelectorAndAffinityTerms(pod, node) ||
			!nodeLabelsMatchSpreadConstraints(node.Labels, m.constraints) {
			return
		}

		for _, c := range m.constraints {
			pair := topologyPair{key: c.topologyKey, value: node.Labels[c.topologyKey]}
			// If current topology pair is not associated with any candidate node,
			// continue to avoid unnecessary calculation.
			if m.topologyPairToPodCounts[pair] == nil {
				continue
			}

			// <matchSum> indicates how many pods (on current node) match the <constraint>.
			matchSum := int64(0)
			for _, existingPod := range nodeInfo.Pods() {
				if c.selector.Matches(labels.Set(existingPod.Labels)) {
					matchSum++
				}
			}
			atomic.AddInt64(m.topologyPairToPodCounts[pair], matchSum)
		}
	}
	workqueue.ParallelizeUntil(context.Background(), 16, len(allNodes), processAllNode)

	return m, nil
}

// initialize iterates "filteredNodes" to filter out the nodes which don't have required topologyKey(s),
// and initialize two maps:
// 1) m.topologyPairToPodCounts: keyed with both eligible topology pair and node names.
// 2) m.nodeNameSet: keyed with node name, and valued with a *int64 pointer for eligible node only.
func (m *podTopologySpreadMap) initialize(pod *v1.Pod, filteredNodes []*v1.Node) error {
	constraints, err := filterSoftTopologySpreadConstraints(pod.Spec.TopologySpreadConstraints)
	if err != nil {
		return err
	}
	if constraints == nil {
		return nil
	}
	m.constraints = constraints
	for _, node := range filteredNodes {
		if !nodeLabelsMatchSpreadConstraints(node.Labels, m.constraints) {
			continue
		}
		for _, constraint := range m.constraints {
			pair := topologyPair{key: constraint.topologyKey, value: node.Labels[constraint.topologyKey]}
			if m.topologyPairToPodCounts[pair] == nil {
				m.topologyPairToPodCounts[pair] = new(int64)
			}
		}
		m.nodeNameSet[node.Name] = struct{}{}
		// For those nodes which don't have all required topologyKeys present, it's intentional to leave
		// their entries absent in nodeNameSet, so that we're able to score them to 0 afterwards.
	}
	return nil
}

// CalculateEvenPodsSpreadPriorityMap calculate the number of matching pods on the passed-in "node",
// and return the number as Score.
func CalculateEvenPodsSpreadPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulernodeinfo.NodeInfo) (framework.NodeScore, error) {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NodeScore{}, fmt.Errorf("node not found")
	}

	var m *podTopologySpreadMap
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		m = priorityMeta.podTopologySpreadMap
	}
	if m == nil {
		return framework.NodeScore{Name: node.Name, Score: 0}, nil
	}

	// no need to continue if the node is not qualified.
	if _, ok := m.nodeNameSet[node.Name]; !ok {
		return framework.NodeScore{Name: node.Name, Score: 0}, nil
	}

	// For each present <pair>, current node gets a credit of <matchSum>.
	// And we sum up <matchSum> and return it as this node's score.
	var score int64
	for _, c := range m.constraints {
		if tpVal, ok := node.Labels[c.topologyKey]; ok {
			pair := topologyPair{key: c.topologyKey, value: tpVal}
			matchSum := *m.topologyPairToPodCounts[pair]
			score += matchSum
		}
	}
	return framework.NodeScore{Name: node.Name, Score: score}, nil
}

// CalculateEvenPodsSpreadPriorityReduce normalizes the score for each filteredNode,
// The basic rule is: the bigger the score(matching number of pods) is, the smaller the
// final normalized score will be.
func CalculateEvenPodsSpreadPriorityReduce(pod *v1.Pod, meta interface{}, sharedLister schedulerlisters.SharedLister,
	result framework.NodeScoreList) error {
	var m *podTopologySpreadMap
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		m = priorityMeta.podTopologySpreadMap
	}
	if m == nil {
		return nil
	}

	// Calculate the summed <total> score and <minScore>.
	var minScore int64 = math.MaxInt64
	var total int64
	for _, score := range result {
		// it's mandatory to check if <score.Name> is present in m.nodeNameSet
		if _, ok := m.nodeNameSet[score.Name]; !ok {
			continue
		}
		total += score.Score
		if score.Score < minScore {
			minScore = score.Score
		}
	}

	maxMinDiff := total - minScore
	for i := range result {
		nodeInfo, err := sharedLister.NodeInfos().Get(result[i].Name)
		if err != nil {
			return err
		}
		node := nodeInfo.Node()
		// Debugging purpose: print the score for each node.
		// Score must be a pointer here, otherwise it's always 0.
		if klog.V(10) {
			defer func(score *int64, nodeName string) {
				klog.Infof("%v -> %v: PodTopologySpread NormalizeScore, Score: (%d)", pod.Name, nodeName, *score)
			}(&result[i].Score, node.Name)
		}

		if maxMinDiff == 0 {
			result[i].Score = framework.MaxNodeScore
			continue
		}

		if _, ok := m.nodeNameSet[node.Name]; !ok {
			result[i].Score = 0
			continue
		}

		flippedScore := total - result[i].Score
		fScore := float64(framework.MaxNodeScore) * (float64(flippedScore) / float64(maxMinDiff))
		result[i].Score = int64(fScore)
	}
	return nil
}

func filterSoftTopologySpreadConstraints(constraints []v1.TopologySpreadConstraint) ([]topologySpreadConstraint, error) {
	var r []topologySpreadConstraint
	for _, c := range constraints {
		if c.WhenUnsatisfiable == v1.ScheduleAnyway {
			selector, err := metav1.LabelSelectorAsSelector(c.LabelSelector)
			if err != nil {
				return nil, err
			}
			r = append(r, topologySpreadConstraint{
				topologyKey: c.TopologyKey,
				selector:    selector,
			})
		}
	}
	return r, nil
}

// nodeLabelsMatchSpreadConstraints checks if ALL topology keys in spread constraints are present in node labels.
func nodeLabelsMatchSpreadConstraints(nodeLabels map[string]string, constraints []topologySpreadConstraint) bool {
	for _, c := range constraints {
		if _, ok := nodeLabels[c.topologyKey]; !ok {
			return false
		}
	}
	return true
}
