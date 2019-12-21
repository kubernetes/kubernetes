/*
Copyright 2016 The Kubernetes Authors.

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

package predicates

import (
	"context"
	"math"
	"sync"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// Metadata interface represents anything that can access a predicate metadata.
// DEPRECATED.
type Metadata interface{}

type criticalPath struct {
	// topologyValue denotes the topology value mapping to topology key.
	topologyValue string
	// matchNum denotes the number of matching pods.
	matchNum int32
}

// CAVEAT: the reason that `[2]criticalPath` can work is based on the implementation of current
// preemption algorithm, in particular the following 2 facts:
// Fact 1: we only preempt pods on the same node, instead of pods on multiple nodes.
// Fact 2: each node is evaluated on a separate copy of the metadata during its preemption cycle.
// If we plan to turn to a more complex algorithm like "arbitrary pods on multiple nodes", this
// structure needs to be revisited.
type criticalPaths [2]criticalPath

func newCriticalPaths() *criticalPaths {
	return &criticalPaths{{matchNum: math.MaxInt32}, {matchNum: math.MaxInt32}}
}

func (paths *criticalPaths) update(tpVal string, num int32) {
	// first verify if `tpVal` exists or not
	i := -1
	if tpVal == paths[0].topologyValue {
		i = 0
	} else if tpVal == paths[1].topologyValue {
		i = 1
	}

	if i >= 0 {
		// `tpVal` exists
		paths[i].matchNum = num
		if paths[0].matchNum > paths[1].matchNum {
			// swap paths[0] and paths[1]
			paths[0], paths[1] = paths[1], paths[0]
		}
	} else {
		// `tpVal` doesn't exist
		if num < paths[0].matchNum {
			// update paths[1] with paths[0]
			paths[1] = paths[0]
			// update paths[0]
			paths[0].topologyValue, paths[0].matchNum = tpVal, num
		} else if num < paths[1].matchNum {
			// update paths[1]
			paths[1].topologyValue, paths[1].matchNum = tpVal, num
		}
	}
}

type topologyPair struct {
	key   string
	value string
}

// PodTopologySpreadMetadata combines tpKeyToCriticalPaths and tpPairToMatchNum
// to represent:
// (1) critical paths where the least pods are matched on each spread constraint.
// (2) number of pods matched on each spread constraint.
type PodTopologySpreadMetadata struct {
	constraints []topologySpreadConstraint
	// We record 2 critical paths instead of all critical paths here.
	// criticalPaths[0].matchNum always holds the minimum matching number.
	// criticalPaths[1].matchNum is always greater or equal to criticalPaths[0].matchNum, but
	// it's not guaranteed to be the 2nd minimum match number.
	tpKeyToCriticalPaths map[string]*criticalPaths
	// tpPairToMatchNum is keyed with topologyPair, and valued with the number of matching pods.
	tpPairToMatchNum map[topologyPair]int32
}

// topologySpreadConstraint is an internal version for a hard (DoNotSchedule
// unsatisfiable constraint action) v1.TopologySpreadConstraint and where the
// selector is parsed.
type topologySpreadConstraint struct {
	maxSkew     int32
	topologyKey string
	selector    labels.Selector
}

// GetPodTopologySpreadMetadata computes pod topology spread metadata.
func GetPodTopologySpreadMetadata(pod *v1.Pod, allNodes []*schedulernodeinfo.NodeInfo) (*PodTopologySpreadMetadata, error) {
	// We have feature gating in APIServer to strip the spec
	// so don't need to re-check feature gate, just check length of constraints.
	constraints, err := filterHardTopologySpreadConstraints(pod.Spec.TopologySpreadConstraints)
	if err != nil {
		return nil, err
	}
	if len(constraints) == 0 {
		return &PodTopologySpreadMetadata{}, nil
	}

	var lock sync.Mutex

	// TODO(Huang-Wei): It might be possible to use "make(map[topologyPair]*int32)".
	// In that case, need to consider how to init each tpPairToCount[pair] in an atomic fashion.
	m := PodTopologySpreadMetadata{
		constraints:          constraints,
		tpKeyToCriticalPaths: make(map[string]*criticalPaths, len(constraints)),
		tpPairToMatchNum:     make(map[topologyPair]int32),
	}
	addTopologyPairMatchNum := func(pair topologyPair, num int32) {
		lock.Lock()
		m.tpPairToMatchNum[pair] += num
		lock.Unlock()
	}

	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
			return
		}
		// In accordance to design, if NodeAffinity or NodeSelector is defined,
		// spreading is applied to nodes that pass those filters.
		if !PodMatchesNodeSelectorAndAffinityTerms(pod, node) {
			return
		}

		// Ensure current node's labels contains all topologyKeys in 'constraints'.
		if !NodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
			return
		}
		for _, constraint := range constraints {
			matchTotal := int32(0)
			// nodeInfo.Pods() can be empty; or all pods don't fit
			for _, existingPod := range nodeInfo.Pods() {
				if existingPod.Namespace != pod.Namespace {
					continue
				}
				if constraint.selector.Matches(labels.Set(existingPod.Labels)) {
					matchTotal++
				}
			}
			pair := topologyPair{key: constraint.topologyKey, value: node.Labels[constraint.topologyKey]}
			addTopologyPairMatchNum(pair, matchTotal)
		}
	}
	workqueue.ParallelizeUntil(context.Background(), 16, len(allNodes), processNode)

	// calculate min match for each topology pair
	for i := 0; i < len(constraints); i++ {
		key := constraints[i].topologyKey
		m.tpKeyToCriticalPaths[key] = newCriticalPaths()
	}
	for pair, num := range m.tpPairToMatchNum {
		m.tpKeyToCriticalPaths[pair.key].update(pair.value, num)
	}

	return &m, nil
}

func filterHardTopologySpreadConstraints(constraints []v1.TopologySpreadConstraint) ([]topologySpreadConstraint, error) {
	var result []topologySpreadConstraint
	for _, c := range constraints {
		if c.WhenUnsatisfiable == v1.DoNotSchedule {
			selector, err := metav1.LabelSelectorAsSelector(c.LabelSelector)
			if err != nil {
				return nil, err
			}
			result = append(result, topologySpreadConstraint{
				maxSkew:     c.MaxSkew,
				topologyKey: c.TopologyKey,
				selector:    selector,
			})
		}
	}
	return result, nil
}

// NodeLabelsMatchSpreadConstraints checks if ALL topology keys in spread constraints are present in node labels.
func NodeLabelsMatchSpreadConstraints(nodeLabels map[string]string, constraints []topologySpreadConstraint) bool {
	for _, c := range constraints {
		if _, ok := nodeLabels[c.topologyKey]; !ok {
			return false
		}
	}
	return true
}

// AddPod updates the metadata with addedPod.
func (m *PodTopologySpreadMetadata) AddPod(addedPod, preemptorPod *v1.Pod, node *v1.Node) {
	m.updateWithPod(addedPod, preemptorPod, node, 1)
}

// RemovePod updates the metadata with deletedPod.
func (m *PodTopologySpreadMetadata) RemovePod(deletedPod, preemptorPod *v1.Pod, node *v1.Node) {
	m.updateWithPod(deletedPod, preemptorPod, node, -1)
}

func (m *PodTopologySpreadMetadata) updateWithPod(updatedPod, preemptorPod *v1.Pod, node *v1.Node, delta int32) {
	if m == nil || updatedPod.Namespace != preemptorPod.Namespace || node == nil {
		return
	}
	if !NodeLabelsMatchSpreadConstraints(node.Labels, m.constraints) {
		return
	}

	podLabelSet := labels.Set(updatedPod.Labels)
	for _, constraint := range m.constraints {
		if !constraint.selector.Matches(podLabelSet) {
			continue
		}

		k, v := constraint.topologyKey, node.Labels[constraint.topologyKey]
		pair := topologyPair{key: k, value: v}
		m.tpPairToMatchNum[pair] = m.tpPairToMatchNum[pair] + delta

		m.tpKeyToCriticalPaths[k].update(v, m.tpPairToMatchNum[pair])
	}
}

// Clone makes a deep copy of PodTopologySpreadMetadata.
func (m *PodTopologySpreadMetadata) Clone() *PodTopologySpreadMetadata {
	// m could be nil when EvenPodsSpread feature is disabled
	if m == nil {
		return nil
	}
	cp := PodTopologySpreadMetadata{
		// constraints are shared because they don't change.
		constraints:          m.constraints,
		tpKeyToCriticalPaths: make(map[string]*criticalPaths, len(m.tpKeyToCriticalPaths)),
		tpPairToMatchNum:     make(map[topologyPair]int32, len(m.tpPairToMatchNum)),
	}
	for tpKey, paths := range m.tpKeyToCriticalPaths {
		cp.tpKeyToCriticalPaths[tpKey] = &criticalPaths{paths[0], paths[1]}
	}
	for tpPair, matchNum := range m.tpPairToMatchNum {
		copyPair := topologyPair{key: tpPair.key, value: tpPair.value}
		cp.tpPairToMatchNum[copyPair] = matchNum
	}
	return &cp
}
