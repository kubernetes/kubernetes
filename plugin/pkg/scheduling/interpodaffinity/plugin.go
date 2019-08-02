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

package interpodaffinity

import (
	"math"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// InterPodAffinity is a plugin implements scoring extension point.
type InterPodAffinity struct {
	nodeInfoSnapshot      *nodeinfo.Snapshot
	hardPodAffinityWeight int32

	currentPodAffinityPriorityMap *podAffinityPriorityMap
}

var _ framework.PostFilterPlugin = &InterPodAffinity{}
var _ framework.ScoreWithNormalizePlugin = &InterPodAffinity{}

// Name is the name of the interpod affinity plugin
const Name = "interpod-affinity-plugin"

// Name .
func (ipa *InterPodAffinity) Name() string {
	return Name
}

// PostFilter .
func (ipa *InterPodAffinity) PostFilter(pc *framework.PluginContext, pod *v1.Pod, _ []*v1.Node, _ framework.NodeToStatusMap) *framework.Status {
	ipa.currentPodAffinityPriorityMap = newPodAffinityPriorityMap()

	return nil
}

// Score .
func (ipa *InterPodAffinity) Score(pc *framework.PluginContext, pod *v1.Pod, nodeName string) (int, *framework.Status) {
	pm := ipa.currentPodAffinityPriorityMap
	if pm == nil {
		return 0, framework.NewStatus(framework.Error, "invalid pod affinity priority map")
	}

	nodeInfo, ok := ipa.nodeInfoSnapshot.NodeInfoMap[nodeName]
	if !ok {
		return 0, framework.NewStatus(framework.Error, "node is empty")
	}

	nodes := []*v1.Node{}
	for _, nodeInfo := range ipa.nodeInfoSnapshot.NodeInfoMap {
		nodes = append(nodes, nodeInfo.Node())
	}

	affinity := pod.Spec.Affinity
	hasAffinityConstraints := affinity != nil && affinity.PodAffinity != nil
	hasAntiAffinityConstraints := affinity != nil && affinity.PodAntiAffinity != nil

	processPod := func(existingPod *v1.Pod) error {
		existingNodeInfo, ok := ipa.nodeInfoSnapshot.NodeInfoMap[existingPod.Spec.NodeName]
		if !ok {
			klog.Errorf("Node not found, %v", existingPod.Spec.NodeName)
			return nil
		}
		existingPodNode := existingNodeInfo.Node()
		existingPodAffinity := existingPod.Spec.Affinity
		existingHasAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAffinity != nil
		existingHasAntiAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAntiAffinity != nil

		if hasAffinityConstraints {
			// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPods>`s node by the term`s weight.
			terms := affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := pm.processTerms(terms, pod, existingPod, existingPodNode, nodes, 1); err != nil {
				return err
			}
		}
		if hasAntiAffinityConstraints {
			// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
			// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>`s node by the term`s weight.
			terms := affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := pm.processTerms(terms, pod, existingPod, existingPodNode, nodes, -1); err != nil {
				return err
			}
		}

		if existingHasAffinityConstraints {
			// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the constant <ipa.hardPodAffinityWeight>
			if ipa.hardPodAffinityWeight > 0 {
				terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
				// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
				//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
				//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
				//}
				for _, term := range terms {
					if err := pm.processTerm(&term, existingPod, pod, existingPodNode, nodes, int64(ipa.hardPodAffinityWeight)); err != nil {
						return err
					}
				}
			}
			// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			terms := existingPodAffinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution

			if err := pm.processTerms(terms, existingPod, pod, existingPodNode, nodes, 1); err != nil {
				return err
			}
		}
		if existingHasAntiAffinityConstraints {
			// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
			// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			terms := existingPodAffinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			if err := pm.processTerms(terms, existingPod, pod, existingPodNode, nodes, -1); err != nil {
				return err
			}
		}
		return nil
	}

	if hasAffinityConstraints || hasAntiAffinityConstraints {
		// We need to process all the pods.
		for _, existingPod := range nodeInfo.Pods() {
			if err := processPod(existingPod); err != nil {
				return 0, framework.NewStatus(framework.Error, err.Error())
			}
		}
	} else {
		// The pod doesn't have any constraints - we need to check only existing
		// ones that have some.
		for _, existingPod := range nodeInfo.PodsWithAffinity() {
			if err := processPod(existingPod); err != nil {
				return 0, framework.NewStatus(framework.Error, err.Error())
			}
		}
	}

	return 0, nil
}

// NormalizeScore .
func (ipa *InterPodAffinity) NormalizeScore(pc *framework.PluginContext, pod *v1.Pod, nodeScores framework.NodeScoreList) *framework.Status {
	pm := ipa.currentPodAffinityPriorityMap
	if pm == nil {
		return framework.NewStatus(framework.Error, "invalid pod affinity priority map")
	}

	// convert the topology key based weights to the node name based weights
	var maxCount, minCount float64

	for _, nodeScore := range nodeScores {
		currentScore, ok := pm.counts[nodeScore.Name]
		if !ok {
			continue
		}

		maxCount = math.Max(maxCount, float64(currentScore))
		minCount = math.Min(minCount, float64(currentScore))
	}

	// calculate final priority score for each node
	maxMinDiff := maxCount - minCount
	for i, nodeScore := range nodeScores {
		fScore := float64(0)
		if maxMinDiff > 0 {
			// TODO: use framework.MaxNodeScore
			fScore = float64(schedulerapi.MaxPriority) * (float64(pm.counts[nodeScore.Name]-int64(minCount)) / (maxCount - minCount))
		}

		nodeScores[i].Name = nodeScore.Name
		nodeScores[i].Score = int(fScore)
		if klog.V(10) {
			klog.Infof("%v -> %v: InterPodAffinityPriority, Score: (%d)", pod.Name, nodeScore.Name, int(fScore))
		}
	}

	return nil
}

// New returns a new interpod affinity plugin.
func New(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
	return &InterPodAffinity{
		nodeInfoSnapshot: fh.NodeInfoSnapshot(),
		// TODO(draveness): update hard pod affinity weight with runtime arguments.
		hardPodAffinityWeight: 1,
	}, nil
}
