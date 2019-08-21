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

package testing

import (
	"fmt"

	"k8s.io/api/core/v1"
)

// MakeNodesAndPods serves as a testing helper for EvenPodsSpread feature.
// It builds a fake cluster containing running Pods and Nodes.
// The size of Pods and Nodes are determined by input arguments.
// The specs of Pods and Nodes are generated with the following rules:
// - If `pod` has "node" as a topologyKey, each generated node is applied with a unique label: "node: node<i>".
// - If `pod` has "zone" as a topologyKey, each generated node is applied with a rotating label: "zone: zone[0-9]".
// - Depending on "lableSelector.MatchExpressions[0].Key" the `pod` has in each topologySpreadConstraint,
//   each generated pod will be applied with label "key1", "key1,key2", ..., "key1,key2,...,keyN" in a rotating manner.
func MakeNodesAndPods(pod *v1.Pod, existingPodsNum, allNodesNum, filteredNodesNum int) (existingPods []*v1.Pod, allNodes []*v1.Node, filteredNodes []*v1.Node) {
	var topologyKeys []string
	var labels []string
	zones := 10
	for _, c := range pod.Spec.TopologySpreadConstraints {
		topologyKeys = append(topologyKeys, c.TopologyKey)
		labels = append(labels, c.LabelSelector.MatchExpressions[0].Key)
	}
	// build nodes
	for i := 0; i < allNodesNum; i++ {
		nodeWrapper := MakeNode().Name(fmt.Sprintf("node%d", i))
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
		podWrapper := MakePod().Name(fmt.Sprintf("pod%d", i)).Node(fmt.Sprintf("node%d", i%allNodesNum))
		// apply labels[0], labels[0,1], ..., labels[all] to each pod in turn
		for _, label := range labels[:i%len(labels)+1] {
			podWrapper = podWrapper.Label(label, "")
		}
		existingPods = append(existingPods, podWrapper.Obj())
	}
	return
}
