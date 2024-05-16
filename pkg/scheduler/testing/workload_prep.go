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

	v1 "k8s.io/api/core/v1"
)

type keyVal struct {
	k string
	v string
}

// MakeNodesAndPodsForEvenPodsSpread serves as a testing helper for EvenPodsSpread feature.
// It builds a fake cluster containing running Pods and Nodes.
// The size of Pods and Nodes are determined by input arguments.
// The specs of Pods and Nodes are generated with the following rules:
//   - Each generated node is applied with a unique label: "node: node<i>".
//   - Each generated node is applied with a rotating label: "zone: zone[0-9]".
//   - Depending on the input labels, each generated pod will be applied with
//     label "key1", "key1,key2", ..., "key1,key2,...,keyN" in a rotating manner.
func MakeNodesAndPodsForEvenPodsSpread(labels map[string]string, existingPodsNum, allNodesNum, filteredNodesNum int) (existingPods []*v1.Pod, allNodes []*v1.Node, filteredNodes []*v1.Node) {
	var labelPairs []keyVal
	for k, v := range labels {
		labelPairs = append(labelPairs, keyVal{k: k, v: v})
	}
	zones := 10
	// build nodes
	for i := 0; i < allNodesNum; i++ {
		node := MakeNode().Name(fmt.Sprintf("node%d", i)).
			Label(v1.LabelTopologyZone, fmt.Sprintf("zone%d", i%zones)).
			Label(v1.LabelHostname, fmt.Sprintf("node%d", i)).Obj()
		allNodes = append(allNodes, node)
	}
	filteredNodes = allNodes[:filteredNodesNum]
	// build pods
	for i := 0; i < existingPodsNum; i++ {
		podWrapper := MakePod().Name(fmt.Sprintf("pod%d", i)).Node(fmt.Sprintf("node%d", i%allNodesNum))
		// apply labels[0], labels[0,1], ..., labels[all] to each pod in turn
		for _, p := range labelPairs[:i%len(labelPairs)+1] {
			podWrapper = podWrapper.Label(p.k, p.v)
		}
		existingPods = append(existingPods, podWrapper.Obj())
	}
	return
}
