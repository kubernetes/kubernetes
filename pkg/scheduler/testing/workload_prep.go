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

type keyVal struct {
	k string
	v string
}

// MakeNodesAndPodsForEvenPodsSpread serves as a testing helper for EvenPodsSpread feature.
// It builds a fake cluster containing running Pods and Nodes.
// The size of Pods and Nodes are determined by input arguments.
// The specs of Pods and Nodes are generated with the following rules:
// - Each generated node is applied with a unique label: "node: node<i>".
// - Each generated node is applied with a rotating label: "zone: zone[0-9]".
// - Depending on the input labels, each generated pod will be applied with
//   label "key1", "key1,key2", ..., "key1,key2,...,keyN" in a rotating manner.
func MakeNodesAndPodsForEvenPodsSpread(labels map[string]string, existingPodsNum, allNodesNum, filteredNodesNum int) (existingPods []*v1.Pod, allNodes []*v1.Node, filteredNodes []*v1.Node) {
	var labelPairs []keyVal
	for k, v := range labels {
		labelPairs = append(labelPairs, keyVal{k: k, v: v})
	}
	zones := 10
	// build nodes
	for i := 0; i < allNodesNum; i++ {
		node := MakeNode().Name(fmt.Sprintf("node%d", i)).
			Label(v1.LabelZoneFailureDomain, fmt.Sprintf("zone%d", i%zones)).
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

// MakeNodesAndPodsForPodAffinity serves as a testing helper for Pod(Anti)Affinity feature.
// It builds a fake cluster containing running Pods and Nodes.
// For simplicity, the Nodes will be labelled with "region", "zone" and "node". Nodes[i] will be applied with:
// - "region": "region" + i%3
// - "zone": "zone" + i%10
// - "node": "node" + i
// The Pods will be applied with various combinations of PodAffinity and PodAntiAffinity terms.
func MakeNodesAndPodsForPodAffinity(existingPodsNum, allNodesNum int) (existingPods []*v1.Pod, allNodes []*v1.Node) {
	tpKeyToSizeMap := map[string]int{
		"region": 3,
		"zone":   10,
		"node":   allNodesNum,
	}
	// build nodes to spread across all topology domains
	for i := 0; i < allNodesNum; i++ {
		nodeName := fmt.Sprintf("node%d", i)
		nodeWrapper := MakeNode().Name(nodeName)
		for tpKey, size := range tpKeyToSizeMap {
			nodeWrapper = nodeWrapper.Label(tpKey, fmt.Sprintf("%s%d", tpKey, i%size))
		}
		allNodes = append(allNodes, nodeWrapper.Obj())
	}

	labels := []string{"foo", "bar", "baz"}
	tpKeys := []string{"region", "zone", "node"}

	// Build pods.
	// Each pod will be created with one affinity and one anti-affinity terms using all combinations of
	// affinity and anti-affinity kinds listed below
	// e.g., the first pod will have {affinity, anti-affinity} terms of kinds {NilPodAffinity, NilPodAffinity};
	// the second will be {NilPodAffinity, PodAntiAffinityWithRequiredReq}, etc.
	affinityKinds := []PodAffinityKind{
		NilPodAffinity,
		PodAffinityWithRequiredReq,
		PodAffinityWithPreferredReq,
		PodAffinityWithRequiredPreferredReq,
	}
	antiAffinityKinds := []PodAffinityKind{
		NilPodAffinity,
		PodAntiAffinityWithRequiredReq,
		PodAntiAffinityWithPreferredReq,
		PodAntiAffinityWithRequiredPreferredReq,
	}

	totalSize := len(affinityKinds) * len(antiAffinityKinds)
	for i := 0; i < existingPodsNum; i++ {
		podWrapper := MakePod().Name(fmt.Sprintf("pod%d", i)).Node(fmt.Sprintf("node%d", i%allNodesNum))
		label, tpKey := labels[i%len(labels)], tpKeys[i%len(tpKeys)]

		affinityIdx := i % totalSize
		// len(affinityKinds) is equal to len(antiAffinityKinds)
		leftIdx, rightIdx := affinityIdx/len(affinityKinds), affinityIdx%len(affinityKinds)
		podWrapper = podWrapper.PodAffinityExists(label, tpKey, affinityKinds[leftIdx])
		podWrapper = podWrapper.PodAntiAffinityExists(label, tpKey, antiAffinityKinds[rightIdx])
		existingPods = append(existingPods, podWrapper.Obj())
	}

	return
}

// MakeNodesAndPods serves as a testing helper to generate regular Nodes and Pods
// that don't use any advanced scheduling features.
func MakeNodesAndPods(existingPodsNum, allNodesNum int) (existingPods []*v1.Pod, allNodes []*v1.Node) {
	// build nodes
	for i := 0; i < allNodesNum; i++ {
		allNodes = append(allNodes, MakeNode().Name(fmt.Sprintf("node%d", i)).Obj())
	}
	// build pods
	for i := 0; i < existingPodsNum; i++ {
		podWrapper := MakePod().Name(fmt.Sprintf("pod%d", i)).Node(fmt.Sprintf("node%d", i%allNodesNum))
		existingPods = append(existingPods, podWrapper.Obj())
	}
	return
}
