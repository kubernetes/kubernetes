/*
Copyright The Kubernetes Authors.

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

package preemption

import (
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	fwk "k8s.io/kube-scheduler/framework"
)

// This is for test usage only, not for the PRODUCTION code
func NewDomainsForTest(ev Evaluator, preemptor Preemptor, allNodes []fwk.NodeInfo, enableWorkloadAwarePreemption bool) []Domain {
	var domains []Domain
	if enableWorkloadAwarePreemption {
		ev.podGroupIndex = buildPodGroupIndex(allNodes)
		if preemptor.IsPodGroup() {
			return []Domain{ev.newSingleDomainForAllNodes(preemptor, allNodes)}
		}
		for _, node := range allNodes {
			nodes := []fwk.NodeInfo{node}
			domains = append(domains, ev.newDomain(node.Node().Name, nodes, ev.getAllPossibleVictims(nodes)))
		}
		return domains
	}

	for _, node := range allNodes {
		var victims []PreemptionUnit
		for _, pod := range node.GetPods() {
			victims = append(victims, newPreemptionUnit(
				[]fwk.PodInfo{pod},
				corev1helpers.PodPriority(pod.GetPod()),
				[]fwk.NodeInfo{node},
			))
		}
		domains = append(domains, ev.newDomain(node.Node().Name, []fwk.NodeInfo{node}, victims))
	}
	return domains
}
