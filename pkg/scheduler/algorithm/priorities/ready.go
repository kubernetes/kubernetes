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
	"k8s.io/api/core/v1"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	"math"
)

// ReadyPod contains information to calculate ready pod score.
type ReadyPod struct {
}

// NewReadyPodPriority creates a ReadyPod.
func NewReadyPodPriority() PriorityFunction {
	readyPod := &ReadyPod{}
	return readyPod.CalculateReadyPodPriority
}
func (r *ReadyPod) isPodReady(pod *v1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == "Ready" && cond.Status == "True" {
			return true
		}
	}
	return false
}

// CalculateReadyPodPriority favors nodes with lesser count of non ready pods.
// Calcuated on a scale of 0-10 where 10 is highest priority.
func (r *ReadyPod) CalculateReadyPodPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo, nodes []*v1.Node) (framework.NodeScoreList, error) {
	totalPodsNotReady := 0
	nodeToNonReadyPods := make(map[string]int)
	// Get count of non ready pods for each node.
	for _, node := range nodes {
		nonReadyPods := 0
		nodeinfo := nodeNameToInfo[node.Name]
		pods := nodeinfo.Pods()
		for _, p := range pods {
			isPodReady := r.isPodReady(p)
			klog.V(10).Infof("node: %v, pod: %v, ready: %v", node.Name, p.ObjectMeta.Name, isPodReady)
			if !r.isPodReady(p) {
				nonReadyPods++
			}
		}
		klog.V(10).Infof("node: %v, non ready pods: %v", node.Name, nonReadyPods)
		nodeToNonReadyPods[node.Name] = nonReadyPods
		totalPodsNotReady += nonReadyPods
	}
	// Assign host priority to nodes with lesser count of non ready pods.
	result := make(framework.NodeScoreList, 0, len(nodes))
	for _, node := range nodes {
		if totalPodsNotReady == 0 {
			// all nodes have all pods in ready state
			result = append(result, framework.NodeScore{Name: node.Name, Score: 10})
		} else {
			// prefer nodes nodes with least non ready pods and thus higher score
			// formula: 10 - ( nonReadyPods / totalPodsNotReady ) * 10
			score := math.Round(10 - (float64(nodeToNonReadyPods[node.Name])/float64(totalPodsNotReady))*10)
			result = append(result, framework.NodeScore{Name: node.Name, Score: int64(score)})
		}
	}
	klog.V(5).Infof("result: %v", result)
	return result, nil
}
