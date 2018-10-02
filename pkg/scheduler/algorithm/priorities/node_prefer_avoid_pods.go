/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
)

// CalculateNodePreferAvoidPodsPriorityMap priorities nodes according to the node annotation
// "scheduler.alpha.kubernetes.io/preferAvoidPods".
func CalculateNodePreferAvoidPodsPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}
	var controllerRef *metav1.OwnerReference
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		controllerRef = priorityMeta.controllerRef
	} else {
		// We couldn't parse metadata - fallback to the podspec.
		controllerRef = metav1.GetControllerOf(pod)
	}

	if controllerRef != nil {
		// Ignore pods that are owned by other controller than ReplicationController
		// or ReplicaSet.
		if controllerRef.Kind != "ReplicationController" && controllerRef.Kind != "ReplicaSet" {
			controllerRef = nil
		}
	}
	if controllerRef == nil {
		return schedulerapi.HostPriority{Host: node.Name, Score: schedulerapi.MaxPriority}, nil
	}

	avoids, err := v1helper.GetAvoidPodsFromNodeAnnotations(node.Annotations)
	if err != nil {
		// If we cannot get annotation, assume it's schedulable there.
		return schedulerapi.HostPriority{Host: node.Name, Score: schedulerapi.MaxPriority}, nil
	}
	for i := range avoids.PreferAvoidPods {
		avoid := &avoids.PreferAvoidPods[i]
		if avoid.PodSignature.PodController.Kind == controllerRef.Kind && avoid.PodSignature.PodController.UID == controllerRef.UID {
			return schedulerapi.HostPriority{Host: node.Name, Score: 0}, nil
		}
	}
	return schedulerapi.HostPriority{Host: node.Name, Score: schedulerapi.MaxPriority}, nil
}
