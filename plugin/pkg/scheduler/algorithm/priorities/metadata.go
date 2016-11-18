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

package priorities

import (
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// priorityMetadata is a type that is passed as metadata for priority functions
type priorityMetadata struct {
	nonZeroRequest *schedulercache.Resource
	podTolerations []v1.Toleration
	affinity       *v1.Affinity
}

// PriorityMetadata is a MetadataProducer.  Node info can be nil.
func PriorityMetadata(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo) interface{} {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	tolerations, err := getTolerationListFromPod(pod)
	if err != nil {
		return nil
	}
	affinity, err := v1.GetAffinityFromPodAnnotations(pod.Annotations)
	if err != nil {
		return nil
	}
	return &priorityMetadata{
		nonZeroRequest: getNonZeroRequests(pod),
		podTolerations: tolerations,
		affinity:       affinity,
	}
}
