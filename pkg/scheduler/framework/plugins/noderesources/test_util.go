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

package noderesources

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func makeNode(node string, milliCPU, memory int64) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: node},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
			},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
			},
		},
	}
}

func makeNodeWithExtendedResource(node string, milliCPU, memory int64, extendedResource map[string]int64) *v1.Node {
	resourceList := make(map[v1.ResourceName]resource.Quantity)
	for res, quantity := range extendedResource {
		resourceList[v1.ResourceName(res)] = *resource.NewQuantity(quantity, resource.DecimalSI)
	}
	resourceList[v1.ResourceCPU] = *resource.NewMilliQuantity(milliCPU, resource.DecimalSI)
	resourceList[v1.ResourceMemory] = *resource.NewQuantity(memory, resource.BinarySI)
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: node},
		Status: v1.NodeStatus{
			Capacity:    resourceList,
			Allocatable: resourceList,
		},
	}
}
