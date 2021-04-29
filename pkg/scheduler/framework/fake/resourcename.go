/*
Copyright 2021 The Kubernetes Authors.

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

package fake

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// ResourceNameQualifier implements ResourceNameQualifier interface
// to inject explicit resource names of given kinds for testing purposes
//
// E.g.
// &resourceNameQualifier{
// 	ExtendedResourceNames: []v1.ResourceName{"scalar.test/scalar1"},
// 	HugePageResourceNames: []v1.ResourceName{"hugepages-test"},
// }
//
type ResourceNameQualifier struct {
	ExtendedResourceNames         []v1.ResourceName
	HugePageResourceNames         []v1.ResourceName
	AttachableVolumeResourceNames []v1.ResourceName
	PrefixedNativeResourceNames   []v1.ResourceName
}

func (rnq *ResourceNameQualifier) IsExtended(name v1.ResourceName) bool {
	return inList(rnq.ExtendedResourceNames, name)
}

func (rnq *ResourceNameQualifier) IsHugePage(name v1.ResourceName) bool {
	return inList(rnq.HugePageResourceNames, name)
}

func (rnq *ResourceNameQualifier) IsAttachableVolume(name v1.ResourceName) bool {
	return inList(rnq.AttachableVolumeResourceNames, name)
}

func (rnq *ResourceNameQualifier) IsPrefixedNativeResource(name v1.ResourceName) bool {
	return inList(rnq.PrefixedNativeResourceNames, name)
}

func inList(list []v1.ResourceName, item v1.ResourceName) bool {
	for i := range list {
		if list[i] == item {
			return true
		}
	}
	return false
}

var _ framework.ResourceNameQualifier = &ResourceNameQualifier{}

func NewNodeInfoWithEmptyResourceNameQualifier() *framework.NodeInfo {
	return framework.NewNodeInfo(&ResourceNameQualifier{})
}

func EmptyResourceNameQualifier() framework.ResourceNameQualifier {
	return &ResourceNameQualifier{}
}
