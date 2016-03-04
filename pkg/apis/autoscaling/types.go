/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package autoscaling

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// Scale represents a scaling request for a resource.
type Scale struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata; More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#metadata.
	api.ObjectMeta `json:"metadata,omitempty"`

	// defines the behavior of the scale. More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status.
	Spec ScaleSpec `json:"spec,omitempty"`

	// current status of the scale. More info: http://releases.k8s.io/release-1.2/docs/devel/api-conventions.md#spec-and-status. Read-only.
	Status ScaleStatus `json:"status,omitempty"`
}

// ScaleSpec describes the attributes of a scale subresource.
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	Replicas int `json:"replicas,omitempty"`
}

// ScaleStatus represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int `json:"replicas"`

	// label query over pods that should match the replicas count. This is same
	// as the label selector but in the string format to avoid introspection
	// by clients. The string will be in the same format as the query-param syntax.
	// More info: http://releases.k8s.io/release-1.2/docs/user-guide/labels.md#label-selectors
	Selector string `json:"selector,omitempty"`
}
