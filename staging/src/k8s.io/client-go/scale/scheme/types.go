/*
Copyright 2017 The Kubernetes Authors.

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

package scheme

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// This file contains our own "internal" version of scale that we use for conversions,
// since we can't use the main Kubernetes internal versions.

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Scale represents a scaling request for a resource.
type Scale struct {
	metav1.TypeMeta
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta

	// defines the behavior of the scale. More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#spec-and-status.
	// +optional
	Spec ScaleSpec

	// current status of the scale. More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#spec-and-status. Read-only.
	// +optional
	Status ScaleStatus
}

// ScaleSpec describes the attributes of a scale subresource.
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	// +optional
	Replicas int32
}

// ScaleStatus represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int32

	// label query over pods that should match the replicas count.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	Selector *metav1.LabelSelector
}
