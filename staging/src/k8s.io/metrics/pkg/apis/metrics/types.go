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

package metrics

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +resourceName=nodes
// +genclient:readonly
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// resource usage metrics of a node.
type NodeMetrics struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// The following fields define time interval from which metrics were
	// collected from the interval [Timestamp-Window, Timestamp].
	Timestamp metav1.Time
	Window    metav1.Duration

	// The memory usage is the memory working set.
	Usage corev1.ResourceList
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeMetricsList is a list of NodeMetrics.
type NodeMetricsList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds
	metav1.ListMeta

	// List of node metrics.
	Items []NodeMetrics
}

// +genclient
// +resourceName=pods
// +genclient:readonly
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// resource usage metrics of a pod.
type PodMetrics struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// The following fields define time interval from which metrics were
	// collected from the interval [Timestamp-Window, Timestamp].
	Timestamp metav1.Time
	Window    metav1.Duration

	// Metrics for all containers are collected within the same time window.
	Containers []ContainerMetrics
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodMetricsList is a list of PodMetrics.
type PodMetricsList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds
	metav1.ListMeta

	// List of pod metrics.
	Items []PodMetrics
}

// resource usage metrics of a container.
type ContainerMetrics struct {
	// Container name corresponding to the one from pod.spec.containers.
	Name string
	// The memory usage is the memory working set.
	Usage corev1.ResourceList
}
