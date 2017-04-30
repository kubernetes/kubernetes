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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/pkg/api"
)

// +genclient=true
// +resourceName=nodes
// +readonly=true
// +nonNamespaced=true

// resource usage metrics of a node.
type NodeMetrics struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// The following fields define time interval from which metrics were
	// collected from the interval [Timestamp-Window, Timestamp].
	Timestamp metav1.Time
	Window    metav1.Duration

	// The memory usage is the memory working set.
	Usage api.ResourceList
}

// NodeMetricsList is a list of NodeMetrics.
type NodeMetricsList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	metav1.ListMeta

	// List of node metrics.
	Items []NodeMetrics
}

// +genclient=true
// +resourceName=pods
// +readonly=true

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

// PodMetricsList is a list of PodMetrics.
type PodMetricsList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	metav1.ListMeta

	// List of pod metrics.
	Items []PodMetrics
}

// resource usage metrics of a container.
type ContainerMetrics struct {
	// Container name corresponding to the one from pod.spec.containers.
	Name string
	// The memory usage is the memory working set.
	Usage api.ResourceList
}
