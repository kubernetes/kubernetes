// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v1alpha1

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
)

// resource usage metrics of a node.
type NodeMetrics struct {
	unversioned.TypeMeta `json:",inline"`
	v1.ObjectMeta        `json:"metadata,omitempty"`

	// The following fields define time interval from which metrics were
	// collected from the interval [Timestamp-Window, Timestamp].
	Timestamp unversioned.Time     `json:"timestamp"`
	Window    unversioned.Duration `json:"window"`

	// The memory usage is the memory working set.
	Usage v1.ResourceList `json:"usage"`
}

// NodeMetricsList is a list of NodeMetrics.
type NodeMetricsList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`

	// List of node metrics.
	Items []NodeMetrics `json:"items"`
}

// resource usage metrics of a pod.
type PodMetrics struct {
	unversioned.TypeMeta `json:",inline"`
	v1.ObjectMeta        `json:"metadata,omitempty"`

	// The following fields define time interval from which metrics were
	// collected from the interval [Timestamp-Window, Timestamp].
	Timestamp unversioned.Time     `json:"timestamp"`
	Window    unversioned.Duration `json:"window"`

	// Metrics for all containers are collected within the same time window.
	Containers []ContainerMetrics `json:"containers"`
}

// PodMetricsList is a list of PodMetrics.
type PodMetricsList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`

	// List of pod metrics.
	Items []PodMetrics `json:"items"`
}

// resource usage metrics of a container.
type ContainerMetrics struct {
	// Container name corresponding to the one from pod.spec.containers.
	Name string `json:"name"`
	// The memory usage is the memory working set.
	Usage v1.ResourceList `json:"usage"`
}
