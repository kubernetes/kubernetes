// Copyright 2015 Google Inc. All Rights Reserved.
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

package types

import (
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// DerivedNodeMetricsList contains derived metrics for a single node. It holds
// derived metrics in windows about the node itself and all of its system
// containers.
type DerivedNodeMetrics struct {
	// NodeName is the name of the node
	NodeName string `json:"nodeName"`
	// nodeMetrics contains derived metrics about the node.
	NodeMetrics MetricsWindows `json:"nodeMetrics"`
	// systemContainers contains derived container metrics for all the
	// cgroups that are managed by the kubelet on the node.
	SystemContainers []DerivedContainerMetrics `json:"systemContainers"`
}

// DerivedNodeMetricsList contains derived metrics for a single container.
type DerivedContainerMetrics struct {
	// containerName is the name of the raw cgroup.
	ContainerName string `json:"containerName"`
	// containerMetrics contains derived metrics about the container.
	ContainerMetrics MetricsWindows `json:"containerMetrics"`
}

// MetricsWindows holds multiple derived metrics windows.
type MetricsWindows struct {
	// endTime is the end time of all the time windows.
	EndTime unversioned.Time `json:"endTime"`

	// windows is a list of all the time windows with metrics. All of the
	// windows are rolling.
	Windows []MetricsWindow `json:"windows"`
}

// MetricsWindow holds derived metrics for multiple resources.
type MetricsWindow struct {
	// duration is the length in time of the window. The start of the
	// window will be the subtraction of this duration to EndTime.
	Duration unversioned.Duration `json:"duration"`

	// mean holds the averages over the window.
	Mean ResourceUsage `json:"mean"`
	// max holds the maximum values over the window.
	Max ResourceUsage `json:"max"`
	// ninetyFifthPercentile holds the 95th percentile values over the window.
	NinetyFifthPercentile ResourceUsage `json:"ninetyFifthPercentile"`
}

// ResourceUsage maps resource names to their metric values.
// Resource names are "cpu" and "memory".
type ResourceUsage map[string]resource.Quantity
