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

package v1

// TODO(vmarmol): Things we should consider adding (note that we only get 10 labels):
// - POD name, container name, and host IP: Useful to users but maybe we should just mangle them with ID and IP
// - Namespace: Are IDs unique only per namespace? If so, mangle it into the ID.
var (
	LabelPodId = LabelDescriptor{
		Key:         "pod_id",
		Description: "The unique ID of the pod",
	}
	LabelPodName = LabelDescriptor{
		Key:         "pod_name",
		Description: "The name of the pod",
	}
	LabelPodNamespace = LabelDescriptor{
		Key:         "pod_namespace",
		Description: "The namespace of the pod",
	}
	LabelPodNamespaceUID = LabelDescriptor{
		Key:         "namespace_id",
		Description: "The UID of namespace of the pod",
	}
	LabelContainerName = LabelDescriptor{
		Key:         "container_name",
		Description: "User-provided name of the container or full container name for system containers",
	}
	LabelLabels = LabelDescriptor{
		Key:         "labels",
		Description: "Comma-separated list of user-provided labels",
	}
	LabelHostname = LabelDescriptor{
		Key:         "hostname",
		Description: "Hostname where the container ran",
	}
	LabelResourceID = LabelDescriptor{
		Key:         "resource_id",
		Description: "Identifier(s) specific to a metric",
	}
	LabelHostID = LabelDescriptor{
		Key:         "host_id",
		Description: "Identifier specific to a host. Set by cloud provider or user",
	}
	LabelContainerBaseImage = LabelDescriptor{
		Key:         "container_base_image",
		Description: "User-defined image name that is run inside the container",
	}
)

var commonLabels = []LabelDescriptor{
	LabelHostname,
	LabelHostID,
	LabelContainerName,
	LabelContainerBaseImage,
}

var podLabels = []LabelDescriptor{
	LabelPodName,
	LabelPodId,
	LabelPodNamespace,
	LabelPodNamespaceUID,
	LabelLabels,
}

var metricLabels = []LabelDescriptor{
	LabelResourceID,
}

func CommonLabels() []LabelDescriptor {
	result := make([]LabelDescriptor, len(commonLabels))
	copy(result, commonLabels)
	return result
}

func PodLabels() []LabelDescriptor {
	result := make([]LabelDescriptor, len(podLabels))
	copy(result, podLabels)
	return result
}

func MetricLabels() []LabelDescriptor {
	result := make([]LabelDescriptor, len(metricLabels))
	copy(result, metricLabels)
	return result
}

func SupportedLabels() []LabelDescriptor {
	result := CommonLabels()
	result = append(result, PodLabels()...)
	return append(result, MetricLabels()...)
}
