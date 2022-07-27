/*
Copyright 2022 The Kubernetes Authors.

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
package dra

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ActivePodsFunc is a function that returns a list of pods to reconcile.
type ActivePodsFunc func() []*v1.Pod

// Manager manages all the kubelet resource plugins running on a node.
type Manager interface {
	// Configure configures DRA manager
	Configure(activePods ActivePodsFunc, sourcesReady config.SourcesReady)

	// Allocate prepares and assigns resources to a container in a pod. From
	// the requested resources, Allocate will communicate with the
	// kubelet resource plugin to prepare resources.
	Allocate(pod *v1.Pod, container *v1.Container) error

	// TopologyManager HintProvider provider indicates the Device Manager implements the Topology Manager Interface
	// and is consulted to make Topology aware resource alignments
	GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint

	// TopologyManager HintProvider provider indicates the Device Manager implements the Topology Manager Interface
	// and is consulted to make Topology aware resource alignments per Pod
	GetPodTopologyHints(pod *v1.Pod) map[string][]topologymanager.TopologyHint

	// GetCDIAnnotations checks whether we have cached resource
	// for the passed-in <pod, container> and returns its container annotations or
	// empty map if resource is not cached
	GetCDIAnnotations(pod *v1.Pod, container *v1.Container) []kubecontainer.Annotation

	// UnprepareResources calls NodeUnprepareResource GRPC from DRA plugin to unprepare pod resources
	UnprepareResources(pod *v1.Pod) error
}

const DRACheckpointDir = "/var/lib/kubelet/dra-plugins"
