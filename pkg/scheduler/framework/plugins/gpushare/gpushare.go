/*
Copyright 2024 The Kubernetes Authors.

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

package gpushare

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.GPUShare

	// GPUShareResourceName is the extended resource name for fractional GPU allocation
	GPUShareResourceName = v1.ResourceName("gpushare.com/vgpu")
)

var _ framework.FilterPlugin = &GPUShare{}
var _ framework.ScorePlugin = &GPUShare{}
var _ framework.EnqueueExtensions = &GPUShare{}

// GPUShare is a plugin that schedules pods requesting fractional GPU resources.
type GPUShare struct {
	handle framework.Handle
}

// Name returns name of the plugin.
func (pl *GPUShare) Name() string {
	return Name
}

// ScoreExtensions of the Score plugin.
func (pl *GPUShare) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// Filter checks if the node has enough fractional GPU resources.
// We allow multiple pods to fit on the same node if their total fractional GPU requests
// do not exceed the node's allocatable gpushare.com/vgpu capacity.
func (pl *GPUShare) Filter(ctx context.Context, cycleState framework.CycleState, pod *v1.Pod, nodeInfo framework.NodeInfo) *framework.Status {
	// Calculate the pod's fractional GPU request
	var podGPURequest int64
	var maxInitGPURequest int64

	for i := range pod.Spec.InitContainers {
		if val, ok := pod.Spec.InitContainers[i].Resources.Requests[GPUShareResourceName]; ok {
			if pod.Spec.InitContainers[i].RestartPolicy != nil && *pod.Spec.InitContainers[i].RestartPolicy == v1.ContainerRestartPolicyAlways {
				podGPURequest += val.Value()
			} else if val.Value() > maxInitGPURequest {
				maxInitGPURequest = val.Value()
			}
		}
	}

	for i := range pod.Spec.Containers {
		if val, ok := pod.Spec.Containers[i].Resources.Requests[GPUShareResourceName]; ok {
			podGPURequest += val.Value()
		}
	}

	for i := range pod.Spec.EphemeralContainers {
		if val, ok := pod.Spec.EphemeralContainers[i].Resources.Requests[GPUShareResourceName]; ok {
			podGPURequest += val.Value()
		}
	}

	if maxInitGPURequest > podGPURequest {
		podGPURequest = maxInitGPURequest
	}

	if podGPURequest == 0 {
		return nil
	}

	allocatable := nodeInfo.Node().Status.Allocatable
	nodeCapacity, ok := allocatable[GPUShareResourceName]
	if !ok {
		return framework.NewStatus(framework.Unschedulable, "Node does not have GPUShare resources")
	}

	// Calculate used GPU resources on this node
	var usedGPU int64
	for _, p := range nodeInfo.GetPods() {
		for i := range p.GetPod().Spec.Containers {
			if val, exists := p.GetPod().Spec.Containers[i].Resources.Requests[GPUShareResourceName]; exists {
				usedGPU += val.Value()
			}
		}
	}

	if nodeCapacity.Value()-usedGPU < podGPURequest {
		return framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Insufficient GPUShare: requested %d, available %d", podGPURequest, nodeCapacity.Value()-usedGPU))
	}

	return nil
}

// Score gives higher score to nodes that already have GPU pods,
// effectively bin-packing GPU workloads to save completely free GPUs for large workloads.
func (pl *GPUShare) Score(ctx context.Context, state framework.CycleState, pod *v1.Pod, nodeInfo framework.NodeInfo) (int64, *framework.Status) {
	node := nodeInfo.Node()
	if node == nil {
		return 0, framework.NewStatus(framework.Error, "node not found")
	}

	var podGPURequest int64
	var maxInitGPURequest int64

	for i := range pod.Spec.InitContainers {
		if val, ok := pod.Spec.InitContainers[i].Resources.Requests[GPUShareResourceName]; ok {
			if pod.Spec.InitContainers[i].RestartPolicy != nil && *pod.Spec.InitContainers[i].RestartPolicy == v1.ContainerRestartPolicyAlways {
				podGPURequest += val.Value()
			} else if val.Value() > maxInitGPURequest {
				maxInitGPURequest = val.Value()
			}
		}
	}

	for i := range pod.Spec.Containers {
		if val, ok := pod.Spec.Containers[i].Resources.Requests[GPUShareResourceName]; ok {
			podGPURequest += val.Value()
		}
	}

	for i := range pod.Spec.EphemeralContainers {
		if val, ok := pod.Spec.EphemeralContainers[i].Resources.Requests[GPUShareResourceName]; ok {
			podGPURequest += val.Value()
		}
	}

	if maxInitGPURequest > podGPURequest {
		podGPURequest = maxInitGPURequest
	}

	if podGPURequest == 0 {
		return framework.MinNodeScore, nil
	}

	allocatable := node.Status.Allocatable
	nodeCapacity, ok := allocatable[GPUShareResourceName]
	if !ok || nodeCapacity.Value() == 0 {
		return framework.MinNodeScore, nil
	}

	// Calculate used GPU resources on this node
	var usedGPU int64
	for _, p := range nodeInfo.GetPods() {
		for i := range p.GetPod().Spec.Containers {
			if val, exists := p.GetPod().Spec.Containers[i].Resources.Requests[GPUShareResourceName]; exists {
				usedGPU += val.Value()
			}
		}
	}

	// Formula for bin-packing: ((used + request) / capacity) * MaxNodeScore
	fraction := float64(usedGPU+podGPURequest) / float64(nodeCapacity.Value())
	if fraction > 1.0 {
		fraction = 1.0 // Should not happen if Filter passed, but just in case
	}

	score := int64(fraction * float64(framework.MaxNodeScore))
	return score, nil
}

// EventsToRegister returns the possible events that may make a Pod failed by this plugin schedulable.
func (pl *GPUShare) EventsToRegister(_ context.Context) ([]framework.ClusterEventWithHint, error) {
	// Add Node, Update Node Allocatable, and Pod Deletion could alter the GPUShare fit decision.
	return []framework.ClusterEventWithHint{
		{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeAllocatable}, QueueingHintFn: nil},
		{Event: framework.ClusterEvent{Resource: framework.Pod, ActionType: framework.Delete}, QueueingHintFn: nil},
	}, nil
}

// New initializes a new GPUShare plugin and returns it.
func New(_ context.Context, _ runtime.Object, h framework.Handle, _ feature.Features) (framework.Plugin, error) {
	return &GPUShare{handle: h}, nil
}
