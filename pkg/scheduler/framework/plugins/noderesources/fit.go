/*
Copyright 2019 The Kubernetes Authors.

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

package noderesources

import (
	"context"
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var _ framework.PreFilterPlugin = &Fit{}
var _ framework.FilterPlugin = &Fit{}

const (
	// FitName is the name of the plugin used in the plugin registry and configurations.
	FitName = "NodeResourcesFit"

	// preFilterStateKey is the key in CycleState to NodeResourcesFit pre-computed data.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + FitName
)

// Fit is a plugin that checks if a node has sufficient resources.
type Fit struct {
	ignoredResources sets.String
}

// FitArgs holds the args that are used to configure the plugin.
type FitArgs struct {
	// IgnoredResources is the list of resources that NodeResources fit filter
	// should ignore.
	IgnoredResources []string `json:"IgnoredResources,omitempty"`
}

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	schedulernodeinfo.Resource
}

// Clone the prefilter state.
func (s *preFilterState) Clone() framework.StateData {
	return s
}

// Name returns name of the plugin. It is used in logs, etc.
func (f *Fit) Name() string {
	return FitName
}

// computePodResourceRequest returns a schedulernodeinfo.Resource that covers the largest
// width in each resource dimension. Because init-containers run sequentially, we collect
// the max in each dimension iteratively. In contrast, we sum the resource vectors for
// regular containers since they run simultaneously.
//
// If Pod Overhead is specified and the feature gate is set, the resources defined for Overhead
// are added to the calculated Resource request sum
//
// Example:
//
// Pod:
//   InitContainers
//     IC1:
//       CPU: 2
//       Memory: 1G
//     IC2:
//       CPU: 2
//       Memory: 3G
//   Containers
//     C1:
//       CPU: 2
//       Memory: 1G
//     C2:
//       CPU: 1
//       Memory: 1G
//
// Result: CPU: 3, Memory: 3G
func computePodResourceRequest(pod *v1.Pod) *preFilterState {
	result := &preFilterState{}
	for _, container := range pod.Spec.Containers {
		result.Add(container.Resources.Requests)
	}

	// take max_resource(sum_pod, any_init_container)
	for _, container := range pod.Spec.InitContainers {
		result.SetMaxResource(container.Resources.Requests)
	}

	// If Overhead is being utilized, add to the total requests for the pod
	if pod.Spec.Overhead != nil && utilfeature.DefaultFeatureGate.Enabled(features.PodOverhead) {
		result.Add(pod.Spec.Overhead)
	}

	return result
}

// PreFilter invoked at the prefilter extension point.
func (f *Fit) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	cycleState.Write(preFilterStateKey, computePodResourceRequest(pod))
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (f *Fit) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func getPreFilterState(cycleState *framework.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// The preFilterState doesn't exist. We ignore the error for now since
		// Filter is able to handle that by computing it again.
		klog.V(5).Infof("Error reading %q from cycleState: %v", preFilterStateKey, err)
		return nil, nil
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to NodeResourcesFit.preFilterState error", c)
	}
	return s, nil
}

// Filter invoked at the filter extension point.
// Checks if a node has sufficient resources, such as cpu, memory, gpu, opaque int resources etc to run a pod.
// It returns a list of insufficient resources, if empty, then the node has all the resources requested by the pod.
func (f *Fit) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *framework.Status {
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	var insufficientResources []InsufficientResource
	if s != nil {
		insufficientResources = fitsRequest(s, nodeInfo, f.ignoredResources)
	} else {
		insufficientResources = Fits(pod, nodeInfo, f.ignoredResources)
	}

	// We will keep all failure reasons.
	var failureReasons []string
	for _, r := range insufficientResources {
		failureReasons = append(failureReasons, getErrReason(r.ResourceName))
	}

	if len(insufficientResources) != 0 {
		return framework.NewStatus(framework.Unschedulable, failureReasons...)
	}
	return nil
}

func getErrReason(rn v1.ResourceName) string {
	return fmt.Sprintf("Insufficient %v", rn)
}

// InsufficientResource describes what kind of resource limit is hit and caused the pod to not fit the node.
type InsufficientResource struct {
	ResourceName v1.ResourceName
	Requested    int64
	Used         int64
	Capacity     int64
}

// Fits checks if node have enough resources to host the pod.
func Fits(pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo, ignoredExtendedResources sets.String) []InsufficientResource {
	return fitsRequest(computePodResourceRequest(pod), nodeInfo, ignoredExtendedResources)
}

func fitsRequest(podRequest *preFilterState, nodeInfo *schedulernodeinfo.NodeInfo, ignoredExtendedResources sets.String) []InsufficientResource {
	var insufficientResources []InsufficientResource

	allowedPodNumber := nodeInfo.AllowedPodNumber()
	if len(nodeInfo.Pods())+1 > allowedPodNumber {
		insufficientResources = append(insufficientResources, InsufficientResource{
			v1.ResourcePods,
			1,
			int64(len(nodeInfo.Pods())),
			int64(allowedPodNumber),
		})
	}

	if ignoredExtendedResources == nil {
		ignoredExtendedResources = sets.NewString()
	}

	if podRequest.MilliCPU == 0 &&
		podRequest.Memory == 0 &&
		podRequest.EphemeralStorage == 0 &&
		len(podRequest.ScalarResources) == 0 {
		return insufficientResources
	}

	allocatable := nodeInfo.AllocatableResource()
	if allocatable.MilliCPU < podRequest.MilliCPU+nodeInfo.RequestedResource().MilliCPU {
		insufficientResources = append(insufficientResources, InsufficientResource{
			v1.ResourceCPU,
			podRequest.MilliCPU,
			nodeInfo.RequestedResource().MilliCPU,
			allocatable.MilliCPU,
		})
	}
	if allocatable.Memory < podRequest.Memory+nodeInfo.RequestedResource().Memory {
		insufficientResources = append(insufficientResources, InsufficientResource{
			v1.ResourceMemory,
			podRequest.Memory,
			nodeInfo.RequestedResource().Memory,
			allocatable.Memory,
		})
	}
	if allocatable.EphemeralStorage < podRequest.EphemeralStorage+nodeInfo.RequestedResource().EphemeralStorage {
		insufficientResources = append(insufficientResources, InsufficientResource{
			v1.ResourceEphemeralStorage,
			podRequest.EphemeralStorage,
			nodeInfo.RequestedResource().EphemeralStorage,
			allocatable.EphemeralStorage,
		})
	}

	for rName, rQuant := range podRequest.ScalarResources {
		if v1helper.IsExtendedResourceName(rName) {
			// If this resource is one of the extended resources that should be
			// ignored, we will skip checking it.
			if ignoredExtendedResources.Has(string(rName)) {
				continue
			}
		}
		if allocatable.ScalarResources[rName] < rQuant+nodeInfo.RequestedResource().ScalarResources[rName] {
			insufficientResources = append(insufficientResources, InsufficientResource{
				rName,
				podRequest.ScalarResources[rName],
				nodeInfo.RequestedResource().ScalarResources[rName],
				allocatable.ScalarResources[rName],
			})
		}
	}

	return insufficientResources
}

// NewFit initializes a new plugin and returns it.
func NewFit(plArgs *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	args := &FitArgs{}
	if err := framework.DecodeInto(plArgs, args); err != nil {
		return nil, err
	}

	fit := &Fit{}
	fit.ignoredResources = sets.NewString(args.IgnoredResources...)
	return fit, nil
}
