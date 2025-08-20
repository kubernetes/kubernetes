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
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources/extended"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

var _ framework.PreFilterPlugin = &Fit{}
var _ framework.FilterPlugin = &Fit{}
var _ framework.EnqueueExtensions = &Fit{}
var _ framework.PreScorePlugin = &Fit{}
var _ framework.ScorePlugin = &Fit{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.NodeResourcesFit

	// preFilterStateKey is the key in CycleState to NodeResourcesFit pre-computed data.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + Name

	// preScoreStateKey is the key in CycleState to NodeResourcesFit pre-computed data for Scoring.
	preScoreStateKey = "PreScore" + Name
)

// nodeResourceStrategyTypeMap maps strategy to scorer implementation
var nodeResourceStrategyTypeMap = map[config.ScoringStrategyType]scorer{
	config.LeastAllocated: func(args *config.NodeResourcesFitArgs) *resourceAllocationScorer {
		resources := args.ScoringStrategy.Resources
		return &resourceAllocationScorer{
			Name:      string(config.LeastAllocated),
			scorer:    leastResourceScorer(resources),
			resources: resources,
		}
	},
	config.MostAllocated: func(args *config.NodeResourcesFitArgs) *resourceAllocationScorer {
		resources := args.ScoringStrategy.Resources
		return &resourceAllocationScorer{
			Name:      string(config.MostAllocated),
			scorer:    mostResourceScorer(resources),
			resources: resources,
		}
	},
	config.RequestedToCapacityRatio: func(args *config.NodeResourcesFitArgs) *resourceAllocationScorer {
		resources := args.ScoringStrategy.Resources
		return &resourceAllocationScorer{
			Name:      string(config.RequestedToCapacityRatio),
			scorer:    requestedToCapacityRatioScorer(resources, args.ScoringStrategy.RequestedToCapacityRatio.Shape),
			resources: resources,
		}
	},
}

// Fit is a plugin that checks if a node has sufficient resources.
type Fit struct {
	ignoredResources                sets.Set[string]
	ignoredResourceGroups           sets.Set[string]
	enableInPlacePodVerticalScaling bool
	enableSidecarContainers         bool
	enableSchedulingQueueHint       bool
	enablePodLevelResources         bool
	enableDRAExtendedResource       bool
	handle                          framework.Handle
	resourceAllocationScorer
}

// ScoreExtensions of the Score plugin.
func (f *Fit) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	framework.Resource
	// resourceToDeviceClass holds the mapping of extended resource to device class name.
	resourceToDeviceClass map[v1.ResourceName]string
}

// Clone the prefilter state.
func (s *preFilterState) Clone() fwk.StateData {
	return s
}

// preScoreState computed at PreScore and used at Score.
type preScoreState struct {
	// podRequests have the same order as the resources defined in NodeResourcesBalancedAllocationArgs.Resources,
	// same for other place we store a list like that.
	podRequests []int64
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *preScoreState) Clone() fwk.StateData {
	return s
}

// PreScore calculates incoming pod's resource requests and writes them to the cycle state used.
func (f *Fit) PreScore(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status {
	state := &preScoreState{
		podRequests: f.calculatePodResourceRequestList(pod, f.resources),
	}
	cycleState.Write(preScoreStateKey, state)
	return nil
}

func getPreScoreState(cycleState fwk.CycleState) (*preScoreState, error) {
	c, err := cycleState.Read(preScoreStateKey)
	if err != nil {
		return nil, fmt.Errorf("reading %q from cycleState: %w", preScoreStateKey, err)
	}

	s, ok := c.(*preScoreState)
	if !ok {
		return nil, fmt.Errorf("invalid PreScore state, got type %T", c)
	}
	return s, nil
}

// Name returns name of the plugin. It is used in logs, etc.
func (f *Fit) Name() string {
	return Name
}

// NewFit initializes a new plugin and returns it.
func NewFit(_ context.Context, plArgs runtime.Object, h framework.Handle, fts feature.Features) (framework.Plugin, error) {
	args, ok := plArgs.(*config.NodeResourcesFitArgs)
	if !ok {
		return nil, fmt.Errorf("want args to be of type NodeResourcesFitArgs, got %T", plArgs)
	}
	if err := validation.ValidateNodeResourcesFitArgs(nil, args); err != nil {
		return nil, err
	}

	if args.ScoringStrategy == nil {
		return nil, fmt.Errorf("scoring strategy not specified")
	}

	strategy := args.ScoringStrategy.Type
	scorePlugin, exists := nodeResourceStrategyTypeMap[strategy]
	if !exists {
		return nil, fmt.Errorf("scoring strategy %s is not supported", strategy)
	}

	return &Fit{
		ignoredResources:                sets.New(args.IgnoredResources...),
		ignoredResourceGroups:           sets.New(args.IgnoredResourceGroups...),
		enableInPlacePodVerticalScaling: fts.EnableInPlacePodVerticalScaling,
		enableSidecarContainers:         fts.EnableSidecarContainers,
		enableSchedulingQueueHint:       fts.EnableSchedulingQueueHint,
		handle:                          h,
		enablePodLevelResources:         fts.EnablePodLevelResources,
		enableDRAExtendedResource:       fts.EnableDRAExtendedResource,
		resourceAllocationScorer:        *scorePlugin(args),
	}, nil
}

type ResourceRequestsOptions struct {
	EnablePodLevelResources   bool
	EnableDRAExtendedResource bool
}

// computePodResourceRequest returns a framework.Resource that covers the largest
// width in each resource dimension. Because init-containers run sequentially, we collect
// the max in each dimension iteratively. In contrast, we sum the resource vectors for
// regular containers since they run simultaneously.
//
// # The resources defined for Overhead should be added to the calculated Resource request sum
//
// Example:
//
// Pod:
//
//	InitContainers
//	  IC1:
//	    CPU: 2
//	    Memory: 1G
//	  IC2:
//	    CPU: 2
//	    Memory: 3G
//	Containers
//	  C1:
//	    CPU: 2
//	    Memory: 1G
//	  C2:
//	    CPU: 1
//	    Memory: 1G
//
// Result: CPU: 3, Memory: 3G
// TODO(ndixita): modify computePodResourceRequest to accept opts of type
// ResourceRequestOptions as the second parameter.
func computePodResourceRequest(pod *v1.Pod, opts ResourceRequestsOptions) *preFilterState {
	// pod hasn't scheduled yet so we don't need to worry about InPlacePodVerticalScalingEnabled
	reqs := resource.PodRequests(pod, resource.PodResourcesOptions{
		// SkipPodLevelResources is set to false when PodLevelResources feature is enabled.
		SkipPodLevelResources: !opts.EnablePodLevelResources,
	})
	result := &preFilterState{}
	result.SetMaxResource(reqs)
	return result
}

// withDeviceClass adds resource to device class mapping to preFilterState.
func withDeviceClass(result *preFilterState, draManager framework.SharedDRAManager) *fwk.Status {
	hasExtendedResource := false
	for rName, rQuant := range result.ScalarResources {
		// Skip in case request quantity is zero
		if rQuant == 0 {
			continue
		}

		if v1helper.IsExtendedResourceName(rName) {
			hasExtendedResource = true
			break
		}
	}
	if hasExtendedResource {
		resourceToDeviceClass, err := extended.DeviceClassMapping(draManager)
		if err != nil {
			return fwk.AsStatus(err)
		}
		result.resourceToDeviceClass = resourceToDeviceClass
		if len(resourceToDeviceClass) == 0 {
			// ensure it is empty map, not nil.
			result.resourceToDeviceClass = make(map[v1.ResourceName]string, 0)
		}
	}
	return nil
}

// PreFilter invoked at the prefilter extension point.
func (f *Fit) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*framework.PreFilterResult, *fwk.Status) {
	if !f.enableSidecarContainers && hasRestartableInitContainer(pod) {
		// Scheduler will calculate resources usage for a Pod containing
		// restartable init containers that will be equal or more than kubelet will
		// require to run the Pod. So there will be no overbooking. However, to
		// avoid the inconsistency in resource calculation between the scheduler
		// and the older (before v1.28) kubelet, make the Pod unschedulable.
		return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "Pod has a restartable init container and the SidecarContainers feature is disabled")
	}
	result := computePodResourceRequest(pod, ResourceRequestsOptions{EnablePodLevelResources: f.enablePodLevelResources})
	if f.enableDRAExtendedResource {
		if err := withDeviceClass(result, f.handle.SharedDRAManager()); err != nil {
			return nil, err
		}
	}

	cycleState.Write(preFilterStateKey, result)
	return nil, nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (f *Fit) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func getPreFilterState(cycleState fwk.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// preFilterState doesn't exist, likely PreFilter wasn't invoked.
		return nil, fmt.Errorf("error reading %q from cycleState: %w", preFilterStateKey, err)
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to NodeResourcesFit.preFilterState error", c)
	}
	return s, nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (f *Fit) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	podActionType := fwk.Delete
	if f.enableInPlacePodVerticalScaling {
		// If InPlacePodVerticalScaling (KEP 1287) is enabled, then UpdatePodScaleDown event should be registered
		// for this plugin since a Pod update may free up resources that make other Pods schedulable.
		podActionType |= fwk.UpdatePodScaleDown
	}

	// A note about UpdateNodeTaint/UpdateNodeLabel event:
	// Ideally, it's supposed to register only Add | UpdateNodeAllocatable because the only resource update could change the node resource fit plugin's result.
	// But, we may miss Node/Add event due to preCheck, and we decided to register UpdateNodeTaint | UpdateNodeLabel for all plugins registering Node/Add.
	// See: https://github.com/kubernetes/kubernetes/issues/109437
	nodeActionType := fwk.Add | fwk.UpdateNodeAllocatable | fwk.UpdateNodeTaint | fwk.UpdateNodeLabel
	if f.enableSchedulingQueueHint {
		// preCheck is not used when QHint is enabled, and hence Update event isn't necessary.
		nodeActionType = fwk.Add | fwk.UpdateNodeAllocatable
	}

	return []fwk.ClusterEventWithHint{
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: podActionType}, QueueingHintFn: f.isSchedulableAfterPodEvent},
		{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: nodeActionType}, QueueingHintFn: f.isSchedulableAfterNodeChange},
	}, nil
}

// isSchedulableAfterPodEvent is invoked whenever a pod deleted or scaled down. It checks whether
// that change made a previously unschedulable pod schedulable.
func (f *Fit) isSchedulableAfterPodEvent(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	originalPod, modifiedPod, err := schedutil.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if modifiedPod == nil {
		if originalPod.Spec.NodeName == "" && originalPod.Status.NominatedNodeName == "" {
			logger.V(5).Info("the deleted pod was unscheduled and it wouldn't make the unscheduled pod schedulable", "pod", klog.KObj(pod), "deletedPod", klog.KObj(originalPod))
			return fwk.QueueSkip, nil
		}

		// any deletion event to a scheduled pod could make the unscheduled pod schedulable.
		logger.V(5).Info("another scheduled pod was deleted, and it may make the unscheduled pod schedulable", "pod", klog.KObj(pod), "deletedPod", klog.KObj(originalPod))
		return fwk.Queue, nil
	}

	if !f.enableInPlacePodVerticalScaling {
		// If InPlacePodVerticalScaling (KEP 1287) is disabled, the pod scale down event cannot free up any resources.
		logger.V(5).Info("another pod was modified, but InPlacePodVerticalScaling is disabled, so it doesn't make the unscheduled pod schedulable", "pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
		return fwk.QueueSkip, nil
	}

	if !f.isSchedulableAfterPodScaleDown(pod, originalPod, modifiedPod) {
		if loggerV := logger.V(10); loggerV.Enabled() {
			// Log more information.
			loggerV.Info("pod got scaled down, but the modification isn't related to the resource requests of the target pod", "pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod), "diff", diff.Diff(originalPod, modifiedPod))
		} else {
			logger.V(5).Info("pod got scaled down, but the modification isn't related to the resource requests of the target pod", "pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
		}
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("another scheduled pod or the target pod itself got scaled down, and it may make the unscheduled pod schedulable", "pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
	return fwk.Queue, nil
}

// isSchedulableAfterPodScaleDown checks whether the scale down event may make the target pod schedulable. Specifically:
// - Returns true when the update event is for the target pod itself.
// - Returns true when the update event shows a scheduled pod's resource request that the target pod also requests got reduced.
func (f *Fit) isSchedulableAfterPodScaleDown(targetPod, originalPod, modifiedPod *v1.Pod) bool {
	if modifiedPod.UID == targetPod.UID {
		// If the scaling down event is for targetPod, it would make targetPod schedulable.
		return true
	}

	if modifiedPod.Spec.NodeName == "" {
		// If the update event is for a unscheduled Pod,
		// it wouldn't make targetPod schedulable.
		return false
	}

	// the other pod was scheduled, so modification or deletion may free up some resources.
	originalMaxResourceReq, modifiedMaxResourceReq := &framework.Resource{}, &framework.Resource{}
	originalMaxResourceReq.SetMaxResource(resource.PodRequests(originalPod, resource.PodResourcesOptions{UseStatusResources: f.enableInPlacePodVerticalScaling}))
	modifiedMaxResourceReq.SetMaxResource(resource.PodRequests(modifiedPod, resource.PodResourcesOptions{UseStatusResources: f.enableInPlacePodVerticalScaling}))

	// check whether the resource request of the modified pod is less than the original pod.
	podRequests := resource.PodRequests(targetPod, resource.PodResourcesOptions{UseStatusResources: f.enableInPlacePodVerticalScaling})
	for rName, rValue := range podRequests {
		if rValue.IsZero() {
			// We only care about the resources requested by the pod we are trying to schedule.
			continue
		}
		switch rName {
		case v1.ResourceCPU:
			if originalMaxResourceReq.MilliCPU > modifiedMaxResourceReq.MilliCPU {
				return true
			}
		case v1.ResourceMemory:
			if originalMaxResourceReq.Memory > modifiedMaxResourceReq.Memory {
				return true
			}
		case v1.ResourceEphemeralStorage:
			if originalMaxResourceReq.EphemeralStorage > modifiedMaxResourceReq.EphemeralStorage {
				return true
			}
		default:
			if schedutil.IsScalarResourceName(rName) && originalMaxResourceReq.ScalarResources[rName] > modifiedMaxResourceReq.ScalarResources[rName] {
				return true
			}
		}
	}
	return false
}

// isSchedulableAfterNodeChange is invoked whenever a node added or changed. It checks whether
// that change could make a previously unschedulable pod schedulable.
func (f *Fit) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	originalNode, modifiedNode, err := schedutil.As[*v1.Node](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}
	// Leaving in the queue, since the pod won't fit into the modified node anyway.
	if !isFit(pod, modifiedNode, ResourceRequestsOptions{EnablePodLevelResources: f.enablePodLevelResources, EnableDRAExtendedResource: f.enableDRAExtendedResource}) {
		logger.V(5).Info("node was created or updated, but it doesn't have enough resource(s) to accommodate this pod", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
		return fwk.QueueSkip, nil
	}
	// The pod will fit, so since it's add, unblock scheduling.
	if originalNode == nil {
		logger.V(5).Info("node was added and it might fit the pod's resource requests", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
		return fwk.Queue, nil
	}
	// The pod will fit, but since there was no increase in available resources, the change won't make the pod schedulable.
	if !haveAnyRequestedResourcesIncreased(pod, originalNode, modifiedNode, ResourceRequestsOptions{EnablePodLevelResources: f.enablePodLevelResources, EnableDRAExtendedResource: f.enableDRAExtendedResource}) {
		logger.V(5).Info("node was updated, but haven't changed the pod's resource requestments fit assessment", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("node was updated, and may now fit the pod's resource requests", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
	return fwk.Queue, nil
}

// haveAnyRequestedResourcesIncreased returns true if any of the resources requested by the pod have increased or if allowed pod number increased.
func haveAnyRequestedResourcesIncreased(pod *v1.Pod, originalNode, modifiedNode *v1.Node, opts ResourceRequestsOptions) bool {
	podRequest := computePodResourceRequest(pod, opts)
	originalNodeInfo := framework.NewNodeInfo()
	originalNodeInfo.SetNode(originalNode)
	modifiedNodeInfo := framework.NewNodeInfo()
	modifiedNodeInfo.SetNode(modifiedNode)

	if modifiedNodeInfo.Allocatable.GetAllowedPodNumber() > originalNodeInfo.Allocatable.GetAllowedPodNumber() {
		return true
	}

	if podRequest.MilliCPU == 0 &&
		podRequest.Memory == 0 &&
		podRequest.EphemeralStorage == 0 &&
		len(podRequest.ScalarResources) == 0 {
		return false
	}

	if (podRequest.MilliCPU > 0 && modifiedNodeInfo.Allocatable.GetMilliCPU() > originalNodeInfo.Allocatable.GetMilliCPU()) ||
		(podRequest.Memory > 0 && modifiedNodeInfo.Allocatable.GetMemory() > originalNodeInfo.Allocatable.GetMemory()) ||
		(podRequest.EphemeralStorage > 0 && modifiedNodeInfo.Allocatable.GetEphemeralStorage() > originalNodeInfo.Allocatable.GetEphemeralStorage()) {
		return true
	}

	for rName, rQuant := range podRequest.ScalarResources {
		// Skip in case request quantity is zero
		if rQuant == 0 {
			continue
		}

		if modifiedNodeInfo.Allocatable.GetScalarResources()[rName] > originalNodeInfo.Allocatable.GetScalarResources()[rName] {
			return true
		}

		if opts.EnableDRAExtendedResource {
			_, okScalar := modifiedNodeInfo.GetAllocatable().GetScalarResources()[rName]
			_, okDynamic := podRequest.resourceToDeviceClass[rName]

			if (okDynamic || podRequest.resourceToDeviceClass == nil) && !okScalar {
				// The extended resource request matches a device class or no device class mapping
				// provided and it is not in the node's Allocatable (i.e. it is not provided
				// by the node's device plugin), then leave it to the dynamicresources
				// plugin to evaluate whether it can be satisfy by DRA resources.
				return true
			}
		}
	}
	return false
}

// isFit checks if the pod fits the node. If the node is nil, it returns false.
// It constructs a fake NodeInfo object for the node and checks if the pod fits the node.
func isFit(pod *v1.Pod, node *v1.Node, opts ResourceRequestsOptions) bool {
	if node == nil {
		return false
	}
	nodeInfo := framework.NewNodeInfo()
	nodeInfo.SetNode(node)
	return len(Fits(pod, nodeInfo, opts)) == 0
}

// Filter invoked at the filter extension point.
// Checks if a node has sufficient resources, such as cpu, memory, gpu, opaque int resources etc to run a pod.
// It returns a list of insufficient resources, if empty, then the node has all the resources requested by the pod.
func (f *Fit) Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}

	insufficientResources := fitsRequest(s, nodeInfo, f.ignoredResources, f.ignoredResourceGroups, ResourceRequestsOptions{
		EnablePodLevelResources:   f.enablePodLevelResources,
		EnableDRAExtendedResource: f.enableDRAExtendedResource})

	if len(insufficientResources) != 0 {
		// We will keep all failure reasons.
		failureReasons := make([]string, 0, len(insufficientResources))
		statusCode := fwk.Unschedulable
		for i := range insufficientResources {
			failureReasons = append(failureReasons, insufficientResources[i].Reason)

			if insufficientResources[i].Unresolvable {
				statusCode = fwk.UnschedulableAndUnresolvable
			}
		}

		return fwk.NewStatus(statusCode, failureReasons...)
	}
	return nil
}

func hasRestartableInitContainer(pod *v1.Pod) bool {
	for _, c := range pod.Spec.InitContainers {
		if c.RestartPolicy != nil && *c.RestartPolicy == v1.ContainerRestartPolicyAlways {
			return true
		}
	}
	return false
}

// InsufficientResource describes what kind of resource limit is hit and caused the pod to not fit the node.
type InsufficientResource struct {
	ResourceName v1.ResourceName
	// We explicitly have a parameter for reason to avoid formatting a message on the fly
	// for common resources, which is expensive for cluster autoscaler simulations.
	Reason    string
	Requested int64
	Used      int64
	Capacity  int64
	// Unresolvable indicates whether this node could be schedulable for the pod by the preemption,
	// which is determined by comparing the node's size and the pod's request.
	Unresolvable bool
}

// Fits checks if node have enough resources to host the pod.
func Fits(pod *v1.Pod, nodeInfo fwk.NodeInfo, opts ResourceRequestsOptions) []InsufficientResource {
	return fitsRequest(computePodResourceRequest(pod, opts), nodeInfo, nil, nil, opts)
}

func fitsRequest(podRequest *preFilterState, nodeInfo fwk.NodeInfo, ignoredExtendedResources, ignoredResourceGroups sets.Set[string], opts ResourceRequestsOptions) []InsufficientResource {
	insufficientResources := make([]InsufficientResource, 0, 4)

	allowedPodNumber := nodeInfo.GetAllocatable().GetAllowedPodNumber()
	if len(nodeInfo.GetPods())+1 > allowedPodNumber {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourcePods,
			Reason:       "Too many pods",
			Requested:    1,
			Used:         int64(len(nodeInfo.GetPods())),
			Capacity:     int64(allowedPodNumber),
		})
	}

	if podRequest.MilliCPU == 0 &&
		podRequest.Memory == 0 &&
		podRequest.EphemeralStorage == 0 &&
		len(podRequest.ScalarResources) == 0 {
		return insufficientResources
	}

	if podRequest.MilliCPU > 0 && podRequest.MilliCPU > (nodeInfo.GetAllocatable().GetMilliCPU()-nodeInfo.GetRequested().GetMilliCPU()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceCPU,
			Reason:       "Insufficient cpu",
			Requested:    podRequest.MilliCPU,
			Used:         nodeInfo.GetRequested().GetMilliCPU(),
			Capacity:     nodeInfo.GetAllocatable().GetMilliCPU(),
			Unresolvable: podRequest.MilliCPU > nodeInfo.GetAllocatable().GetMilliCPU(),
		})
	}
	if podRequest.Memory > 0 && podRequest.Memory > (nodeInfo.GetAllocatable().GetMemory()-nodeInfo.GetRequested().GetMemory()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceMemory,
			Reason:       "Insufficient memory",
			Requested:    podRequest.Memory,
			Used:         nodeInfo.GetRequested().GetMemory(),
			Capacity:     nodeInfo.GetAllocatable().GetMemory(),
			Unresolvable: podRequest.Memory > nodeInfo.GetAllocatable().GetMemory(),
		})
	}
	if podRequest.EphemeralStorage > 0 &&
		podRequest.EphemeralStorage > (nodeInfo.GetAllocatable().GetEphemeralStorage()-nodeInfo.GetRequested().GetEphemeralStorage()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceEphemeralStorage,
			Reason:       "Insufficient ephemeral-storage",
			Requested:    podRequest.EphemeralStorage,
			Used:         nodeInfo.GetRequested().GetEphemeralStorage(),
			Capacity:     nodeInfo.GetAllocatable().GetEphemeralStorage(),
			Unresolvable: podRequest.GetEphemeralStorage() > nodeInfo.GetAllocatable().GetEphemeralStorage(),
		})
	}

	for rName, rQuant := range podRequest.ScalarResources {
		// Skip in case request quantity is zero
		if rQuant == 0 {
			continue
		}

		if v1helper.IsExtendedResourceName(rName) {
			// If this resource is one of the extended resources that should be ignored, we will skip checking it.
			// rName is guaranteed to have a slash due to API validation.
			var rNamePrefix string
			if ignoredResourceGroups.Len() > 0 {
				rNamePrefix = strings.Split(string(rName), "/")[0]
			}
			if ignoredExtendedResources.Has(string(rName)) || ignoredResourceGroups.Has(rNamePrefix) {
				continue
			}
		}

		if opts.EnableDRAExtendedResource {
			_, okScalar := nodeInfo.GetAllocatable().GetScalarResources()[rName]
			_, okDynamic := podRequest.resourceToDeviceClass[rName]

			if (okDynamic || podRequest.resourceToDeviceClass == nil) && !okScalar {
				// The extended resource request matches a device class or no device class mapping
				// provided and it is not in the node's Allocatable (i.e. it is not provided
				// by the node's device plugin), then leave it to the dynamicresources
				// plugin to evaluate whether it can be satisfy by DRA resources.
				continue
			}
		}
		if rQuant > (nodeInfo.GetAllocatable().GetScalarResources()[rName] - nodeInfo.GetRequested().GetScalarResources()[rName]) {
			insufficientResources = append(insufficientResources, InsufficientResource{
				ResourceName: rName,
				Reason:       fmt.Sprintf("Insufficient %v", rName),
				Requested:    podRequest.ScalarResources[rName],
				Used:         nodeInfo.GetRequested().GetScalarResources()[rName],
				Capacity:     nodeInfo.GetAllocatable().GetScalarResources()[rName],
				Unresolvable: rQuant > nodeInfo.GetAllocatable().GetScalarResources()[rName],
			})
		}
	}

	return insufficientResources
}

// Score invoked at the Score extension point.
func (f *Fit) Score(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	s, err := getPreScoreState(state)
	if err != nil {
		s = &preScoreState{
			podRequests: f.calculatePodResourceRequestList(pod, f.resources),
		}
	}

	return f.score(ctx, pod, nodeInfo, s.podRequests)
}
