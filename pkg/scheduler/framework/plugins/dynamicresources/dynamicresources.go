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

package dynamicresources

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/utils/ptr"
)

const (
	// Name is the name of the plugin used in Registry and configurations.
	Name = names.DynamicResources

	stateKey fwk.StateKey = Name
)

// The state is initialized in PreFilter phase. Because we save the pointer in
// fwk.CycleState, in the later phases we don't need to call Write method
// to update the value
type stateData struct {
	// A copy of all claims for the Pod (i.e. 1:1 match with
	// pod.Spec.ResourceClaims), initially with the status from the start
	// of the scheduling cycle. Each claim instance is read-only because it
	// might come from the informer cache. The instances get replaced when
	// the plugin itself successfully does an Update.
	//
	// When the DRAExtendedResource feature is enabled, the special ResourceClaim
	// which represents extended resource requests is also stored here.
	// The plugin code should treat this field as a black box and only
	// access it via its methods, in particular:
	// - The all method can be used to iterate over all claims, including
	//   the special one.
	// - The allUserClaims method excludes the special one.
	//
	// Empty if the Pod has no claims and no special claim for extended
	// resources backed by DRA, in which case the plugin has no work to do for
	// the Pod.
	claims claimStore

	// draExtendedResource stores data for extended resources backed by DRA.
	// It will remain empty when the DRAExtendedResource feature is disabled.
	draExtendedResource draExtendedResource

	// Allocator handles claims with structured parameters, which is all of them nowadays.
	allocator structured.Allocator

	// mutex must be locked while accessing any of the fields below.
	mutex sync.Mutex

	// The indices of all claims that:
	// - are allocated
	// - were not available on at least one node
	//
	// Set in parallel during Filter, so write access there must be
	// protected by the mutex. Used by PostFilter.
	unavailableClaims sets.Set[int]

	// informationsForClaim has one entry for each claim in claims.
	informationsForClaim []informationForClaim

	// nodeAllocations caches the result of Filter for the nodes, its key is node name.
	nodeAllocations map[string]nodeAllocation
}

func (d *stateData) Clone() fwk.StateData {
	return d
}

type informationForClaim struct {
	// Node selector based on the claim status if allocated.
	availableOnNodes *nodeaffinity.NodeSelector

	// Set by Reserved, published by PreBind, empty if nothing had to be allocated.
	allocation *resourceapi.AllocationResult
}

// nodeAllocation holds the allocation results and extended resource claim per node.
type nodeAllocation struct {
	// allocationResults has the allocation results, matching the order of
	// claims which had to be allocated.
	allocationResults []resourceapi.AllocationResult
	// extendedResourceClaim has the special claim for extended resource backed by DRA
	// created during Filter for the nodes.
	extendedResourceClaim *resourceapi.ResourceClaim
	// containerResourceRequestMappings has the container, extended resource, and device request mappings
	// calculated at the Filter phase, and used at the PreBind phase.
	containerResourceRequestMappings []v1.ContainerExtendedResourceRequest
}

// DynamicResources is a plugin that ensures that ResourceClaims are allocated.
type DynamicResources struct {
	enabled        bool
	fts            feature.Features
	filterTimeout  time.Duration
	bindingTimeout time.Duration
	fh             fwk.Handle
	clientset      kubernetes.Interface
	celCache       *cel.Cache
	draManager     fwk.SharedDRAManager
}

// New initializes a new plugin and returns it.
func New(ctx context.Context, plArgs runtime.Object, fh fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	if !fts.EnableDynamicResourceAllocation {
		// Disabled, won't do anything.
		return &DynamicResources{}, nil
	}

	args, ok := plArgs.(*config.DynamicResourcesArgs)
	if !ok {
		return nil, fmt.Errorf("got args of type %T, want *DynamicResourcesArgs", plArgs)
	}
	if err := validation.ValidateDynamicResourcesArgs(nil, args, fts); err != nil {
		return nil, err
	}

	pl := &DynamicResources{
		enabled:       true,
		fts:           fts,
		filterTimeout: ptr.Deref(args.FilterTimeout, metav1.Duration{}).Duration,
		bindingTimeout: ptr.Deref(
			args.BindingTimeout,
			metav1.Duration{Duration: config.DynamicResourcesBindingTimeoutDefault},
		).Duration,
		fh:        fh,
		clientset: fh.ClientSet(),
		// This is a LRU cache for compiled CEL expressions. The most
		// recent 10 of them get reused across different scheduling
		// cycles.
		celCache:   cel.NewCache(10, cel.Features{EnableConsumableCapacity: fts.EnableDRAConsumableCapacity}),
		draManager: fh.SharedDRAManager(),
	}

	return pl, nil
}

var _ fwk.PreEnqueuePlugin = &DynamicResources{}
var _ fwk.PreFilterPlugin = &DynamicResources{}
var _ fwk.FilterPlugin = &DynamicResources{}
var _ fwk.PostFilterPlugin = &DynamicResources{}
var _ fwk.ScorePlugin = &DynamicResources{}
var _ fwk.ReservePlugin = &DynamicResources{}
var _ fwk.EnqueueExtensions = &DynamicResources{}
var _ fwk.PreBindPlugin = &DynamicResources{}
var _ fwk.SignPlugin = &DynamicResources{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *DynamicResources) Name() string {
	return Name
}

// Because it isn't simple to determine if DRA claims are single host or more complex,
// we exclude any pod with a DRA claim from signatures. We should improve this.
// See https://github.com/kubernetes/kubernetes/issues/134986
func (pl *DynamicResources) SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status) {
	if len(pod.Spec.ResourceClaims) > 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "pods with dra resource claims are not signable")
	}
	return nil, nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *DynamicResources) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	if !pl.enabled {
		return nil, nil
	}
	// A resource might depend on node labels for topology filtering.
	// A new or updated node may make pods schedulable.
	//
	// A note about UpdateNodeTaint event:
	// Ideally, it's supposed to register only Add | UpdateNodeLabel because UpdateNodeTaint will never change the result from this plugin.
	// But, we may miss Node/Add event due to preCheck, and we decided to register UpdateNodeTaint | UpdateNodeLabel for all plugins registering Node/Add.
	// See: https://github.com/kubernetes/kubernetes/issues/109437
	nodeActionType := fwk.Add | fwk.UpdateNodeLabel | fwk.UpdateNodeTaint | fwk.UpdateNodeAllocatable
	if pl.fts.EnableSchedulingQueueHint {
		// When QHint is enabled, the problematic preCheck is already removed, and we can remove UpdateNodeTaint.
		nodeActionType = fwk.Add | fwk.UpdateNodeLabel | fwk.UpdateNodeAllocatable
	}

	events := []fwk.ClusterEventWithHint{
		{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: nodeActionType}},
		// Allocation is tracked in ResourceClaims, so any changes may make the pods schedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.ResourceClaim, ActionType: fwk.Add | fwk.Update}, QueueingHintFn: pl.isSchedulableAfterClaimChange},
		// Adding the ResourceClaim name to the pod status makes pods waiting for their ResourceClaim schedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdatePodGeneratedResourceClaim}, QueueingHintFn: pl.isSchedulableAfterPodChange},
		// A pod might be waiting for a class to get created or modified.
		{Event: fwk.ClusterEvent{Resource: fwk.DeviceClass, ActionType: fwk.Add | fwk.Update}},
		// Adding or updating a ResourceSlice might make a pod schedulable because new resources became available.
		{Event: fwk.ClusterEvent{Resource: fwk.ResourceSlice, ActionType: fwk.Add | fwk.Update}},
	}

	return events, nil
}

// PreEnqueue checks if there are known reasons why a pod currently cannot be
// scheduled. When this fails, one of the registered events can trigger another
// attempt.
func (pl *DynamicResources) PreEnqueue(ctx context.Context, pod *v1.Pod) (status *fwk.Status) {
	if !pl.enabled {
		return nil
	}

	if err := pl.foreachPodResourceClaim(pod, nil); err != nil {
		return statusUnschedulable(klog.FromContext(ctx), err.Error())
	}
	return nil
}

// isSchedulableAfterClaimChange is invoked for add and update claim events reported by
// an informer. It checks whether that change made a previously unschedulable
// pod schedulable. It errs on the side of letting a pod scheduling attempt
// happen. The delete claim event will not invoke it, so newObj will never be nil.
func (pl *DynamicResources) isSchedulableAfterClaimChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	originalClaim, modifiedClaim, err := schedutil.As[*resourceapi.ResourceClaim](oldObj, newObj)
	if err != nil {
		// Shouldn't happen.
		return fwk.Queue, fmt.Errorf("unexpected object in isSchedulableAfterClaimChange: %w", err)
	}

	usesClaim := false
	if err := pl.foreachPodResourceClaim(pod, func(_ string, claim *resourceapi.ResourceClaim) {
		if claim.UID == modifiedClaim.UID {
			usesClaim = true
		}
	}); err != nil {
		// This is not an unexpected error: we know that
		// foreachPodResourceClaim only returns errors for "not
		// schedulable".
		if loggerV := logger.V(6); loggerV.Enabled() {
			owner := metav1.GetControllerOf(modifiedClaim)
			loggerV.Info("pod is not schedulable after resource claim change", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim), "claimOwner", owner, "reason", err.Error())
		}
		return fwk.QueueSkip, nil
	}

	if originalClaim != nil &&
		originalClaim.Status.Allocation != nil &&
		modifiedClaim.Status.Allocation == nil {
		// A claim with structured parameters was deallocated. This might have made
		// resources available for other pods.
		logger.V(6).Info("claim with structured parameters got deallocated", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
		return fwk.Queue, nil
	}

	if !usesClaim {
		// This was not the claim the pod was waiting for.
		logger.V(6).Info("unrelated claim got modified", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
		return fwk.QueueSkip, nil
	}

	if originalClaim == nil {
		logger.V(5).Info("claim for pod got created", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
		return fwk.Queue, nil
	}

	// Modifications may or may not be relevant. If the entire
	// status is as before, then something else must have changed
	// and we don't care. What happens in practice is that the
	// resource driver adds the finalizer.
	if apiequality.Semantic.DeepEqual(&originalClaim.Status, &modifiedClaim.Status) {
		if loggerV := logger.V(7); loggerV.Enabled() {
			// Log more information.
			loggerV.Info("claim for pod got modified where the pod doesn't care", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim), "diff", diff.Diff(originalClaim, modifiedClaim))
		} else {
			logger.V(6).Info("claim for pod got modified where the pod doesn't care", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
		}
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("status of claim for pod got updated", "pod", klog.KObj(pod), "claim", klog.KObj(modifiedClaim))
	return fwk.Queue, nil
}

// isSchedulableAfterPodChange is invoked for update pod events reported by
// an informer. It checks whether that change adds the ResourceClaim(s) that the
// pod has been waiting for.
func (pl *DynamicResources) isSchedulableAfterPodChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, modifiedPod, err := schedutil.As[*v1.Pod](nil, newObj)
	if err != nil {
		// Shouldn't happen.
		return fwk.Queue, fmt.Errorf("unexpected object in isSchedulableAfterClaimChange: %w", err)
	}

	if pod.UID != modifiedPod.UID {
		logger.V(7).Info("pod is not schedulable after change in other pod", "pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
		return fwk.QueueSkip, nil
	}

	if err := pl.foreachPodResourceClaim(modifiedPod, nil); err != nil {
		// This is not an unexpected error: we know that
		// foreachPodResourceClaim only returns errors for "not
		// schedulable".
		logger.V(6).Info("pod is not schedulable after being updated", "pod", klog.KObj(pod))
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("pod got updated and is schedulable", "pod", klog.KObj(pod))
	return fwk.Queue, nil
}

// podResourceClaims returns the ResourceClaims for all pod.Spec.PodResourceClaims.
func (pl *DynamicResources) podResourceClaims(pod *v1.Pod) ([]*resourceapi.ResourceClaim, error) {
	claims := make([]*resourceapi.ResourceClaim, 0, len(pod.Spec.ResourceClaims))
	if err := pl.foreachPodResourceClaim(pod, func(_ string, claim *resourceapi.ResourceClaim) {
		// We store the pointer as returned by the lister. The
		// assumption is that if a claim gets modified while our code
		// runs, the cache will store a new pointer, not mutate the
		// existing object that we point to here.
		claims = append(claims, claim)
	}); err != nil {
		return nil, err
	}
	return claims, nil
}

// foreachPodResourceClaim checks that each ResourceClaim for the pod exists.
// It calls an optional handler for those claims that it finds.
func (pl *DynamicResources) foreachPodResourceClaim(pod *v1.Pod, cb func(podResourceName string, claim *resourceapi.ResourceClaim)) error {
	for _, resource := range pod.Spec.ResourceClaims {
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, &resource)
		if err != nil {
			return err
		}
		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			continue
		}
		claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
		if err != nil {
			return err
		}

		if claim.DeletionTimestamp != nil {
			return fmt.Errorf("resourceclaim %q is being deleted", claim.Name)
		}

		if mustCheckOwner {
			if err := resourceclaim.IsForPod(pod, claim); err != nil {
				return err
			}
		}
		if cb != nil {
			cb(resource.Name, claim)
		}
	}
	return nil
}

// PreFilter invoked at the prefilter extension point to check if pod has all
// immediate claims bound. UnschedulableAndUnresolvable is returned if
// the pod cannot be scheduled at the moment on any node.
func (pl *DynamicResources) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if !pl.enabled {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	logger := klog.FromContext(ctx)

	// If the pod does not reference any claim, we don't need to do
	// anything for it. We just initialize an empty state to record that
	// observation for the other functions. This gets updated below
	// if we get that far.
	s := &stateData{}
	state.Write(stateKey, s)

	userClaims, err := pl.podResourceClaims(pod)
	if err != nil {
		return nil, statusUnschedulable(logger, err.Error())
	}
	logger.V(5).Info("pod resource claims", "pod", klog.KObj(pod), "resourceclaims", klog.KObjSlice(userClaims))
	extendedResourceClaim, status := pl.preFilterExtendedResources(pod, logger, s)
	if status != nil {
		return nil, status
	}
	claims := newClaimStore(userClaims, extendedResourceClaim)

	// This check covers user and extended ResourceClaim.
	if claims.empty() {
		return nil, fwk.NewStatus(fwk.Skip)
	}

	// Counts all claims which the scheduler needs to allocate itself.
	numClaimsToAllocate := 0
	s.informationsForClaim = make([]informationForClaim, claims.len())
	for index, claim := range claims.all() {
		if claim.Status.Allocation != nil &&
			!resourceclaim.CanBeReserved(claim) &&
			!resourceclaim.IsReservedForPod(pod, claim) {
			// Resource is in use. The pod has to wait.
			return nil, statusUnschedulable(logger, "resourceclaim in use", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
		}

		if claim.Status.Allocation != nil {
			if claim.Status.Allocation.NodeSelector != nil {
				nodeSelector, err := nodeaffinity.NewNodeSelector(claim.Status.Allocation.NodeSelector)
				if err != nil {
					return nil, statusError(logger, err)
				}
				s.informationsForClaim[index].availableOnNodes = nodeSelector
			}
		} else {
			numClaimsToAllocate++

			// Allocation in flight? Better wait for that
			// to finish, see inFlightAllocations
			// documentation for details.
			if pl.draManager.ResourceClaims().ClaimHasPendingAllocation(claim.UID) {
				return nil, statusUnschedulable(logger, fmt.Sprintf("resource claim %s is in the process of being allocated", klog.KObj(claim)))
			}

			// Continue without validating the special claim for extended resource backed by DRA.
			// For the claim template, it is not allocated yet at this point, and it does not have a spec.
			// For the claim from prior scheduling cycle, leave it to Filter Phase to validate.
			if claim == extendedResourceClaim {
				continue
			}

			// Check all requests and device classes. If a class
			// does not exist, scheduling cannot proceed, no matter
			// how the claim is being allocated.
			//
			// When using a control plane controller, a class might
			// have a node filter. This is useful for trimming the
			// initial set of potential nodes before we ask the
			// driver(s) for information about the specific pod.
			for _, request := range claim.Spec.Devices.Requests {
				// The requirements differ depending on whether the request has a list of
				// alternative subrequests defined in the firstAvailable field.
				switch {
				case request.Exactly != nil:
					if status := pl.validateDeviceClass(logger, request.Exactly.DeviceClassName, request.Name); status != nil {
						return nil, status
					}
				case len(request.FirstAvailable) > 0:
					if !pl.fts.EnableDRAPrioritizedList {
						return nil, statusUnschedulable(logger, fmt.Sprintf("resource claim %s, request %s: has subrequests, but the DRAPrioritizedList feature is disabled", klog.KObj(claim), request.Name))
					}
					for _, subRequest := range request.FirstAvailable {
						qualRequestName := strings.Join([]string{request.Name, subRequest.Name}, "/")
						if status := pl.validateDeviceClass(logger, subRequest.DeviceClassName, qualRequestName); status != nil {
							return nil, status
						}
					}
				default:
					return nil, statusUnschedulable(logger, fmt.Sprintf("resource claim %s, request %s: unknown request type", klog.KObj(claim), request.Name))
				}
			}
		}
	}

	if numClaimsToAllocate > 0 {
		if loggerV := logger.V(5); loggerV.Enabled() {
			claimsToAllocate := make([]*resourceapi.ResourceClaim, 0, claims.len())
			for _, claim := range claims.toAllocate() {
				claimsToAllocate = append(claimsToAllocate, claim)
			}
			loggerV.Info("Preparing allocation with structured parameters", "pod", klog.KObj(pod), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		}

		// Doing this over and over again for each pod could be avoided
		// by setting the allocator up once and then keeping it up-to-date
		// as changes are observed.
		//
		// But that would cause problems for using the plugin in the
		// Cluster Autoscaler. If this step here turns out to be
		// expensive, we may have to maintain and update state more
		// persistently.
		//
		// Claims (and thus their devices) are treated as "allocated" if they are in the assume cache
		// or currently their allocation is in-flight. This does not change
		// during filtering, so we can determine that once.
		var allocatedState *structured.AllocatedState
		if pl.fts.EnableDRAConsumableCapacity {
			allocatedState, err = pl.draManager.ResourceClaims().GatherAllocatedState()
			if err != nil {
				return nil, statusError(logger, err)
			}
			if allocatedState == nil {
				return nil, statusError(logger, errors.New("nil allocated state"))
			}
		} else {
			allocatedDevices, err := pl.draManager.ResourceClaims().ListAllAllocatedDevices()
			if err != nil {
				return nil, statusError(logger, err)
			}
			allocatedState = &structured.AllocatedState{
				AllocatedDevices:         allocatedDevices,
				AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
				AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
			}
		}
		slices, err := pl.draManager.ResourceSlices().ListWithDeviceTaintRules()
		if err != nil {
			return nil, statusError(logger, err)
		}
		features := AllocatorFeatures(pl.fts)
		allocator, err := structured.NewAllocator(ctx, features, *allocatedState, pl.draManager.DeviceClasses(), slices, pl.celCache)
		if err != nil {
			return nil, statusError(logger, err)
		}
		s.allocator = allocator
		s.nodeAllocations = make(map[string]nodeAllocation)
	}
	s.claims = claims
	return nil, nil
}

func AllocatorFeatures(fts feature.Features) structured.Features {
	return structured.Features{
		AdminAccess:            fts.EnableDRAAdminAccess,
		PrioritizedList:        fts.EnableDRAPrioritizedList,
		PartitionableDevices:   fts.EnableDRAPartitionableDevices,
		DeviceTaints:           fts.EnableDRADeviceTaints,
		DeviceBindingAndStatus: fts.EnableDRADeviceBindingConditions && fts.EnableDRAResourceClaimDeviceStatus,
		ConsumableCapacity:     fts.EnableDRAConsumableCapacity,
	}
}

func (pl *DynamicResources) validateDeviceClass(logger klog.Logger, deviceClassName, requestName string) *fwk.Status {
	if deviceClassName == "" {
		return statusError(logger, fmt.Errorf("request %s: unsupported request type", requestName))
	}

	_, err := pl.draManager.DeviceClasses().Get(deviceClassName)
	if err != nil {
		// If the class cannot be retrieved, allocation cannot proceed.
		if apierrors.IsNotFound(err) {
			// Here we mark the pod as "unschedulable", so it'll sleep in
			// the unscheduleable queue until a DeviceClass event occurs.
			return statusUnschedulable(logger, fmt.Sprintf("request %s: device class %s does not exist", requestName, deviceClassName))
		}
	}
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *DynamicResources) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

func getStateData(cs fwk.CycleState) (*stateData, error) {
	state, err := cs.Read(stateKey)
	if err != nil {
		return nil, err
	}
	s, ok := state.(*stateData)
	if !ok {
		return nil, errors.New("unable to convert state into stateData")
	}
	return s, nil
}

// Filter invoked at the filter extension point.
// It evaluates if a pod can fit due to the resources it requests,
// for both allocated and unallocated claims.
//
// For claims that are bound, then it checks that the node affinity is
// satisfied by the given node.
//
// For claims that are unbound, it checks whether the claim might get allocated
// for the node.
func (pl *DynamicResources) Filter(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if !pl.enabled {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if state.claims.empty() {
		return nil
	}

	logger := klog.FromContext(ctx)
	node := nodeInfo.Node()
	nodeExtendedResourceClaim, containerResourceRequestMappings, status := pl.filterExtendedResources(state, pod, nodeInfo, logger)
	if status != nil {
		return status
	}
	// The pod has no user claim, it may have extended resource claim that is satisfied by device plugin.
	// Then there is nothing left to do for this plugin.
	if nodeExtendedResourceClaim == nil && state.claims.noUserClaim() {
		return nil
	}

	var unavailableClaims []int
	for index, claim := range state.claims.all() {
		logger.V(10).Info("filtering based on resource claims of the pod", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim))

		// This node selector only gets set if the claim is allocated.
		if nodeSelector := state.informationsForClaim[index].availableOnNodes; nodeSelector != nil && !nodeSelector.Match(node) {
			logger.V(5).Info("allocation's node selector does not match", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim))
			unavailableClaims = append(unavailableClaims, index)
			continue
		}

		if claim.Status.Allocation == nil {
			// The claim is not allocated yet, don't have to check
			// anything else.
			continue
		}

		// The claim is allocated, check whether it is ready for binding.
		if pl.fts.EnableDRADeviceBindingConditions && pl.fts.EnableDRAResourceClaimDeviceStatus {
			ready, err := pl.isClaimReadyForBinding(claim)
			// If the claim is not ready yet (ready false, no error) and binding has timed out
			// or binding has failed (err non-nil), then the scheduler should consider deallocating this
			// claim in PostFilter to unblock trying other devices.
			if err != nil {
				logger.V(5).Info("Claim failed binding conditions check", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim), "err", err)
				unavailableClaims = append(unavailableClaims, index)
			} else if !ready && pl.isClaimTimeout(claim) {
				logger.V(5).Info("Claim timed out waiting for binding conditions", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim))
				unavailableClaims = append(unavailableClaims, index)
			}
		}
	}

	// Use allocator to check the node and cache the result in case that the node is picked.
	var allocations []resourceapi.AllocationResult
	if state.allocator != nil {
		allocCtx := ctx
		if loggerV := logger.V(5); loggerV.Enabled() {
			allocCtx = klog.NewContext(allocCtx, klog.LoggerWithValues(logger, "node", klog.KObj(node)))
		}

		// Apply timeout to the operation?
		if pl.fts.EnableDRASchedulerFilterTimeout && pl.filterTimeout > 0 {
			c, cancel := context.WithTimeout(allocCtx, pl.filterTimeout)
			defer cancel()
			allocCtx = c
		}

		// Check which claims need to be allocated.
		//
		// This replaces the special ResourceClaim for extended resources with one
		// matching the node.
		claimsToAllocate := make([]*resourceapi.ResourceClaim, 0, state.claims.len())
		extendedResourceClaim := state.claims.extendedResourceClaim()
		for _, claim := range state.claims.toAllocate() {
			if claim == extendedResourceClaim && nodeExtendedResourceClaim != nil {
				claim = nodeExtendedResourceClaim
			}
			claimsToAllocate = append(claimsToAllocate, claim)
		}
		a, err := state.allocator.Allocate(allocCtx, node, claimsToAllocate)
		switch {
		case errors.Is(err, context.DeadlineExceeded):
			return statusUnschedulable(logger, "timed out trying to allocate devices", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		case errors.Is(err, structured.ErrFailedAllocationOnNode):
			// Not a fatal error, allocation on other nodes may proceed.
			// The error is only surfaced if allocation fails on all nodes.
			return statusUnschedulable(logger, err.Error(), "pod", klog.KObj(pod), "node", klog.KObj(node))
		case ctx.Err() != nil:
			return statusUnschedulable(logger, fmt.Sprintf("asked by caller to stop allocating devices: %v", context.Cause(ctx)), "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		case err != nil:
			// This should only fail if there is something wrong with the claim or class.
			// Return an error to abort scheduling of it.
			//
			// This will cause retries. It would be slightly nicer to mark it as unschedulable
			// *and* abort scheduling. Then only cluster event for updating the claim or class
			// with the broken CEL expression would trigger rescheduling.
			//
			// But we cannot do both. As this shouldn't occur often, aborting like this is
			// better than the more complicated alternative (return Unschedulable here, remember
			// the error, then later raise it again later if needed).
			return statusError(logger, err, "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		}
		// Check for exact length just to be sure. In practice this is all-or-nothing.
		if len(a) != len(claimsToAllocate) {
			return statusUnschedulable(logger, "cannot allocate all claims", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaims", klog.KObjSlice(claimsToAllocate))
		}
		// Reserve uses this information.
		allocations = a
	}

	// Store information in state while holding the mutex.
	if state.allocator != nil || len(unavailableClaims) > 0 {
		state.mutex.Lock()
		defer state.mutex.Unlock()
	}

	if len(unavailableClaims) > 0 {
		// Remember all unavailable claims. This might be observed
		// concurrently, so we have to lock the state before writing.

		if state.unavailableClaims == nil {
			state.unavailableClaims = sets.New[int]()
		}

		for _, index := range unavailableClaims {
			state.unavailableClaims.Insert(index)
		}
		return statusUnschedulable(logger, "resourceclaim not available on the node", "pod", klog.KObj(pod))
	}

	if state.allocator != nil {
		state.nodeAllocations[node.Name] = nodeAllocation{
			allocationResults:                allocations,
			extendedResourceClaim:            nodeExtendedResourceClaim,
			containerResourceRequestMappings: containerResourceRequestMappings,
		}
	}

	return nil
}

// PostFilter checks whether there are allocated claims that could get
// deallocated to help get the Pod schedulable. If yes, it picks one and
// requests its deallocation.  This only gets called when filtering found no
// suitable node.
func (pl *DynamicResources) PostFilter(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	if !pl.enabled {
		return nil, fwk.NewStatus(fwk.Unschedulable, "plugin disabled")
	}

	logger := klog.FromContext(ctx)
	state, err := getStateData(cs)
	if err != nil {
		return nil, statusError(logger, err)
	}
	// If a Pod doesn't have any resource claims attached to it, there is no need for further processing.
	// Thus we provide a fast path for this case to avoid unnecessary computations.
	if state.claims.empty() {
		return nil, fwk.NewStatus(fwk.Unschedulable, "no new claims to deallocate")
	}
	extendedResourceClaim := state.claims.extendedResourceClaim()

	// Iterating over a map is random. This is intentional here, we want to
	// pick one claim randomly because there is no better heuristic.
	for index := range state.unavailableClaims {
		claim := state.claims.get(index)
		if claim == extendedResourceClaim {
			if extendedResourceClaim != nil && !isSpecialClaimName(extendedResourceClaim.Name) {
				// Handled below.
				break
			}
			continue
		}

		if len(claim.Status.ReservedFor) == 0 ||
			len(claim.Status.ReservedFor) == 1 && claim.Status.ReservedFor[0].UID == pod.UID {
			claim := claim.DeepCopy()
			claim.Status.ReservedFor = nil
			claim.Status.Allocation = nil
			claim.Status.Devices = nil
			logger.V(5).Info("Deallocation of ResourceClaim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
			if _, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{}); err != nil {
				return nil, statusError(logger, err)
			}
			return nil, fwk.NewStatus(fwk.Unschedulable, "deallocation of ResourceClaim completed")
		}
	}

	if extendedResourceClaim != nil && !isSpecialClaimName(extendedResourceClaim.Name) {
		// If the special resource claim for extended resource backed by DRA
		// is reserved or allocated at prior scheduling cycle, then it should be deleted.
		extendedResourceClaim := extendedResourceClaim.DeepCopy()
		if err := pl.deleteClaim(ctx, extendedResourceClaim, logger); err != nil {
			return nil, statusError(logger, err)
		}
		return nil, fwk.NewStatus(fwk.Unschedulable, "deletion of ResourceClaim completed")
	}
	return nil, fwk.NewStatus(fwk.Unschedulable, "still not schedulable")
}

func (pl *DynamicResources) Score(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	if !pl.enabled {
		return 0, nil
	}
	logger := klog.FromContext(ctx)

	state, err := getStateData(cs)
	if err != nil {
		return 0, statusError(logger, err)
	}

	// If there are no claims, no need to do anything.
	if state.claims.empty() {
		return 0, nil
	}

	allocations, found := state.nodeAllocations[nodeInfo.Node().Name]
	if !found {
		return 0, nil
	}

	score, err := computeScore(state.claims.all(), allocations)
	if err != nil {
		return 0, statusError(logger, err)
	}
	return score, nil
}

func computeScore(iterator iter.Seq2[int, *resourceapi.ResourceClaim], allocations nodeAllocation) (int64, error) {
	var score int64
	for i, claim := range iterator {
		// Collect the names for all allocated subrequests.
		allocatedSubRequests := sets.New[string]()
		if i >= len(allocations.allocationResults) {
			return 0, fmt.Errorf("number of allocations %d is smaller than number of claims", len(allocations.allocationResults))
		}
		allocation := allocations.allocationResults[i]
		for _, res := range allocation.Devices.Results {
			request := res.Request
			if resourceclaim.IsSubRequestRef(request) {
				allocatedSubRequests.Insert(request)
			}
		}

		for _, req := range claim.Spec.Devices.Requests {
			if req.Exactly != nil {
				continue
			}
			for i, subReq := range req.FirstAvailable {
				subRequestRef := resourceclaim.CreateSubRequestRef(req.Name, subReq.Name)
				if allocatedSubRequests.Has(subRequestRef) {
					score += int64(resourceapi.FirstAvailableDeviceRequestMaxSize - i)
				}
			}
		}
	}
	return score, nil
}

func (pl *DynamicResources) ScoreExtensions() fwk.ScoreExtensions {
	return pl
}

func (pl *DynamicResources) NormalizeScore(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, scores fwk.NodeScoreList) *fwk.Status {
	if !pl.enabled {
		return nil
	}
	return helper.DefaultNormalizeScore(fwk.MaxNodeScore, false, scores)
}

// Reserve reserves claims for the pod.
func (pl *DynamicResources) Reserve(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) (status *fwk.Status) {
	if !pl.enabled {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if state.claims.empty() {
		return nil
	}

	logger := klog.FromContext(ctx)

	numClaimsWithAllocator := 0
	for _, claim := range state.claims.all() {
		if claim.Status.Allocation != nil {
			// Allocated, but perhaps not reserved yet. We checked in PreFilter that
			// the pod could reserve the claim. Instead of reserving here by
			// updating the ResourceClaim status, we assume that reserving
			// will work and only do it for real during binding. If it fails at
			// that time, some other pod was faster and we have to try again.
			continue
		}

		numClaimsWithAllocator++
	}

	if numClaimsWithAllocator == 0 {
		// Nothing left to do.
		return nil
	}
	extendedResourceClaim := state.claims.extendedResourceClaim()
	numClaimsToAllocate := 0
	needToAllocateUserClaims := false
	for _, claim := range state.claims.toAllocate() {
		numClaimsToAllocate++
		if claim != extendedResourceClaim {
			needToAllocateUserClaims = true
		}
	}

	// Prepare allocation of claims handled by the schedulder.
	if state.allocator != nil {
		// Entries in these two slices match each other.
		allocations, ok := state.nodeAllocations[nodeName]
		if !ok || len(allocations.allocationResults) == 0 {
			// This can happen only when claimsToAllocate has a single special claim template for extended resource backed by DRA,
			// But it is satisfied by the node with device plugin, hence no DRA allocation.
			if !needToAllocateUserClaims {
				return nil
			}

			// We checked before that the node is suitable. This shouldn't have failed,
			// so treat this as an error.
			return statusError(logger, errors.New("claim allocation not found for node"))
		}

		// Sanity check: do we have results for all pending claims?
		if len(allocations.allocationResults) != numClaimsToAllocate ||
			len(allocations.allocationResults) != numClaimsWithAllocator {
			return statusError(logger, fmt.Errorf("internal error, have %d allocations, %d claims to allocate, want %d claims", len(allocations.allocationResults), numClaimsToAllocate, numClaimsWithAllocator))
		}

		allocIndex := 0
		for index, claim := range state.claims.toAllocate() {
			// The index returned is the original index in the underlying claim store, it
			// may not be sequentially numbered (e.g. 0, 1, 2 ...).
			allocation := &allocations.allocationResults[allocIndex]
			state.informationsForClaim[index].allocation = allocation

			if claim == extendedResourceClaim {
				// replace the special claim template for extended
				// resource backed by DRA with the real instantiated claim.
				claim = allocations.extendedResourceClaim
			}

			// Strictly speaking, we don't need to store the full modified object.
			// The allocation would be enough. The full object is useful for
			// debugging, testing and the allocator, so let's make it realistic.
			claim = claim.DeepCopy()
			if !slices.Contains(claim.Finalizers, resourceapi.Finalizer) {
				claim.Finalizers = append(claim.Finalizers, resourceapi.Finalizer)
			}
			claim.Status.Allocation = allocation
			err := pl.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, claim)
			if err != nil {
				return statusError(logger, fmt.Errorf("internal error, couldn't signal allocation for claim %s", claim.Name))
			}
			logger.V(5).Info("Reserved resource in allocation result", "claim", klog.KObj(claim), "uid", claim.UID, "resourceVersion", claim.ResourceVersion, "allocation", klog.Format(allocation))
			allocIndex++
		}
	}

	return nil
}

// Unreserve clears the ReservedFor field for all claims.
// It's idempotent, and does nothing if no state found for the given pod.
func (pl *DynamicResources) Unreserve(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) {
	if !pl.enabled {
		return
	}
	state, err := getStateData(cs)
	if err != nil {
		return
	}
	if state.claims.empty() {
		return
	}

	logger := klog.FromContext(ctx)

	// we process user claims here first, extendedResourceClaim if any is handled below.
	for _, claim := range state.claims.allUserClaims() {
		// If allocation was in-flight, then it's not anymore and we need to revert the
		// claim object in the assume cache to what it was before.
		if deleted := pl.draManager.ResourceClaims().RemoveClaimPendingAllocation(claim.UID); deleted {
			logger.V(5).Info("Released resource in allocation result", "claim", klog.KObj(claim), "uid", claim.UID, "resourceVersion", claim.ResourceVersion, "allocation", klog.Format(claim.Status.Allocation))
			pl.draManager.ResourceClaims().AssumedClaimRestore(claim.Namespace, claim.Name)
		}

		if claim.Status.Allocation != nil &&
			resourceclaim.IsReservedForPod(pod, claim) {
			// Remove pod from ReservedFor. A strategic-merge-patch is used
			// because that allows removing an individual entry without having
			// the latest ResourceClaim.
			patch := fmt.Sprintf(`{"metadata": {"uid": %q}, "status": { "reservedFor": [ {"$patch": "delete", "uid": %q} ] }}`,
				claim.UID,
				pod.UID,
			)
			logger.V(5).Info("unreserve", "resourceclaim", klog.KObj(claim), "pod", klog.KObj(pod))
			claim, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).Patch(ctx, claim.Name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}, "status")
			if err != nil {
				// We will get here again when pod scheduling is retried.
				logger.Error(err, "unreserve", "resourceclaim", klog.KObj(claim))
			}
		}
	}
	pl.unreserveExtendedResourceClaim(ctx, logger, pod, state)
}

// PreBind gets called in a separate goroutine after it has been determined
// that the pod should get bound to this node. Because Reserve did not actually
// reserve claims, we need to do it now. For claims with the builtin controller,
// we also handle the allocation.
//
// If anything fails, we return an error and
// the pod will have to go into the backoff queue. The scheduler will call
// Unreserve as part of the error handling.
func (pl *DynamicResources) PreBind(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if !pl.enabled {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if state.claims.empty() {
		return nil
	}

	logger := klog.FromContext(ctx)

	for index, claim := range state.claims.all() {
		if !resourceclaim.IsReservedForPod(pod, claim) {
			claim, err := pl.bindClaim(ctx, state, index, pod, nodeName)
			if err != nil {
				return statusError(logger, err)
			}
			// Updated here such that Unreserve can work with patched claim.
			state.claims.set(index, claim)
		}
	}

	if !pl.fts.EnableDRADeviceBindingConditions || !pl.fts.EnableDRAResourceClaimDeviceStatus {
		// If we don't have binding conditions, we can return early.
		// The claim is now reserved for the pod and the scheduler can proceed with binding.
		return nil
	}

	// We need to check if the device is attached to the node.
	needToWait := hasBindingConditions(state)

	// If no device needs to be prepared, we can return early.
	if !needToWait {
		return nil
	}

	// We need to wait for the device to be attached to the node.
	pl.fh.EventRecorder().Eventf(pod, nil, v1.EventTypeNormal, "BindingConditionsPending", "Scheduling", "waiting for binding conditions for device on node %s", nodeName)
	err = wait.PollUntilContextTimeout(ctx, 5*time.Second, pl.bindingTimeout, true,
		func(ctx context.Context) (bool, error) {
			return pl.isPodReadyForBinding(state)
		})
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			err = errors.New("device binding timeout")
		}
		// Returning an error here causes another scheduling attempt.
		// In that next attempt, PreFilter will detect the timeout or
		// error and try to recover.
		return statusError(logger, err)
	}

	// If we get here, we know that reserving the claim for
	// the pod worked and we can proceed with binding it.
	return nil
}

// PreBindPreFlight is called before PreBind, and determines whether PreBind is going to do something for this pod, or not.
// It just checks state.claims to determine whether there are any claims and hence the plugin has to handle them at PreBind.
func (pl *DynamicResources) PreBindPreFlight(ctx context.Context, cs fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if !pl.enabled {
		return fwk.NewStatus(fwk.Skip)
	}
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if state.claims.empty() {
		return fwk.NewStatus(fwk.Skip)
	}
	return nil
}

// bindClaim gets called by PreBind for claim which is not reserved for the pod yet.
// It might not even be allocated. bindClaim then ensures that the allocation
// and reservation are recorded. This finishes the work started in Reserve.
func (pl *DynamicResources) bindClaim(ctx context.Context, state *stateData, index int, pod *v1.Pod, nodeName string) (*resourceapi.ResourceClaim, error) {
	logger := klog.FromContext(ctx)
	claim := state.claims.get(index)
	allocation := state.informationsForClaim[index].allocation
	isExtendedResourceClaim := false
	if claim == state.claims.extendedResourceClaim() {
		// extended resource requests satisfied by device plugin
		if allocation == nil && claim.Spec.Devices.Requests == nil {
			return claim, nil
		}
		isExtendedResourceClaim = true
	}
	claimUIDs := []types.UID{claim.UID}
	resourceClaimModified := false
	defer func() {
		// The scheduler was handling allocation. Now that has
		// completed, either successfully or with a failure.
		if resourceClaimModified {
			if isExtendedResourceClaim {
				pl.waitForExtendedClaimInAssumeCache(ctx, logger, claim)
			} else {
				// This can fail, but only for reasons that are okay (concurrent delete or update).
				// Shouldn't happen in this case.
				if err := pl.draManager.ResourceClaims().AssumeClaimAfterAPICall(claim); err != nil {
					logger.V(5).Info("Claim not stored in assume cache", "err", err)
				}
			}
		}
		if allocation != nil {
			for _, claimUID := range claimUIDs {
				if deleted := pl.draManager.ResourceClaims().RemoveClaimPendingAllocation(claimUID); deleted {
					// Creating the claim may have failed.
					resourceVersion := ""
					if claim != nil {
						resourceVersion = claim.ResourceVersion
					}
					logger.V(5).Info("Released resource in allocation result", "claim", klog.KObj(claim), "uid", claimUID, "resourceVersion", resourceVersion, "allocation", klog.Format(allocation))
				}
			}
		}
	}()

	// Create the special claim for extended resource backed by DRA
	if isExtendedResourceClaim && isSpecialClaimName(claim.Name) {
		var err error
		claim, err = pl.createExtendedResourceClaimInAPI(ctx, logger, pod, nodeName, state)
		if err != nil {
			return nil, err
		}

		resourceClaimModified = true
		// Track the actual extended ResourceClaim from now.
		// Relevant if we need to delete again in Unreserve.
		if err := state.claims.updateExtendedResourceClaim(claim); err != nil {
			return nil, fmt.Errorf("internal error: update extended ResourceClaim: %w", err)
		}
	}

	logger.V(5).Info("preparing claim status update", "claim", klog.KObj(state.claims.get(index)), "allocation", klog.Format(allocation))

	// We may run into a ResourceVersion conflict because there may be some
	// benign concurrent changes. In that case we get the latest claim and
	// try again.
	refreshClaim := false
	claim = claim.DeepCopy()
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if refreshClaim {
			updatedClaim, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).Get(ctx, claim.Name, metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("get updated claim %s after conflict: %w", klog.KObj(claim), err)
			}
			logger.V(5).Info("retrying update after conflict", "claim", klog.KObj(claim))
			claim = updatedClaim
		} else {
			// All future retries must get a new claim first.
			refreshClaim = true
		}

		if claim.DeletionTimestamp != nil {
			return fmt.Errorf("claim %s got deleted in the meantime", klog.KObj(claim))
		}

		// Do we need to store an allocation result from Reserve?
		if allocation != nil {
			if claim.Status.Allocation != nil {
				return fmt.Errorf("claim %s got allocated elsewhere in the meantime", klog.KObj(claim))
			}

			// The finalizer needs to be added in a normal update.
			// If we were interrupted in the past, it might already be set and we simply continue.
			if !slices.Contains(claim.Finalizers, resourceapi.Finalizer) {
				claim.Finalizers = append(claim.Finalizers, resourceapi.Finalizer)
				updatedClaim, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).Update(ctx, claim, metav1.UpdateOptions{})
				if err != nil {
					return fmt.Errorf("add finalizer to claim %s: %w", klog.KObj(claim), err)
				}
				claim = updatedClaim
			}
			claim.Status.Allocation = allocation
		}

		// We can simply try to add the pod here without checking
		// preconditions. The apiserver will tell us with a
		// non-conflict error if this isn't possible.
		claim.Status.ReservedFor = append(claim.Status.ReservedFor, resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: pod.Name, UID: pod.UID})
		if pl.fts.EnableDRADeviceBindingConditions && pl.fts.EnableDRAResourceClaimDeviceStatus && claim.Status.Allocation.AllocationTimestamp == nil {
			claim.Status.Allocation.AllocationTimestamp = &metav1.Time{Time: time.Now()}
		}
		updatedClaim, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
		if err != nil {
			if allocation != nil {
				return fmt.Errorf("add allocation and reservation to claim %s: %w", klog.KObj(claim), err)
			}
			return fmt.Errorf("add reservation to claim %s: %w", klog.KObj(claim), err)
		}
		claim = updatedClaim
		resourceClaimModified = true
		return nil
	})

	if retryErr != nil {
		return nil, retryErr
	}

	logger.V(5).Info("reserved", "pod", klog.KObj(pod), "node", nodeName, "resourceclaim", klog.Format(claim))

	// Patch the pod status with the new information about the generated
	// special resource claim.
	if isExtendedResourceClaim {
		err := pl.patchPodExtendedResourceClaimStatus(ctx, pod, claim, nodeName, state)
		if err != nil {
			return nil, err
		}
	}

	return claim, nil
}

// isClaimReadyForBinding checks whether a given resource claim is
// ready for binding.
// It returns an error if the claim is not ready for binding.
// It returns true if (and only if) all binding conditions are true,
// and no binding failure conditions are true,
// which includes the case that there are no binding conditions.
func (pl *DynamicResources) isClaimReadyForBinding(claim *resourceapi.ResourceClaim) (bool, error) {
	if claim.Status.Allocation == nil {
		return false, nil
	}
	for _, deviceRequest := range claim.Status.Allocation.Devices.Results {
		if len(deviceRequest.BindingConditions) == 0 {
			continue
		}
		deviceStatus := getAllocatedDeviceStatus(claim, &deviceRequest)
		if deviceStatus == nil {
			return false, nil
		}
		for _, cond := range deviceRequest.BindingFailureConditions {
			failedCond := apimeta.FindStatusCondition(deviceStatus.Conditions, cond)
			if failedCond != nil && failedCond.Status == metav1.ConditionTrue {
				return false, fmt.Errorf("claim %s binding failed: reason=%s, message=%q",
					claim.Name,
					failedCond.Reason,
					failedCond.Message)
			}
		}
		for _, cond := range deviceRequest.BindingConditions {
			if !apimeta.IsStatusConditionTrue(deviceStatus.Conditions, cond) {
				return false, nil
			}
		}
	}
	return true, nil
}

// isClaimTimeout checks whether a given resource claim has
// reached the binding timeout.
// It returns true if the binding timeout is reached.
// It returns false if the binding timeout is not reached.
func (pl *DynamicResources) isClaimTimeout(claim *resourceapi.ResourceClaim) bool {
	if !pl.fts.EnableDRADeviceBindingConditions || !pl.fts.EnableDRAResourceClaimDeviceStatus {
		return false
	}
	if claim.Status.Allocation == nil || claim.Status.Allocation.AllocationTimestamp == nil {
		return false
	}
	// check if the binding timeout is reached
	for _, deviceRequest := range claim.Status.Allocation.Devices.Results {
		if deviceRequest.BindingConditions == nil {
			continue
		}
		if claim.Status.Allocation.AllocationTimestamp.Add(pl.bindingTimeout).Before(time.Now()) {
			return true
		}
	}
	return false
}

// isPodReadyForBinding checks the binding status of devices within the given state claims.
// It returns true if (and only if) all binding conditions are true,
// and no binding failure conditions are true,
// which includes the case when there are no binding conditions.
// It returns an error if any binding failure condition is set.
func (pl *DynamicResources) isPodReadyForBinding(state *stateData) (bool, error) {
	for claimIndex, claim := range state.claims.all() {
		claim, err := pl.draManager.ResourceClaims().Get(claim.Namespace, claim.Name)
		if err != nil {
			return false, err
		}
		state.claims.set(claimIndex, claim)
		ready, err := pl.isClaimReadyForBinding(claim)
		if err != nil {
			return false, err
		}
		if !ready {
			if pl.isClaimTimeout(claim) {
				return false, fmt.Errorf("claim %s binding timeout", claim.Name)
			}
			return false, nil
		}
	}
	return true, nil
}

// hasBindingConditions checks whether any of the claims in the state
// has binding conditions.
// It returns true if at least one claim has binding conditions.
// It returns false if no claim has binding conditions.
func hasBindingConditions(state *stateData) bool {
	for _, claim := range state.claims.all() {
		if claim.Status.Allocation == nil {
			continue
		}
		for _, device := range claim.Status.Allocation.Devices.Results {
			if len(device.BindingConditions) > 0 {
				return true
			}
		}
	}
	return false
}

// statusUnschedulable ensures that there is a log message associated with the
// line where the status originated.
func statusUnschedulable(logger klog.Logger, reason string, kv ...interface{}) *fwk.Status {
	if loggerV := logger.V(5); loggerV.Enabled() {
		helper, loggerV := loggerV.WithCallStackHelper()
		helper()
		kv = append(kv, "reason", reason)
		// nolint: logcheck // warns because it cannot check key/values
		loggerV.Info("pod unschedulable", kv...)
	}
	return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, reason)
}

// statusError ensures that there is a log message associated with the
// line where the error originated.
func statusError(logger klog.Logger, err error, kv ...interface{}) *fwk.Status {
	if loggerV := logger.V(5); loggerV.Enabled() {
		helper, loggerV := loggerV.WithCallStackHelper()
		helper()
		// nolint: logcheck // warns because it cannot check key/values
		loggerV.Error(err, "dynamic resource plugin failed", kv...)
	}
	return fwk.AsStatus(err)
}

func getAllocatedDeviceStatus(claim *resourceapi.ResourceClaim, deviceRequest *resourceapi.DeviceRequestAllocationResult) *resourceapi.AllocatedDeviceStatus {
	for _, device := range claim.Status.Devices {
		if deviceRequest.Device == device.Device && deviceRequest.Driver == device.Driver && deviceRequest.Pool == device.Pool {
			return &device
		}
	}
	return nil
}
