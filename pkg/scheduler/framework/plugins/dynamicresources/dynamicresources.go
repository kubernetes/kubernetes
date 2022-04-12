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
	"sort"
	"sync"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-helpers/cdi/resourceclaim"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

const (
	// Name is the name of the plugin used in Registry and configurations.
	Name = names.DynamicResources

	stateKey framework.StateKey = Name
)

// The state is initialized in PreFilter phase. Because we save the pointer in
// framework.CycleState, in the later phases we don't need to call Write method
// to update the value
type stateData struct {
	clientset kubernetes.Interface

	// A copy of all claims for the Pod (i.e. 1:1 match with
	// pod.Spec.ResourceClaims), initially with the status from the start
	// of the scheduling cycle. Each claim instance is read-only because it
	// might come from the informer cache. The instances get replaced when
	// the plugin itself successfully does an Update.
	//
	// Empty if the Pod has no claims.
	claims []*corev1.ResourceClaim

	// The indices of all claims that:
	// - are allocated
	// - use delayed allocation
	// - were not available on at least one node
	//
	// Set in parallel during Filter, so write access there must be
	// protected by the mutex. Used by PostFilter.
	unavailableClaims sets.Int

	// A pointer to the PodScheduling object for the pod, if one exists.
	// Gets set on demand.
	//
	// Conceptually, this object belongs into the scheduler framework
	// where it might get shared by different plugins. But in practice,
	// it is currently only used by dynamic provisioning and thus
	// managed entirely here.
	podScheduling *corev1.PodScheduling

	mutex sync.Mutex
}

func (d *stateData) Clone() framework.StateData {
	return d
}

func (d *stateData) updateClaimStatus(ctx context.Context, index int, claim *corev1.ResourceClaim) error {
	// TODO (?): replace with patch operation. Beware that patching must only succeed if the
	// object has not been modified in parallel by someone else.
	claim, err := d.clientset.CoreV1().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
	// TODO: metric for update results, with the operation ("set selected
	// node", "set PotentialNodes", etc.) as one dimension.
	if err != nil {
		return fmt.Errorf("update resource claim: %w", err)
	}

	// Remember the new instance. This is relevant when the plugin must
	// update the same claim multiple times (for example, first set
	// PotentialNodes, then SelectedNode), because otherwise the second
	// update would fail with a "was modified" error.
	d.claims[index] = claim

	return nil
}

// initializePodScheduling can be called concurrently. It returns an existing PodScheduling
// object if there is one already, retrieves one if not, or as a last resort creates
// one from scratch.
func (d *stateData) initializePodScheduling(ctx context.Context, pod *corev1.Pod, podSchedulingLister corev1listers.PodSchedulingLister) (*corev1.PodScheduling, error) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	if d.podScheduling != nil {
		return d.podScheduling, nil
	}

	podScheduling, err := podSchedulingLister.PodSchedulings(pod.Namespace).Get(pod.Name)
	switch {
	case apierrors.IsNotFound(err):
		controller := true
		podScheduling = &corev1.PodScheduling{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pod.Name,
				Namespace: pod.Namespace,
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "Pod",
						Name:       pod.Name,
						UID:        pod.UID,
						Controller: &controller,
					},
				},
			},
		}
		err = nil
	case err != nil:
		return nil, err
	default:
		// We have an object, but it might be obsolete.
		if !metav1.IsControlledBy(podScheduling, pod) {
			return nil, fmt.Errorf("PodScheduling object with UID %s is not owned by Pod %s/%s", podScheduling.UID, pod.Namespace, pod.Name)
		}
	}
	d.podScheduling = podScheduling
	return podScheduling, err
}

// publishPodScheduling creates or updates the PodSchediling object.
func (d *stateData) publishPodScheduling(ctx context.Context, podScheduling *corev1.PodScheduling) error {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	var err error
	logger := klog.FromContext(ctx)
	if podScheduling.UID == "" {
		logger.V(5).Info("creating PodScheduling", "podscheduling", klog.KObj(podScheduling))
		podScheduling, err = d.clientset.CoreV1().PodSchedulings(podScheduling.Namespace).Create(ctx, podScheduling, metav1.CreateOptions{})
	} else {
		// TODO: patch here to avoid racing with drivers which update the status.
		logger.V(5).Info("updating PodScheduling", "podscheduling", klog.KObj(podScheduling))
		podScheduling, err = d.clientset.CoreV1().PodSchedulings(podScheduling.Namespace).Update(ctx, podScheduling, metav1.UpdateOptions{})
	}
	if err != nil {
		return err
	}
	d.podScheduling = podScheduling
	return nil
}

func statusForClaim(podScheduling *corev1.PodScheduling, podClaimName string) *corev1.ResourceClaimSchedulingStatus {
	for _, status := range podScheduling.Status.Claims {
		if status.PodResourceClaimName == podClaimName {
			return &status
		}
	}
	return nil
}

// dynamicResources is a plugin that ensures that ResourceClaims are allocated.
type dynamicResources struct {
	clientset           kubernetes.Interface
	claimLister         corev1listers.ResourceClaimLister
	podSchedulingLister corev1listers.PodSchedulingLister
}

// New initializes a new plugin and returns it.
func New(plArgs runtime.Object, fh framework.Handle) (framework.Plugin, error) {
	return &dynamicResources{
		clientset:           fh.ClientSet(),
		claimLister:         fh.SharedInformerFactory().Core().V1().ResourceClaims().Lister(),
		podSchedulingLister: fh.SharedInformerFactory().Core().V1().PodSchedulings().Lister(),
	}, nil
}

var _ framework.PreFilterPlugin = &dynamicResources{}
var _ framework.FilterPlugin = &dynamicResources{}
var _ framework.PostFilterPlugin = &dynamicResources{}
var _ framework.PreScorePlugin = &dynamicResources{}
var _ framework.ReservePlugin = &dynamicResources{}
var _ framework.EnqueueExtensions = &dynamicResources{}
var _ framework.PostBindPlugin = &dynamicResources{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *dynamicResources) Name() string {
	return Name
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *dynamicResources) EventsToRegister() []framework.ClusterEvent {
	events := []framework.ClusterEvent{
		// Allocation is tracked in ResourceClaims, so any changes may make the pods schedulable.
		{Resource: framework.ResourceClaim, ActionType: framework.Add | framework.Update},
		// When a driver has provided additional information, a pod waiting for that information
		// may be schedulable.
		// TODO: can we change this so that such an event does not trigger *all* pods?
		{Resource: framework.PodScheduling, ActionType: framework.Add | framework.Update},
		// A resource might depend on node labels for topology filtering.
		// A new or updated node may make pods schedulable.
		{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeLabel},
	}
	return events
}

// podHasClaims returns the ResourceClaims for all pod.Spec.PodResourceClaims.
func (pl *dynamicResources) podHasClaims(pod *corev1.Pod) ([]*corev1.ResourceClaim, error) {
	claims := make([]*corev1.ResourceClaim, 0, len(pod.Spec.ResourceClaims))
	for _, resource := range pod.Spec.ResourceClaims {
		claimName := resourceclaim.Name(pod, &resource)
		isEphemeral := resource.Claim.Template != nil
		claim, err := pl.claimLister.ResourceClaims(pod.Namespace).Get(claimName)
		if err != nil {
			// The error usually has already enough context ("resourcevolumeclaim "myclaim" not found"),
			// but we can do better for generic ephemeral inline volumes where that situation
			// is normal directly after creating a pod.
			if isEphemeral && apierrors.IsNotFound(err) {
				err = fmt.Errorf("waiting for dynamic resource controller to create the resourceclaim %q", claimName)
			}
			return nil, err
		}

		if claim.DeletionTimestamp != nil {
			return nil, fmt.Errorf("resourceclaim %q is being deleted", claim.Name)
		}

		if isEphemeral {
			if err := resourceclaim.IsForPod(pod, claim); err != nil {
				return nil, err
			}
		}
		// We store the pointer as returned by the lister. The
		// assumption is that if a claim gets modified while our code
		// runs, the cache will store a new pointer, not mutate the
		// existing object that we point to here.
		claims = append(claims, claim)
	}
	return claims, nil
}

// PreFilter invoked at the prefilter extension point to check if pod has all
// immediate claims bound. UnschedulableAndUnresolvable is returned if
// the pod cannot be scheduled at the moment on any node.
func (pl *dynamicResources) PreFilter(ctx context.Context, state *framework.CycleState, pod *corev1.Pod) (*framework.PreFilterResult, *framework.Status) {
	logger := klog.FromContext(ctx)

	// If pod does not reference any claim, we don't need to do anything.
	claims, err := pl.podHasClaims(pod)
	if err != nil {
		return nil, statusUnschedulable(logger, err.Error())
	}
	logger.V(5).Info("pod resource claims", "pod", klog.KObj(pod), "resourceclaims", klog.KObjs(claims))
	if len(claims) == 0 {
		state.Write(stateKey, &stateData{})
		return nil, nil
	}

	for _, claim := range claims {
		if claim.Spec.AllocationMode == corev1.AllocationModeImmediate &&
			claim.Status.Allocation == nil {
			// This will get resolved by the resource driver.
			return nil, statusUnschedulable(logger, "unallocated immediate resourceclaim", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
		}
		if claim.Status.DeallocationRequested {
			// Same here
			return nil, statusUnschedulable(logger, "resourceclaim must be reallocated", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
		}
		if claim.Status.Allocation != nil &&
			!canBeReserved(claim) &&
			!isReservedForPod(pod, claim) {
			// Resource is in use. The pod has to wait.
			return nil, statusUnschedulable(logger, "resourceclaim in use", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
		}
	}

	state.Write(stateKey, &stateData{clientset: pl.clientset, claims: claims})
	return nil, nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *dynamicResources) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func getStateData(cs *framework.CycleState) (*stateData, error) {
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
func (pl *dynamicResources) Filter(ctx context.Context, cs *framework.CycleState, pod *corev1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if len(state.claims) == 0 {
		return nil
	}

	logger := klog.FromContext(ctx)
	node := nodeInfo.Node()
	if node == nil {
		return statusError(logger, errors.New("node not found"))
	}

	// We bail out early here as soon as we know that the pod is unschedulable.
	// We could also gather all reasons and then report all of them, but that is
	// more complex.
	var unavailableClaims []int
	for index, claim := range state.claims {
		logger.V(5).Info("filter", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim))
		switch {
		case claim.Status.Allocation != nil:
			if claim.Status.Allocation.AvailableOnNodes != nil {
				nodeSelector, err := nodeaffinity.NewNodeSelector(claim.Status.Allocation.AvailableOnNodes)
				if err != nil {
					return statusError(logger, err)
				}
				if !nodeSelector.Match(node) {
					logger.V(5).Info("AvailableOnNodes does not match", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim))
					unavailableClaims = append(unavailableClaims, index)
				}
			}
		case claim.Status.DeallocationRequested:
			// We shouldn't get here. PreFilter already checked this.
			return statusUnschedulable(logger, "resourceclaim must be reallocated", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim))
		default:
			// This must be delayed allocation. Immediate
			// allocation was already checked for in PreFilter.
			// Now we need information from drivers.
			podScheduling, err := state.initializePodScheduling(ctx, pod, pl.podSchedulingLister)
			if err != nil {
				return statusError(logger, err)
			}
			status := statusForClaim(podScheduling, pod.Spec.ResourceClaims[index].Name)
			if status != nil {
				for _, unsuitableNode := range status.UnsuitableNodes {
					if node.Name == unsuitableNode {
						return statusUnschedulable(logger, "resourceclaim cannot be allocated for the node (unsuitable)", "pod", klog.KObj(pod), "node", klog.KObj(node), "resourceclaim", klog.KObj(claim), "unsuitablenodes", status.UnsuitableNodes)
					}
				}
			}
		}
	}

	if len(unavailableClaims) > 0 {
		state.mutex.Lock()
		defer state.mutex.Unlock()
		state.unavailableClaims.Insert(unavailableClaims...)
		return statusUnschedulable(logger, "resourceclaim not available on the node", "pod", klog.KObj(pod))
	}

	return nil
}

// PostFilter checks whether freeing an allocated claim might help to get a Pod
// schedulable. This only gets called when filtering found no suitable node.
func (pl *dynamicResources) PostFilter(ctx context.Context, cs *framework.CycleState, pod *corev1.Pod, filteredNodeStatusMap framework.NodeToStatusMap) (*framework.PostFilterResult, *framework.Status) {
	logger := klog.FromContext(ctx)
	state, err := getStateData(cs)
	if err != nil {
		return nil, statusError(logger, err)
	}
	if len(state.claims) == 0 {
		return nil, nil
	}

	// Iterating over a map is random. This is intentional here, we want to
	// pick one claim randomly because there is no better heuristic.
	for index := range state.unavailableClaims {
		claim := state.claims[index]
		if len(claim.Status.ReservedFor) == 0 ||
			len(claim.Status.ReservedFor) == 1 && claim.Status.ReservedFor[0].UID == pod.UID {
			claim := state.claims[index].DeepCopy()
			claim.Status.DeallocationRequested = true
			claim.Status.ReservedFor = nil
			logger.V(5).Info("reallocate", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
			if err := state.updateClaimStatus(ctx, index, claim); err != nil {
				return nil, statusError(logger, err)
			}
			break
		}
	}
	return nil, nil
}

// PreScore is passed a list of all nodes that would fit the pod. Not all
// claims are necessarily allocated yet, so here we can set the SuitableNodes
// field for those which are pending.
func (pl *dynamicResources) PreScore(ctx context.Context, cs *framework.CycleState, pod *corev1.Pod, nodes []*corev1.Node) *framework.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if len(state.claims) == 0 {
		return nil
	}

	logger := klog.FromContext(ctx)
	podScheduling, err := state.initializePodScheduling(ctx, pod, pl.podSchedulingLister)
	pending := false
	for _, claim := range state.claims {
		if claim.Status.Allocation == nil {
			pending = true
		}
	}
	if pending && !haveAllNodes(podScheduling.Spec.PotentialNodes, nodes) {
		logger.V(5).Info("setting potential nodes", "pod", klog.KObj(pod), "potentialnodes", nodes)
		podScheduling = podScheduling.DeepCopy()
		podScheduling.Spec.PotentialNodes = make([]string, 0, len(nodes))
		for _, node := range nodes {
			podScheduling.Spec.PotentialNodes = append(podScheduling.Spec.PotentialNodes, node.Name)
		}
		sort.Strings(podScheduling.Spec.PotentialNodes)
		if err := state.publishPodScheduling(ctx, podScheduling); err != nil {
			return statusError(logger, err)
		}
	}

	return nil
}

func haveAllNodes(nodeNames []string, nodes []*corev1.Node) bool {
	for _, node := range nodes {
		if !haveNode(nodeNames, node.Name) {
			return false
		}
	}
	return true
}

func haveNode(nodeNames []string, nodeName string) bool {
	for _, n := range nodeNames {
		if n == nodeName {
			return true
		}
	}
	return false
}

// Reserve reserves claims for the pod.
func (pl *dynamicResources) Reserve(ctx context.Context, cs *framework.CycleState, pod *corev1.Pod, nodeName string) *framework.Status {
	state, err := getStateData(cs)
	if err != nil {
		return statusError(klog.FromContext(ctx), err)
	}
	if len(state.claims) == 0 {
		return nil
	}

	pending := 0
	infos := 0
	logger := klog.FromContext(ctx)
	podScheduling, err := state.initializePodScheduling(ctx, pod, pl.podSchedulingLister)
	for index, claim := range state.claims {
		if claim.Status.Allocation != nil {
			// Allocated, but perhaps not reserved yet.
			if isReservedForPod(pod, claim) {
				logger.V(5).Info("is reserved", "pod", klog.KObj(pod), "node", klog.ObjectRef{Name: nodeName}, "resourceclaim", klog.KObj(claim))
				continue
			}
			claim := claim.DeepCopy()
			claim.Status.ReservedFor = append(claim.Status.ReservedFor,
				corev1.ResourceClaimUserReference{
					Version:  "v1",
					Resource: "pods",
					Name:     pod.Name,
					UID:      pod.UID,
				})
			logger.V(5).Info("reserve", "pod", klog.KObj(pod), "node", klog.ObjectRef{Name: nodeName}, "resourceclaim", klog.KObj(claim))
			_, err := pl.clientset.CoreV1().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{})
			// TODO: metric for update errors.
			if err != nil {
				return statusError(logger, err)
			}
			// If we get here, we know that reserving the claim for
			// the pod worked and we can proceed with scheduling
			// it.
		} else {
			// Must be delayed allocation.
			pending++

			// Did the driver provide information that steered node
			// selection towards a node that it can support?
			if statusForClaim(podScheduling, pod.Spec.ResourceClaims[index].Name) != nil {
				infos++
			}
		}
	}

	if pending == 0 {
		// Nothing left to do.
		return nil
	}

	// When there is only one pending resource, we can go ahead with
	// requesting allocation even when we don't have the information from
	// the driver yet. Otherwise we wait for information before blindly
	// making a decision that might have to be reversed later.
	if pending == 1 || infos == pending {
		if err != nil {
			return statusError(logger, err)
		}
		podScheduling = podScheduling.DeepCopy()
		podScheduling.Spec.SelectedNode = nodeName
		logger.V(5).Info("start allocation", "pod", klog.KObj(pod), "node", klog.ObjectRef{Name: nodeName})
		if err := state.publishPodScheduling(ctx, podScheduling); err != nil {
			return statusError(logger, err)
		}
	}

	// TODO: can or should we ensure that scheduling gets aborted while
	// waiting for resources *before* triggering delayed volume
	// provisioning?  On the one hand, volume provisioning is currently
	// irreversible, so it better should come last. On the other,
	// triggering both in parallel might be faster.
	return statusUnschedulable(logger, "waiting for resource driver to allocate resources", "pod", klog.KObj(pod), "node", klog.ObjectRef{Name: nodeName})
}

func isReservedForPod(pod *corev1.Pod, claim *corev1.ResourceClaim) bool {
	for _, reserved := range claim.Status.ReservedFor {
		if reserved.UID == pod.UID {
			return true
		}
	}
	return false
}

func canBeReserved(claim *corev1.ResourceClaim) bool {
	return claim.Status.Allocation.SharedResource ||
		len(claim.Status.ReservedFor) == 0
}

// Unreserve clears the ReservedFor field for all claims.
// It's idempotent, and does nothing if no state found for the given pod.
func (pl *dynamicResources) Unreserve(ctx context.Context, cs *framework.CycleState, pod *corev1.Pod, nodeName string) {
	state, err := getStateData(cs)
	if err != nil {
		return
	}
	if len(state.claims) == 0 {
		return
	}

	logger := klog.FromContext(ctx)
	for index, claim := range state.claims {
		if claim.Status.Allocation != nil &&
			isReservedForPod(pod, claim) {
			// Remove pod from ReservedFor.
			claim := claim.DeepCopy()
			reservedFor := make([]corev1.ResourceClaimUserReference, 0, len(claim.Status.ReservedFor)-1)
			for _, reserved := range claim.Status.ReservedFor {
				// TODO: can UID be assumed to be unique all resources or do we also need to compare Group/Version/Resource?
				if reserved.UID != pod.UID {
					reservedFor = append(reservedFor, reserved)
				}
			}
			claim.Status.ReservedFor = reservedFor
			logger.V(5).Info("unreserve", "resourceclaim", klog.KObj(claim))
			if err := state.updateClaimStatus(ctx, index, claim); err != nil {
				// We will get here again when pod scheduling
				// is retried.
				logger.Error(err, "unreserve", "resourceclaim", klog.KObj(claim))
			}
		}
	}
	return
}

// PostBind is called after a pod is successfully bound to a node. Now we are
// sure that a PodScheduling object, if it exists, is definitely not going to
// be needed anymore and can delete it. This is a one-shot thing, there won't
// be any retries.  This is okay because it should usually work and in those
// cases where it doesn't, the garbage collector will eventually clean up.
func (pl *dynamicResources) PostBind(ctx context.Context, cs *framework.CycleState, pod *corev1.Pod, nodeName string) {
	state, err := getStateData(cs)
	if err != nil {
		return
	}
	if len(state.claims) == 0 {
		return
	}

	// We cannot know for sure whether the PodScheduling object exists. We
	// might have created it in the previous pod scheduling cycle and not
	// have it in our informer cache yet. Let's try to delete, just to be
	// on the safe side.
	logger := klog.FromContext(ctx)
	err = pl.clientset.CoreV1().PodSchedulings(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	switch {
	case apierrors.IsNotFound(err):
		logger.V(5).Info("no PodScheduling object to delete")
	case err != nil:
		logger.Error(err, "delete PodScheduling")
	default:
		logger.V(5).Info("PodScheduling object deleted")
	}
}

// statusUnschedulable ensures that there is a log message associated with the
// line where the status originated.
func statusUnschedulable(logger klog.Logger, reason string, kv ...interface{}) *framework.Status {
	if loggerV := logger.V(5); loggerV.Enabled() {
		helper, loggerV := loggerV.WithCallStackHelper()
		helper()
		kv = append(kv, "reason", reason)
		loggerV.Info("pod unschedulable", kv...)
	}
	return framework.NewStatus(framework.UnschedulableAndUnresolvable, reason)
}

// statusError ensures that there is a log message associated with the
// line where the error originated.
func statusError(logger klog.Logger, err error, kv ...interface{}) *framework.Status {
	if loggerV := logger.V(5); loggerV.Enabled() {
		helper, loggerV := loggerV.WithCallStackHelper()
		helper()
		loggerV.Error(err, "dynamic resource plugin failed", kv...)
	}
	return framework.AsStatus(err)
}
