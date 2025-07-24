/*
Copyright 2015 The Kubernetes Authors.

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

package framework

import (
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/resource"
	resourcehelper "k8s.io/component-helpers/resource"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

var generation int64

var (
	// basicActionTypes is a list of basic ActionTypes.
	basicActionTypes = []fwk.ActionType{fwk.Add, fwk.Delete, fwk.Update}
	// podActionTypes is a list of ActionTypes that are only applicable for Pod events.
	podActionTypes = []fwk.ActionType{fwk.UpdatePodLabel, fwk.UpdatePodScaleDown, fwk.UpdatePodToleration, fwk.UpdatePodSchedulingGatesEliminated, fwk.UpdatePodGeneratedResourceClaim}
	// nodeActionTypes is a list of ActionTypes that are only applicable for Node events.
	nodeActionTypes = []fwk.ActionType{fwk.UpdateNodeAllocatable, fwk.UpdateNodeLabel, fwk.UpdateNodeTaint, fwk.UpdateNodeCondition, fwk.UpdateNodeAnnotation}
)

// Constants for GVKs.
const (
	// These assignedPod and unschedulablePod are internal resources that are used to represent the type of Pod.
	// We don't expose them to the plugins deliberately because we don't publish Pod events with unschedulable Pods in the first place.
	assignedPod      fwk.EventResource = "AssignedPod"
	unschedulablePod fwk.EventResource = "UnschedulablePod"
)

var (
	// allResources is a list of all resources.
	allResources = []fwk.EventResource{
		fwk.Pod,
		assignedPod,
		unschedulablePod,
		fwk.Node,
		fwk.PersistentVolume,
		fwk.PersistentVolumeClaim,
		fwk.CSINode,
		fwk.CSIDriver,
		fwk.CSIStorageCapacity,
		fwk.StorageClass,
		fwk.VolumeAttachment,
		fwk.ResourceClaim,
		fwk.ResourceSlice,
		fwk.DeviceClass,
	}
)

// AllClusterEventLabels returns all possible cluster event labels given to the metrics.
func AllClusterEventLabels() []string {
	labels := []string{UnschedulableTimeout, ForceActivate}
	for _, r := range allResources {
		for _, a := range basicActionTypes {
			labels = append(labels, fwk.ClusterEvent{Resource: r, ActionType: a}.Label())
		}
	}
	for _, a := range podActionTypes {
		labels = append(labels, fwk.ClusterEvent{Resource: fwk.Pod, ActionType: a}.Label())
	}
	for _, a := range nodeActionTypes {
		labels = append(labels, fwk.ClusterEvent{Resource: fwk.Node, ActionType: a}.Label())
	}
	return labels
}

// ClusterEventIsWildCard returns true if the given ClusterEvent follows WildCard semantics
func ClusterEventIsWildCard(ce fwk.ClusterEvent) bool {
	return ce.Resource == fwk.WildCard && ce.ActionType == fwk.All
}

// MatchClusterEvents returns true if ce is matched with incomingEvent.
// "match" means that incomingEvent is the same or more specific than the ce.
// e.g. when ce.ActionType is Update and incomingEvent.ActionType is UpdateNodeLabel, it will return true
// because UpdateNodeLabel is more specific than Update.
// On the other hand, when ce.ActionType is UpdateNodeLabel and incomingEvent.ActionType is Update, it returns false.
// This is based on the fact that the scheduler interprets the incoming cluster event as specific event as possible;
// meaning, if incomingEvent is Node/Update, it means that Node's update is not something that can be interpreted
// as any of Node's specific Update events.
//
// If the ce.Resource is "*", there's no requirement for incomingEvent.Resource.
// Contrarily, if incomingEvent.Resource is "*", the only accepted ce.Resource is "*" (which should never
// happen in the current implementation of the scheduling queue).
//
// Note: we have a special case here when incomingEvent is a wildcard event, it will force all Pods to move
// to activeQ/backoffQ, but we take it as an unmatched event unless ce is also a wildcard event.
func MatchClusterEvents(ce, incomingEvent fwk.ClusterEvent) bool {
	return ClusterEventIsWildCard(ce) ||
		matchEventResources(ce.Resource, incomingEvent.Resource) && ce.ActionType&incomingEvent.ActionType != 0 && incomingEvent.ActionType <= ce.ActionType
}

// match returns true if the resource is matched with the coming resource.
func matchEventResources(r, resource fwk.EventResource) bool {
	// WildCard matches all resources
	return r == fwk.WildCard ||
		// Exact match
		r == resource ||
		// Pod matches assignedPod and unschedulablePod.
		// (assignedPod and unschedulablePod aren't exposed and hence only used for incoming events and never used in EventsToRegister)
		r == fwk.Pod && (resource == assignedPod || resource == unschedulablePod)
}

func MatchAnyClusterEvent(ce fwk.ClusterEvent, incomingEvents []fwk.ClusterEvent) bool {
	for _, e := range incomingEvents {
		if MatchClusterEvents(e, ce) {
			return true
		}
	}
	return false
}

func UnrollWildCardResource() []fwk.ClusterEventWithHint {
	return []fwk.ClusterEventWithHint{
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.PersistentVolume, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.PersistentVolumeClaim, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.CSINode, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.CSIDriver, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.CSIStorageCapacity, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.StorageClass, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.ResourceClaim, ActionType: fwk.All}},
		{Event: fwk.ClusterEvent{Resource: fwk.DeviceClass, ActionType: fwk.All}},
	}
}

// NodeInfo is node level aggregated information.
type NodeInfo struct {
	// Overall node information.
	node *v1.Node

	// Pods running on the node.
	Pods []fwk.PodInfo

	// The subset of pods with affinity.
	PodsWithAffinity []fwk.PodInfo

	// The subset of pods with required anti-affinity.
	PodsWithRequiredAntiAffinity []fwk.PodInfo

	// Ports allocated on the node.
	UsedPorts fwk.HostPortInfo

	// Total requested resources of all pods on this node. This includes assumed
	// pods, which scheduler has sent for binding, but may not be scheduled yet.
	Requested *Resource
	// Total requested resources of all pods on this node with a minimum value
	// applied to each container's CPU and memory requests. This does not reflect
	// the actual resource requests for this node, but is used to avoid scheduling
	// many zero-request pods onto one node.
	NonZeroRequested *Resource
	// We store allocatedResources (which is Node.Status.Allocatable.*) explicitly
	// as int64, to avoid conversions and accessing map.
	Allocatable *Resource

	// ImageStates holds the entry of an image if and only if this image is on the node. The entry can be used for
	// checking an image's existence and advanced usage (e.g., image locality scheduling policy) based on the image
	// state information.
	ImageStates map[string]*fwk.ImageStateSummary

	// PVCRefCounts contains a mapping of PVC names to the number of pods on the node using it.
	// Keys are in the format "namespace/name".
	PVCRefCounts map[string]int

	// Whenever NodeInfo changes, generation is bumped.
	// This is used to avoid cloning it if the object didn't change.
	Generation int64
}

func (n *NodeInfo) GetPods() []fwk.PodInfo {
	return n.Pods
}

func (n *NodeInfo) GetPodsWithAffinity() []fwk.PodInfo {
	return n.PodsWithAffinity
}

func (n *NodeInfo) GetPodsWithRequiredAntiAffinity() []fwk.PodInfo {
	return n.PodsWithRequiredAntiAffinity
}

func (n *NodeInfo) GetUsedPorts() fwk.HostPortInfo {
	return n.UsedPorts
}

func (n *NodeInfo) GetRequested() fwk.Resource {
	return n.Requested
}

func (n *NodeInfo) GetNonZeroRequested() fwk.Resource {
	return n.NonZeroRequested
}

func (n *NodeInfo) GetAllocatable() fwk.Resource {
	return n.Allocatable
}

func (n *NodeInfo) GetImageStates() map[string]*fwk.ImageStateSummary {
	return n.ImageStates
}

func (n *NodeInfo) GetPVCRefCounts() map[string]int {
	return n.PVCRefCounts
}

func (n *NodeInfo) GetGeneration() int64 {
	return n.Generation
}

// NodeInfo implements KMetadata, so for example klog.KObjSlice(nodes) works
// when nodes is a []*NodeInfo.
var _ klog.KMetadata = &NodeInfo{}

// GetName returns the name of the node wrapped by this NodeInfo object, or a meaningful nil representation if node or node name is nil.
// This method is a part of interface KMetadata.
func (n *NodeInfo) GetName() string {
	if n == nil {
		return "<nil>"
	}
	if n.node == nil {
		return "<no node>"
	}
	return n.node.Name
}

// GetNamespace is a part of interface KMetadata. For NodeInfo it should always return an empty string, since Node is not a namespaced resource.
func (n *NodeInfo) GetNamespace() string {
	return ""
}

// Node returns overall information about this node.
func (n *NodeInfo) Node() *v1.Node {
	if n == nil {
		return nil
	}
	return n.node
}

// Snapshot returns a copy of this node, same as SnapshotConcrete, but with returned type fwk.NodeInfo
// (the purpose is to have NodeInfo implement interface fwk.NodeInfo).
func (n *NodeInfo) Snapshot() fwk.NodeInfo {
	return n.SnapshotConcrete()
}

// SnapshotConcrete returns a copy of this node, Except that ImageStates is copied without the Nodes field.
func (n *NodeInfo) SnapshotConcrete() *NodeInfo {
	clone := &NodeInfo{
		node:             n.node,
		Requested:        n.Requested.Clone(),
		NonZeroRequested: n.NonZeroRequested.Clone(),
		Allocatable:      n.Allocatable.Clone(),
		UsedPorts:        make(fwk.HostPortInfo),
		ImageStates:      make(map[string]*fwk.ImageStateSummary),
		PVCRefCounts:     make(map[string]int),
		Generation:       n.Generation,
	}
	if len(n.Pods) > 0 {
		clone.Pods = append([]fwk.PodInfo(nil), n.Pods...)
	}
	if len(n.UsedPorts) > 0 {
		// HostPortInfo is a map-in-map struct
		// make sure it's deep copied
		for ip, portMap := range n.UsedPorts {
			clone.UsedPorts[ip] = make(map[fwk.ProtocolPort]struct{})
			for protocolPort, v := range portMap {
				clone.UsedPorts[ip][protocolPort] = v
			}
		}
	}
	if len(n.PodsWithAffinity) > 0 {
		clone.PodsWithAffinity = append([]fwk.PodInfo(nil), n.PodsWithAffinity...)
	}
	if len(n.PodsWithRequiredAntiAffinity) > 0 {
		clone.PodsWithRequiredAntiAffinity = append([]fwk.PodInfo(nil), n.PodsWithRequiredAntiAffinity...)
	}
	if len(n.ImageStates) > 0 {
		state := make(map[string]*fwk.ImageStateSummary, len(n.ImageStates))
		for imageName, imageState := range n.ImageStates {
			state[imageName] = imageState.Snapshot()
		}
		clone.ImageStates = state
	}
	for key, value := range n.PVCRefCounts {
		clone.PVCRefCounts[key] = value
	}
	return clone
}

// String returns representation of human readable format of this NodeInfo.
func (n *NodeInfo) String() string {
	podKeys := make([]string, len(n.Pods))
	for i, p := range n.Pods {
		podKeys[i] = p.GetPod().Name
	}
	return fmt.Sprintf("&NodeInfo{Pods:%v, RequestedResource:%#v, NonZeroRequest: %#v, UsedPort: %#v, AllocatableResource:%#v}",
		podKeys, n.Requested, n.NonZeroRequested, n.UsedPorts, n.Allocatable)
}

// AddPodInfo adds pod information to this NodeInfo.
// Consider using this instead of AddPod if a PodInfo is already computed.
func (n *NodeInfo) AddPodInfo(podInfo fwk.PodInfo) {
	n.Pods = append(n.Pods, podInfo)
	if podWithAffinity(podInfo.GetPod()) {
		n.PodsWithAffinity = append(n.PodsWithAffinity, podInfo)
	}
	if podWithRequiredAntiAffinity(podInfo.GetPod()) {
		n.PodsWithRequiredAntiAffinity = append(n.PodsWithRequiredAntiAffinity, podInfo)
	}
	n.update(podInfo, 1)
}

// AddPod is a wrapper around AddPodInfo.
func (n *NodeInfo) AddPod(pod *v1.Pod) {
	// ignore this err since apiserver doesn't properly validate affinity terms
	// and we can't fix the validation for backwards compatibility.
	podInfo, _ := NewPodInfo(pod)
	n.AddPodInfo(podInfo)
}

func podWithAffinity(p *v1.Pod) bool {
	affinity := p.Spec.Affinity
	return affinity != nil && (affinity.PodAffinity != nil || affinity.PodAntiAffinity != nil)
}

func podWithRequiredAntiAffinity(p *v1.Pod) bool {
	affinity := p.Spec.Affinity
	return affinity != nil && affinity.PodAntiAffinity != nil &&
		len(affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0
}

func removeFromSlice(logger klog.Logger, s []fwk.PodInfo, k string) ([]fwk.PodInfo, fwk.PodInfo) {
	var removedPod fwk.PodInfo
	for i := range s {
		tmpKey, err := GetPodKey(s[i].GetPod())
		if err != nil {
			utilruntime.HandleErrorWithLogger(logger, err, "Cannot get pod key", "pod", klog.KObj(s[i].GetPod()))
			continue
		}
		if k == tmpKey {
			removedPod = s[i]
			// delete the element
			s[i] = s[len(s)-1]
			s = s[:len(s)-1]
			break
		}
	}
	// resets the slices to nil so that we can do DeepEqual in unit tests.
	if len(s) == 0 {
		return nil, removedPod
	}
	return s, removedPod
}

// RemovePod subtracts pod information from this NodeInfo.
func (n *NodeInfo) RemovePod(logger klog.Logger, pod *v1.Pod) error {
	k, err := GetPodKey(pod)
	if err != nil {
		return err
	}
	if podWithAffinity(pod) {
		n.PodsWithAffinity, _ = removeFromSlice(logger, n.PodsWithAffinity, k)
	}
	if podWithRequiredAntiAffinity(pod) {
		n.PodsWithRequiredAntiAffinity, _ = removeFromSlice(logger, n.PodsWithRequiredAntiAffinity, k)
	}

	var removedPod fwk.PodInfo
	if n.Pods, removedPod = removeFromSlice(logger, n.Pods, k); removedPod != nil {
		n.update(removedPod, -1)
		return nil
	}
	return fmt.Errorf("no corresponding pod %s in pods of node %s", pod.Name, n.node.Name)
}

// update node info based on the pod, and sign.
// The sign will be set to `+1` when AddPod and to `-1` when RemovePod.
func (n *NodeInfo) update(podInfo fwk.PodInfo, sign int64) {
	podResource := podInfo.CalculateResource()
	n.Requested.MilliCPU += sign * podResource.Resource.GetMilliCPU()
	n.Requested.Memory += sign * podResource.Resource.GetMemory()
	n.Requested.EphemeralStorage += sign * podResource.Resource.GetEphemeralStorage()
	if n.Requested.ScalarResources == nil && len(podResource.Resource.GetScalarResources()) > 0 {
		n.Requested.ScalarResources = map[v1.ResourceName]int64{}
	}
	for rName, rQuant := range podResource.Resource.GetScalarResources() {
		n.Requested.ScalarResources[rName] += sign * rQuant
	}
	n.NonZeroRequested.MilliCPU += sign * podResource.Non0CPU
	n.NonZeroRequested.Memory += sign * podResource.Non0Mem

	// Consume ports when pod added or release ports when pod removed.
	n.updateUsedPorts(podInfo.GetPod(), sign > 0)
	n.updatePVCRefCounts(podInfo.GetPod(), sign > 0)

	n.Generation = nextGeneration()
}

// updateUsedPorts updates the UsedPorts of NodeInfo.
func (n *NodeInfo) updateUsedPorts(pod *v1.Pod, add bool) {
	for _, port := range schedutil.GetHostPorts(pod) {
		if add {
			n.UsedPorts.Add(port.HostIP, string(port.Protocol), port.HostPort)
		} else {
			n.UsedPorts.Remove(port.HostIP, string(port.Protocol), port.HostPort)
		}
	}
}

// updatePVCRefCounts updates the PVCRefCounts of NodeInfo.
func (n *NodeInfo) updatePVCRefCounts(pod *v1.Pod, add bool) {
	for _, v := range pod.Spec.Volumes {
		if v.PersistentVolumeClaim == nil {
			continue
		}

		key := GetNamespacedName(pod.Namespace, v.PersistentVolumeClaim.ClaimName)
		if add {
			n.PVCRefCounts[key] += 1
		} else {
			n.PVCRefCounts[key] -= 1
			if n.PVCRefCounts[key] <= 0 {
				delete(n.PVCRefCounts, key)
			}
		}
	}
}

// SetNode sets the overall node information.
func (n *NodeInfo) SetNode(node *v1.Node) {
	n.node = node
	n.Allocatable = NewResource(node.Status.Allocatable)
	n.Generation = nextGeneration()
}

// RemoveNode removes the node object, leaving all other tracking information.
func (n *NodeInfo) RemoveNode() {
	n.node = nil
	n.Generation = nextGeneration()
}

// nextGeneration: Let's make sure history never forgets the name...
// Increments the generation number monotonically ensuring that generation numbers never collide.
// Collision of the generation numbers would be particularly problematic if a node was deleted and
// added back with the same name. See issue#63262.
func nextGeneration() int64 {
	return atomic.AddInt64(&generation, 1)
}

// QueuedPodInfo is a Pod wrapper with additional information related to
// the pod's status in the scheduling queue, such as the timestamp when
// it's added to the queue.
type QueuedPodInfo struct {
	*PodInfo
	// The time pod added to the scheduling queue.
	Timestamp time.Time
	// Number of all schedule attempts before successfully scheduled.
	// It's used to record the # attempts metric.
	Attempts int
	// BackoffExpiration is the time when the Pod will complete its backoff.
	// If the SchedulerPopFromBackoffQ feature is enabled, the value is aligned to the backoff ordering window.
	// Then, two Pods with the same BackoffExpiration (time bucket) are ordered by priority and eventually the timestamp,
	// to make sure popping from the backoffQ considers priority of pods that are close to the expiration time.
	BackoffExpiration time.Time
	// The total number of the scheduling attempts that this Pod gets unschedulable.
	// Basically it equals Attempts, but when the Pod fails with the Error status (e.g., the network error),
	// this count won't be incremented.
	// It's used to calculate the backoff time this Pod is obliged to get before retrying.
	UnschedulableCount int
	// The number of the error status that this Pod gets sequentially.
	// This count is reset when the Pod gets another status than Error.
	//
	// If the error status is returned (e.g., kube-apiserver is unstable), we don't want to immediately retry the Pod and hence need a backoff retry mechanism
	// because that might push more burden to the kube-apiserver.
	// But, we don't want to calculate the backoff time in the same way as the normal unschedulable reason
	// since the purpose is different; the backoff for a unschedulable status etc is for the punishment of wasting the scheduling cycles,
	// whereas the backoff for the error status is for the protection of the kube-apiserver.
	// That's why we need to distinguish ConsecutiveErrorsCount for the error status and UnschedulableCount for the unschedulable status.
	// See https://github.com/kubernetes/kubernetes/issues/128744 for the discussion.
	ConsecutiveErrorsCount int
	// The time when the pod is added to the queue for the first time. The pod may be added
	// back to the queue multiple times before it's successfully scheduled.
	// It shouldn't be updated once initialized. It's used to record the e2e scheduling
	// latency for a pod.
	InitialAttemptTimestamp *time.Time
	// UnschedulablePlugins records the plugin names that the Pod failed with Unschedulable or UnschedulableAndUnresolvable status
	// at specific extension points: PreFilter, Filter, Reserve, or Permit (WaitOnPermit).
	// If Pods are rejected at other extension points,
	// they're assumed to be unexpected errors (e.g., temporal network issue, plugin implementation issue, etc)
	// and retried soon after a backoff period.
	// That is because such failures could be solved regardless of incoming cluster events (registered in EventsToRegister).
	UnschedulablePlugins sets.Set[string]
	// PendingPlugins records the plugin names that the Pod failed with Pending status.
	PendingPlugins sets.Set[string]
	// GatingPlugin records the plugin name that gated the Pod at PreEnqueue.
	GatingPlugin string
	// GatingPluginEvents records the events registered by the plugin that gated the Pod at PreEnqueue.
	// We have it as a cache purpose to avoid re-computing which event(s) might ungate the Pod.
	GatingPluginEvents []fwk.ClusterEvent
}

func (pqi *QueuedPodInfo) GetPodInfo() fwk.PodInfo {
	return pqi.PodInfo
}

func (pqi *QueuedPodInfo) GetTimestamp() time.Time {
	return pqi.Timestamp
}

func (pqi *QueuedPodInfo) GetAttempts() int {
	return pqi.Attempts
}

func (pqi *QueuedPodInfo) GetBackoffExpiration() time.Time {
	return pqi.BackoffExpiration
}

func (pqi *QueuedPodInfo) GetUnschedulableCount() int {
	return pqi.UnschedulableCount
}

func (pqi *QueuedPodInfo) GetConsecutiveErrorsCount() int {
	return pqi.ConsecutiveErrorsCount
}

func (pqi *QueuedPodInfo) GetInitialAttemptTimestamp() *time.Time {
	return pqi.InitialAttemptTimestamp
}

func (pqi *QueuedPodInfo) GetUnschedulablePlugins() sets.Set[string] {
	return pqi.UnschedulablePlugins
}

func (pqi *QueuedPodInfo) GetPendingPlugins() sets.Set[string] {
	return pqi.PendingPlugins
}

func (pqi *QueuedPodInfo) GetGatingPlugin() string {
	return pqi.GatingPlugin
}

func (pqi *QueuedPodInfo) GetGatingPluginEvents() []fwk.ClusterEvent {
	return pqi.GatingPluginEvents
}

// Gated returns true if the pod is gated by any plugin.
func (pqi *QueuedPodInfo) Gated() bool {
	return pqi.GatingPlugin != ""
}

// DeepCopy returns a deep copy of the QueuedPodInfo object.
func (pqi *QueuedPodInfo) DeepCopy() *QueuedPodInfo {
	return &QueuedPodInfo{
		PodInfo:                 pqi.PodInfo.DeepCopy(),
		Timestamp:               pqi.Timestamp,
		Attempts:                pqi.Attempts,
		UnschedulableCount:      pqi.UnschedulableCount,
		InitialAttemptTimestamp: pqi.InitialAttemptTimestamp,
		UnschedulablePlugins:    pqi.UnschedulablePlugins.Clone(),
		BackoffExpiration:       pqi.BackoffExpiration,
		GatingPlugin:            pqi.GatingPlugin,
		GatingPluginEvents:      slices.Clone(pqi.GatingPluginEvents),
		PendingPlugins:          pqi.PendingPlugins.Clone(),
		ConsecutiveErrorsCount:  pqi.ConsecutiveErrorsCount,
	}
}

// PodInfo is a wrapper to a Pod with additional pre-computed information to
// accelerate processing. This information is typically immutable (e.g., pre-processed
// inter-pod affinity selectors).
type PodInfo struct {
	Pod                        *v1.Pod
	RequiredAffinityTerms      []fwk.AffinityTerm
	RequiredAntiAffinityTerms  []fwk.AffinityTerm
	PreferredAffinityTerms     []fwk.WeightedAffinityTerm
	PreferredAntiAffinityTerms []fwk.WeightedAffinityTerm
	// cachedResource contains precomputed resources for Pod (podResource).
	// The value can change only if InPlacePodVerticalScaling is enabled.
	// In that case, the whole PodInfo object is recreated (for assigned pods in cache).
	// cachedResource contains a podResource, computed when adding a scheduled pod to NodeInfo.
	// When removing a pod from a NodeInfo, i.e. finding victims for preemption or removing a pod from a cluster,
	// cachedResource is used instead, what provides a noticeable performance boost.
	// Note: cachedResource field shouldn't be accessed directly.
	// Use calculateResource method to obtain it instead.
	cachedResource *fwk.PodResource
}

func (pi *PodInfo) GetPod() *v1.Pod {
	return pi.Pod
}

func (pi *PodInfo) GetRequiredAffinityTerms() []fwk.AffinityTerm {
	return pi.RequiredAffinityTerms
}

func (pi *PodInfo) GetRequiredAntiAffinityTerms() []fwk.AffinityTerm {
	return pi.RequiredAntiAffinityTerms
}

func (pi *PodInfo) GetPreferredAffinityTerms() []fwk.WeightedAffinityTerm {
	return pi.PreferredAffinityTerms
}

func (pi *PodInfo) GetPreferredAntiAffinityTerms() []fwk.WeightedAffinityTerm {
	return pi.PreferredAntiAffinityTerms
}

// DeepCopy returns a deep copy of the PodInfo object.
func (pi *PodInfo) DeepCopy() *PodInfo {
	return &PodInfo{
		Pod:                        pi.Pod.DeepCopy(),
		RequiredAffinityTerms:      pi.RequiredAffinityTerms,
		RequiredAntiAffinityTerms:  pi.RequiredAntiAffinityTerms,
		PreferredAffinityTerms:     pi.PreferredAffinityTerms,
		PreferredAntiAffinityTerms: pi.PreferredAntiAffinityTerms,
		cachedResource:             pi.cachedResource,
	}
}

// Update creates a full new PodInfo by default. And only updates the pod when the PodInfo
// has been instantiated and the passed pod is the exact same one as the original pod.
func (pi *PodInfo) Update(pod *v1.Pod) error {
	if pod != nil && pi.Pod != nil && pi.Pod.UID == pod.UID {
		// PodInfo includes immutable information, and so it is safe to update the pod in place if it is
		// the exact same pod
		pi.Pod = pod
		return nil
	}
	var preferredAffinityTerms []v1.WeightedPodAffinityTerm
	var preferredAntiAffinityTerms []v1.WeightedPodAffinityTerm
	if affinity := pod.Spec.Affinity; affinity != nil {
		if a := affinity.PodAffinity; a != nil {
			preferredAffinityTerms = a.PreferredDuringSchedulingIgnoredDuringExecution
		}
		if a := affinity.PodAntiAffinity; a != nil {
			preferredAntiAffinityTerms = a.PreferredDuringSchedulingIgnoredDuringExecution
		}
	}

	// Attempt to parse the affinity terms
	var parseErrs []error
	requiredAffinityTerms, err := GetAffinityTerms(pod, GetPodAffinityTerms(pod.Spec.Affinity))
	if err != nil {
		parseErrs = append(parseErrs, fmt.Errorf("requiredAffinityTerms: %w", err))
	}
	requiredAntiAffinityTerms, err := GetAffinityTerms(pod,
		GetPodAntiAffinityTerms(pod.Spec.Affinity))
	if err != nil {
		parseErrs = append(parseErrs, fmt.Errorf("requiredAntiAffinityTerms: %w", err))
	}
	weightedAffinityTerms, err := getWeightedAffinityTerms(pod, preferredAffinityTerms)
	if err != nil {
		parseErrs = append(parseErrs, fmt.Errorf("preferredAffinityTerms: %w", err))
	}
	weightedAntiAffinityTerms, err := getWeightedAffinityTerms(pod, preferredAntiAffinityTerms)
	if err != nil {
		parseErrs = append(parseErrs, fmt.Errorf("preferredAntiAffinityTerms: %w", err))
	}

	pi.Pod = pod
	pi.RequiredAffinityTerms = requiredAffinityTerms
	pi.RequiredAntiAffinityTerms = requiredAntiAffinityTerms
	pi.PreferredAffinityTerms = weightedAffinityTerms
	pi.PreferredAntiAffinityTerms = weightedAntiAffinityTerms
	pi.cachedResource = nil
	return utilerrors.NewAggregate(parseErrs)
}

func (pi *PodInfo) CalculateResource() fwk.PodResource {
	if pi.cachedResource != nil {
		return *pi.cachedResource
	}
	inPlacePodVerticalScalingEnabled := utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling)
	podLevelResourcesEnabled := utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources)
	requests := resourcehelper.PodRequests(pi.Pod, resourcehelper.PodResourcesOptions{
		UseStatusResources: inPlacePodVerticalScalingEnabled,
		// SkipPodLevelResources is set to false when PodLevelResources feature is enabled.
		SkipPodLevelResources: !podLevelResourcesEnabled,
	})
	isPodLevelResourcesSet := podLevelResourcesEnabled && resourcehelper.IsPodLevelRequestsSet(pi.Pod)
	nonMissingContainerRequests := getNonMissingContainerRequests(requests, isPodLevelResourcesSet)
	non0Requests := requests
	if len(nonMissingContainerRequests) > 0 {
		non0Requests = resourcehelper.PodRequests(pi.Pod, resourcehelper.PodResourcesOptions{
			UseStatusResources: inPlacePodVerticalScalingEnabled,
			// SkipPodLevelResources is set to false when PodLevelResources feature is enabled.
			SkipPodLevelResources:       !podLevelResourcesEnabled,
			NonMissingContainerRequests: nonMissingContainerRequests,
		})
	}
	non0CPU := non0Requests[v1.ResourceCPU]
	non0Mem := non0Requests[v1.ResourceMemory]

	var res Resource
	res.Add(requests)
	podResource := fwk.PodResource{
		Resource: &res,
		Non0CPU:  non0CPU.MilliValue(),
		Non0Mem:  non0Mem.Value(),
	}
	pi.cachedResource = &podResource
	return podResource
}

// ExtenderName is a fake plugin name put in UnschedulablePlugins when Extender rejected some Nodes.
const ExtenderName = "Extender"

// Diagnosis records the details to diagnose a scheduling failure.
type Diagnosis struct {
	// NodeToStatus records the status of nodes and generic status for absent ones.
	// if they're rejected in PreFilter (via PreFilterResult) or Filter plugins.
	// Nodes that pass PreFilter/Filter plugins are not included in this map.
	NodeToStatus *NodeToStatus
	// UnschedulablePlugins are plugins that returns Unschedulable or UnschedulableAndUnresolvable.
	UnschedulablePlugins sets.Set[string]
	// UnschedulablePlugins are plugins that returns Pending.
	PendingPlugins sets.Set[string]
	// PreFilterMsg records the messages returned from PreFilter plugins.
	PreFilterMsg string
	// PostFilterMsg records the messages returned from PostFilter plugins.
	PostFilterMsg string
}

// FitError describes a fit error of a pod.
type FitError struct {
	Pod         *v1.Pod
	NumAllNodes int
	Diagnosis   Diagnosis
}

const (
	// NoNodeAvailableMsg is used to format message when no nodes available.
	NoNodeAvailableMsg = "0/%v nodes are available"
)

func (d *Diagnosis) AddPluginStatus(sts *fwk.Status) {
	if sts.Plugin() == "" {
		return
	}
	if sts.IsRejected() {
		if d.UnschedulablePlugins == nil {
			d.UnschedulablePlugins = sets.New[string]()
		}
		d.UnschedulablePlugins.Insert(sts.Plugin())
	}
	if sts.Code() == fwk.Pending {
		if d.PendingPlugins == nil {
			d.PendingPlugins = sets.New[string]()
		}
		d.PendingPlugins.Insert(sts.Plugin())
	}
}

// Error returns detailed information of why the pod failed to fit on each node.
// A message format is "0/X nodes are available: <PreFilterMsg>. <FilterMsg>. <PostFilterMsg>."
func (f *FitError) Error() string {
	reasonMsg := fmt.Sprintf(NoNodeAvailableMsg+":", f.NumAllNodes)
	preFilterMsg := f.Diagnosis.PreFilterMsg
	if preFilterMsg != "" {
		// PreFilter plugin returns unschedulable.
		// Add the messages from PreFilter plugins to reasonMsg.
		reasonMsg += fmt.Sprintf(" %v.", preFilterMsg)
	}

	if preFilterMsg == "" {
		// the scheduling cycle went through PreFilter extension point successfully.
		//
		// When the prefilter plugin returns unschedulable,
		// the scheduling framework inserts the same unschedulable status to all nodes in NodeToStatusMap.
		// So, we shouldn't add the message from NodeToStatusMap when the PreFilter failed.
		// Otherwise, we will have duplicated reasons in the error message.
		reasons := make(map[string]int)
		f.Diagnosis.NodeToStatus.ForEachExplicitNode(func(_ string, status *fwk.Status) {
			for _, reason := range status.Reasons() {
				reasons[reason]++
			}
		})
		if f.Diagnosis.NodeToStatus.Len() < f.NumAllNodes {
			// Adding predefined reasons for nodes that are absent in NodeToStatusMap
			for _, reason := range f.Diagnosis.NodeToStatus.AbsentNodesStatus().Reasons() {
				reasons[reason] += f.NumAllNodes - f.Diagnosis.NodeToStatus.Len()
			}
		}

		sortReasonsHistogram := func() []string {
			var reasonStrings []string
			for k, v := range reasons {
				reasonStrings = append(reasonStrings, fmt.Sprintf("%v %v", v, k))
			}
			sort.Strings(reasonStrings)
			return reasonStrings
		}
		sortedFilterMsg := sortReasonsHistogram()
		if len(sortedFilterMsg) != 0 {
			reasonMsg += fmt.Sprintf(" %v.", strings.Join(sortedFilterMsg, ", "))
		}
	}

	// Add the messages from PostFilter plugins to reasonMsg.
	// We can add this message regardless of whether the scheduling cycle fails at PreFilter or Filter
	// since we may run PostFilter (if enabled) in both cases.
	postFilterMsg := f.Diagnosis.PostFilterMsg
	if postFilterMsg != "" {
		reasonMsg += fmt.Sprintf(" %v", postFilterMsg)
	}
	return reasonMsg
}

func newAffinityTerm(pod *v1.Pod, term *v1.PodAffinityTerm) (*fwk.AffinityTerm, error) {
	selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
	if err != nil {
		return nil, err
	}

	namespaces := getNamespacesFromPodAffinityTerm(pod, term)
	nsSelector, err := metav1.LabelSelectorAsSelector(term.NamespaceSelector)
	if err != nil {
		return nil, err
	}

	return &fwk.AffinityTerm{Namespaces: namespaces, Selector: selector, TopologyKey: term.TopologyKey, NamespaceSelector: nsSelector}, nil
}

// GetAffinityTerms receives a Pod and affinity terms and returns the namespaces and
// selectors of the terms.
func GetAffinityTerms(pod *v1.Pod, v1Terms []v1.PodAffinityTerm) ([]fwk.AffinityTerm, error) {
	if v1Terms == nil {
		return nil, nil
	}

	var terms []fwk.AffinityTerm
	for i := range v1Terms {
		t, err := newAffinityTerm(pod, &v1Terms[i])
		if err != nil {
			// We get here if the label selector failed to process
			return nil, err
		}
		terms = append(terms, *t)
	}
	return terms, nil
}

// getWeightedAffinityTerms returns the list of processed affinity terms.
func getWeightedAffinityTerms(pod *v1.Pod, v1Terms []v1.WeightedPodAffinityTerm) ([]fwk.WeightedAffinityTerm, error) {
	if v1Terms == nil {
		return nil, nil
	}

	var terms []fwk.WeightedAffinityTerm
	for i := range v1Terms {
		t, err := newAffinityTerm(pod, &v1Terms[i].PodAffinityTerm)
		if err != nil {
			// We get here if the label selector failed to process
			return nil, err
		}
		terms = append(terms, fwk.WeightedAffinityTerm{AffinityTerm: *t, Weight: v1Terms[i].Weight})
	}
	return terms, nil
}

// NewPodInfo returns a new PodInfo.
func NewPodInfo(pod *v1.Pod) (*PodInfo, error) {
	pInfo := &PodInfo{}
	err := pInfo.Update(pod)
	return pInfo, err
}

func GetPodAffinityTerms(affinity *v1.Affinity) (terms []v1.PodAffinityTerm) {
	if affinity != nil && affinity.PodAffinity != nil {
		if len(affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 {
			terms = affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
		}
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		// if len(affinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
		//	terms = append(terms, affinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
		// }
	}
	return terms
}

func GetPodAntiAffinityTerms(affinity *v1.Affinity) (terms []v1.PodAffinityTerm) {
	if affinity != nil && affinity.PodAntiAffinity != nil {
		if len(affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 {
			terms = affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution
		}
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		// if len(affinity.PodAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
		//	terms = append(terms, affinity.PodAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
		// }
	}
	return terms
}

// returns a set of names according to the namespaces indicated in podAffinityTerm.
// If namespaces is empty it considers the given pod's namespace.
func getNamespacesFromPodAffinityTerm(pod *v1.Pod, podAffinityTerm *v1.PodAffinityTerm) sets.Set[string] {
	names := sets.Set[string]{}
	if len(podAffinityTerm.Namespaces) == 0 && podAffinityTerm.NamespaceSelector == nil {
		names.Insert(pod.Namespace)
	} else {
		names.Insert(podAffinityTerm.Namespaces...)
	}
	return names
}

// Resource is a collection of compute resource.
// Implementation is separate from interface fwk.Resource, because implementation of functions Add and SetMaxResource
// depends on internal scheduler util functions.
type Resource struct {
	MilliCPU         int64
	Memory           int64
	EphemeralStorage int64
	// We store allowedPodNumber (which is Node.Status.Allocatable.Pods().Value())
	// explicitly as int, to avoid conversions and improve performance.
	AllowedPodNumber int
	// ScalarResources
	ScalarResources map[v1.ResourceName]int64
}

func (r *Resource) GetMilliCPU() int64 {
	return r.MilliCPU
}

func (r *Resource) GetMemory() int64 {
	return r.Memory
}

func (r *Resource) GetEphemeralStorage() int64 {
	return r.EphemeralStorage
}

func (r *Resource) GetAllowedPodNumber() int {
	return r.AllowedPodNumber
}

func (r *Resource) GetScalarResources() map[v1.ResourceName]int64 {
	return r.ScalarResources
}

// NewResource creates a Resource from ResourceList
func NewResource(rl v1.ResourceList) *Resource {
	r := &Resource{}
	r.Add(rl)
	return r
}

// Add adds ResourceList into Resource.
func (r *Resource) Add(rl v1.ResourceList) {
	if r == nil {
		return
	}

	for rName, rQuant := range rl {
		switch rName {
		case v1.ResourceCPU:
			r.MilliCPU += rQuant.MilliValue()
		case v1.ResourceMemory:
			r.Memory += rQuant.Value()
		case v1.ResourcePods:
			r.AllowedPodNumber += int(rQuant.Value())
		case v1.ResourceEphemeralStorage:
			r.EphemeralStorage += rQuant.Value()
		default:
			if schedutil.IsScalarResourceName(rName) {
				r.AddScalar(rName, rQuant.Value())
			}
		}
	}
}

// Clone returns a copy of this resource.
func (r *Resource) Clone() *Resource {
	res := &Resource{
		MilliCPU:         r.MilliCPU,
		Memory:           r.Memory,
		AllowedPodNumber: r.AllowedPodNumber,
		EphemeralStorage: r.EphemeralStorage,
	}
	if r.ScalarResources != nil {
		res.ScalarResources = make(map[v1.ResourceName]int64, len(r.ScalarResources))
		for k, v := range r.ScalarResources {
			res.ScalarResources[k] = v
		}
	}
	return res
}

// AddScalar adds a resource by a scalar value of this resource.
func (r *Resource) AddScalar(name v1.ResourceName, quantity int64) {
	r.SetScalar(name, r.ScalarResources[name]+quantity)
}

// SetScalar sets a resource by a scalar value of this resource.
func (r *Resource) SetScalar(name v1.ResourceName, quantity int64) {
	// Lazily allocate scalar resource map.
	if r.ScalarResources == nil {
		r.ScalarResources = map[v1.ResourceName]int64{}
	}
	r.ScalarResources[name] = quantity
}

// SetMaxResource compares with ResourceList and takes max value for each Resource.
func (r *Resource) SetMaxResource(rl v1.ResourceList) {
	if r == nil {
		return
	}

	for rName, rQuantity := range rl {
		switch rName {
		case v1.ResourceMemory:
			r.Memory = max(r.Memory, rQuantity.Value())
		case v1.ResourceCPU:
			r.MilliCPU = max(r.MilliCPU, rQuantity.MilliValue())
		case v1.ResourceEphemeralStorage:
			r.EphemeralStorage = max(r.EphemeralStorage, rQuantity.Value())
		default:
			if schedutil.IsScalarResourceName(rName) {
				r.SetScalar(rName, max(r.ScalarResources[rName], rQuantity.Value()))
			}
		}
	}
}

// NewNodeInfo returns a ready to use empty NodeInfo object.
// If any pods are given in arguments, their information will be aggregated in
// the returned object.
func NewNodeInfo(pods ...*v1.Pod) *NodeInfo {
	ni := &NodeInfo{
		Requested:        &Resource{},
		NonZeroRequested: &Resource{},
		Allocatable:      &Resource{},
		Generation:       nextGeneration(),
		UsedPorts:        make(fwk.HostPortInfo),
		ImageStates:      make(map[string]*fwk.ImageStateSummary),
		PVCRefCounts:     make(map[string]int),
	}
	for _, pod := range pods {
		ni.AddPod(pod)
	}
	return ni
}

// getNonMissingContainerRequests returns the default non-zero CPU and memory
// requests for a container that the scheduler uses when container-level and
// pod-level requests are not set for a resource. It returns a ResourceList that
// includes these default non-zero requests, which are essential for the
// scheduler to function correctly.
// The method's behavior depends on whether pod-level resources are set or not:
// 1. When the pod level resources are not set, the method returns a ResourceList
// with the following defaults:
//   - CPU: schedutil.DefaultMilliCPURequest
//   - Memory: schedutil.DefaultMemoryRequest
//
// These defaults ensure that each container has a minimum resource request,
// allowing the scheduler to aggregate these requests and find a suitable node
// for the pod.
//
// 2. When the pod level resources are set, if a CPU or memory request is
// missing at the container-level *and* at the pod-level, the corresponding
// default value (schedutil.DefaultMilliCPURequest or schedutil.DefaultMemoryRequest)
// is included in the returned ResourceList.
// Note that these default values are not set in the Pod object itself, they are only used
// by the scheduler during node selection.
func getNonMissingContainerRequests(requests v1.ResourceList, podLevelResourcesSet bool) v1.ResourceList {
	if !podLevelResourcesSet {
		return v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(schedutil.DefaultMilliCPURequest, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(schedutil.DefaultMemoryRequest, resource.DecimalSI),
		}
	}

	nonMissingContainerRequests := make(v1.ResourceList, 2)
	// DefaultMilliCPURequest serves as the fallback value when both
	// pod-level and container-level CPU requests are not set.
	// Note that the apiserver defaulting logic will propagate a non-zero
	// container-level CPU request to the pod level if a pod-level request
	// is not explicitly set.
	if _, exists := requests[v1.ResourceCPU]; !exists {
		nonMissingContainerRequests[v1.ResourceCPU] = *resource.NewMilliQuantity(schedutil.DefaultMilliCPURequest, resource.DecimalSI)
	}

	// DefaultMemoryRequest serves as the fallback value when both
	// pod-level and container-level CPU requests are unspecified.
	// Note that the apiserver defaulting logic will propagate a non-zero
	// container-level memory request to the pod level if a pod-level request
	// is not explicitly set.
	if _, exists := requests[v1.ResourceMemory]; !exists {
		nonMissingContainerRequests[v1.ResourceMemory] = *resource.NewQuantity(schedutil.DefaultMemoryRequest, resource.DecimalSI)
	}
	return nonMissingContainerRequests

}

// GetPodKey returns the string key of a pod.
func GetPodKey(pod *v1.Pod) (string, error) {
	uid := string(pod.UID)
	if len(uid) == 0 {
		return "", errors.New("cannot get cache key for pod with empty UID")
	}
	return uid, nil
}

// GetNamespacedName returns the string format of a namespaced resource name.
func GetNamespacedName(namespace, name string) string {
	return fmt.Sprintf("%s/%s", namespace, name)
}
