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

// This file defines the scheduling framework plugin interfaces.

package framework

import (
	"context"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// NodeToStatus contains the statuses of the Nodes where the incoming Pod was not schedulable.
type NodeToStatus struct {
	// nodeToStatus contains specific statuses of the nodes.
	nodeToStatus map[string]*fwk.Status
	// absentNodesStatus defines a status for all nodes that are absent in nodeToStatus map.
	// By default, all absent nodes are UnschedulableAndUnresolvable.
	absentNodesStatus *fwk.Status
}

// NewDefaultNodeToStatus creates NodeToStatus without any node in the map.
// The absentNodesStatus is set by default to UnschedulableAndUnresolvable.
func NewDefaultNodeToStatus() *NodeToStatus {
	return NewNodeToStatus(make(map[string]*fwk.Status), fwk.NewStatus(fwk.UnschedulableAndUnresolvable))
}

// NewNodeToStatus creates NodeToStatus initialized with given nodeToStatus and absentNodesStatus.
func NewNodeToStatus(nodeToStatus map[string]*fwk.Status, absentNodesStatus *fwk.Status) *NodeToStatus {
	return &NodeToStatus{
		nodeToStatus:      nodeToStatus,
		absentNodesStatus: absentNodesStatus,
	}
}

// Get returns the status for given nodeName. If the node is not in the map, the absentNodesStatus is returned.
func (m *NodeToStatus) Get(nodeName string) *fwk.Status {
	if status, ok := m.nodeToStatus[nodeName]; ok {
		return status
	}
	return m.absentNodesStatus
}

// Set sets status for given nodeName.
func (m *NodeToStatus) Set(nodeName string, status *fwk.Status) {
	m.nodeToStatus[nodeName] = status
}

// Len returns length of nodeToStatus map. It is not aware of number of absent nodes.
func (m *NodeToStatus) Len() int {
	return len(m.nodeToStatus)
}

// AbsentNodesStatus returns absentNodesStatus value.
func (m *NodeToStatus) AbsentNodesStatus() *fwk.Status {
	return m.absentNodesStatus
}

// SetAbsentNodesStatus sets absentNodesStatus value.
func (m *NodeToStatus) SetAbsentNodesStatus(status *fwk.Status) {
	m.absentNodesStatus = status
}

// ForEachExplicitNode runs fn for each node which status is explicitly set.
// Imporatant note, it runs the fn only for nodes with a status explicitly registered,
// and hence may not run the fn for all existing nodes.
// For example, if PreFilter rejects all Nodes, the scheduler would NOT set a failure status to every Node,
// but set a failure status as AbsentNodesStatus.
// You're supposed to get a status from AbsentNodesStatus(), and consider all other nodes that are rejected by them.
func (m *NodeToStatus) ForEachExplicitNode(fn func(nodeName string, status *fwk.Status)) {
	for nodeName, status := range m.nodeToStatus {
		fn(nodeName, status)
	}
}

// NodesForStatusCode returns a list of NodeInfos for the nodes that matches a given status code.
// If the absentNodesStatus matches the code, all existing nodes are fetched using nodeLister
// and filtered using NodeToStatus.Get.
// If the absentNodesStatus doesn't match the code, nodeToStatus map is used to create a list of nodes
// and nodeLister.Get is used to obtain NodeInfo for each.
func (m *NodeToStatus) NodesForStatusCode(nodeLister fwk.NodeInfoLister, code fwk.Code) ([]fwk.NodeInfo, error) {
	var resultNodes []fwk.NodeInfo

	if m.AbsentNodesStatus().Code() == code {
		allNodes, err := nodeLister.List()
		if err != nil {
			return nil, err
		}
		if m.Len() == 0 {
			// All nodes are absent and status code is matching, so can return all nodes.
			return allNodes, nil
		}
		// Need to find all the nodes that are absent or have a matching code using the allNodes.
		for _, node := range allNodes {
			nodeName := node.Node().Name
			if status := m.Get(nodeName); status.Code() == code {
				resultNodes = append(resultNodes, node)
			}
		}
		return resultNodes, nil
	}

	m.ForEachExplicitNode(func(nodeName string, status *fwk.Status) {
		if status.Code() == code {
			if nodeInfo, err := nodeLister.Get(nodeName); err == nil {
				resultNodes = append(resultNodes, nodeInfo)
			}
		}
	})

	return resultNodes, nil
}

// PodsToActivateKey is a reserved state key for stashing pods.
// If the stashed pods are present in unschedulablePods or backoffQ，they will be
// activated (i.e., moved to activeQ) in two phases:
// - end of a scheduling cycle if it succeeds (will be cleared from `PodsToActivate` if activated)
// - end of a binding cycle if it succeeds
var PodsToActivateKey fwk.StateKey = "kubernetes.io/pods-to-activate"

// PodsToActivate stores pods to be activated.
type PodsToActivate struct {
	sync.Mutex
	// Map is keyed with namespaced pod name, and valued with the pod.
	Map map[string]*v1.Pod
}

// Clone just returns the same state.
func (s *PodsToActivate) Clone() fwk.StateData {
	return s
}

// NewPodsToActivate instantiates a PodsToActivate object.
func NewPodsToActivate() *PodsToActivate {
	return &PodsToActivate{Map: make(map[string]*v1.Pod)}
}

// SortedScoredNodes is a list of scored nodes, returned from scheduling.
type SortedScoredNodes interface {
	Pop() string
	Len() int
}

// Framework manages the set of plugins in use by the scheduling framework.
// Configured plugins are called at specified points in a scheduling context.
type Framework interface {
	fwk.Handle

	// PreEnqueuePlugins returns the registered preEnqueue plugins.
	PreEnqueuePlugins() []fwk.PreEnqueuePlugin

	// EnqueueExtensions returns the registered Enqueue extensions.
	EnqueueExtensions() []fwk.EnqueueExtensions

	// QueueSortFunc returns the function to sort pods in scheduling queue
	QueueSortFunc() fwk.LessFunc

	// Create a scheduling signature for a given pod, if possible. Two pods with the same signature
	// should get the same feasibility and scores for any given set of nodes even after one of them gets assigned. If some plugins
	// are unable to create a signature, the pod may be "unsignable" which disables results caching
	// and gang scheduling optimizations.
	// See https://github.com/kubernetes/enhancements/tree/master/keps/sig-scheduling/5598-opportunistic-batching
	SignPod(ctx context.Context, pod *v1.Pod, recordPluginStats bool) fwk.PodSignature

	// RunPreFilterPlugins runs the set of configured PreFilter plugins. It returns
	// *fwk.Status and its code is set to non-success if any of the plugins returns
	// anything but Success. If a non-success status is returned, then the scheduling
	// cycle is aborted.
	// It also returns a PreFilterResult, which may influence what or how many nodes to
	// evaluate downstream.
	// The third returns value contains PreFilter plugin that rejected some or all Nodes with PreFilterResult.
	// But, note that it doesn't contain any plugin when a plugin rejects this Pod with non-success status,
	// not with PreFilterResult.
	RunPreFilterPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod) (*fwk.PreFilterResult, *fwk.Status, sets.Set[string])

	// RunPostFilterPlugins runs the set of configured PostFilter plugins.
	// PostFilter plugins can either be informational, in which case should be configured
	// to execute first and return Unschedulable status, or ones that try to change the
	// cluster state to make the pod potentially schedulable in a future scheduling cycle.
	RunPostFilterPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status)

	// Get a "node hint" for a given pod. A node hint is the name of a node provided by the batching code when information
	// from the previous scheduling cycle can be reused for this cycle.
	// If the batching code cannot provide a hint, the function returns "".
	// See git.k8s.io/enhancements/keps/sig-scheduling/5598-opportunistic-batching
	GetNodeHint(ctx context.Context, pod *v1.Pod, state fwk.CycleState, cycleCount int64) (hint string, signature fwk.PodSignature)

	// StoreScheduleResults stores the results after we have sorted and filtered nodes.
	StoreScheduleResults(ctx context.Context, signature fwk.PodSignature, hintedNode, chosenNode string, otherNodes SortedScoredNodes, cycleCount int64)

	// RunPreBindPlugins runs the set of configured PreBind plugins. It returns
	// *fwk.Status and its code is set to non-success if any of the plugins returns
	// anything but Success. If the Status code is "Unschedulable", it is
	// considered as a scheduling check failure, otherwise, it is considered as an
	// internal error. In either case the pod is not going to be bound.
	RunPreBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// RunPreBindPreFlights runs the set of configured PreBindPreFlight functions from PreBind plugins.
	// It returns immediately if any of the plugins returns a non-skip status.
	RunPreBindPreFlights(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// RunPostBindPlugins runs the set of configured PostBind plugins.
	RunPostBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string)

	// RunReservePluginsReserve runs the Reserve method of the set of
	// configured Reserve plugins. If any of these calls returns an error, it
	// does not continue running the remaining ones and returns the error. In
	// such case, pod will not be scheduled.
	RunReservePluginsReserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// RunReservePluginsUnreserve runs the Unreserve method of the set of
	// configured Reserve plugins.
	RunReservePluginsUnreserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string)

	// RunPermitPlugins runs the set of configured Permit plugins. If any of these
	// plugins returns a status other than "Success" or "Wait", it does not continue
	// running the remaining plugins and returns an error. Otherwise, if any of the
	// plugins returns "Wait", then this function will create and add waiting pod
	// to a map of currently waiting pods and return status with "Wait" code.
	// Pod will remain waiting pod for the minimum duration returned by the Permit plugins.
	RunPermitPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// WillWaitOnPermit returns whether this pod will wait on permit by checking if the pod is a waiting pod.
	WillWaitOnPermit(ctx context.Context, pod *v1.Pod) bool

	// WaitOnPermit will block, if the pod is a waiting pod, until the waiting pod is rejected or allowed.
	WaitOnPermit(ctx context.Context, pod *v1.Pod) *fwk.Status

	// RunBindPlugins runs the set of configured Bind plugins. A Bind plugin may choose
	// whether or not to handle the given Pod. If a Bind plugin chooses to skip the
	// binding, it should return code=5("skip") status. Otherwise, it should return "Error"
	// or "Success". If none of the plugins handled binding, RunBindPlugins returns
	// code=5("skip") status.
	RunBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// HasFilterPlugins returns true if at least one Filter plugin is defined.
	HasFilterPlugins() bool

	// HasPostFilterPlugins returns true if at least one PostFilter plugin is defined.
	HasPostFilterPlugins() bool

	// HasScorePlugins returns true if at least one Score plugin is defined.
	HasScorePlugins() bool

	// ListPlugins returns a map of extension point name to list of configured Plugins.
	ListPlugins() *config.Plugins

	// PercentageOfNodesToScore returns percentageOfNodesToScore associated to a profile.
	PercentageOfNodesToScore() *int32

	// SetPodNominator sets the PodNominator
	SetPodNominator(nominator fwk.PodNominator)
	// SetPodActivator sets the PodActivator
	SetPodActivator(activator fwk.PodActivator)
	// SetAPICacher sets the APICacher
	SetAPICacher(apiCacher fwk.APICacher)

	// Close calls Close method of each plugin.
	Close() error
}

func NewPostFilterResultWithNominatedNode(name string) *fwk.PostFilterResult {
	return &fwk.PostFilterResult{
		NominatingInfo: &fwk.NominatingInfo{
			NominatedNodeName: name,
			NominatingMode:    fwk.ModeOverride,
		},
	}
}
