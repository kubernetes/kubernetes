/*
Copyright The Kubernetes Authors.

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

package preemption

import (
	"fmt"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

type podGroupPreemptor struct {
	*fwk.GenericPodGroup
	pods             []*v1.Pod
	preemptionPolicy v1.PreemptionPolicy
}

func newPodGroupPreemptor(pgInfo fwk.PodGroupInfo, enablePodGroupPreemptionPolicy bool) *podGroupPreemptor {
	p := &podGroupPreemptor{
		GenericPodGroup: pgInfo.GetGenericPodGroup(),
		pods:            pgInfo.GetAllUnscheduledPods(),
	}
	p.preemptionPolicy = resolvePreemptionPolicy(pgInfo, p.pods, enablePodGroupPreemptionPolicy)
	return p
}

func resolvePreemptionPolicy(pgInfo fwk.PodGroupInfo, pods []*v1.Pod, enablePodGroupPreemptionPolicy bool) v1.PreemptionPolicy {
	if enablePodGroupPreemptionPolicy {
		return pgInfo.GetPreemptionPolicy()
	}
	for _, pod := range pods {
		if p := pod.Spec.PreemptionPolicy; p != nil && *p == v1.PreemptNever {
			return *p
		}
	}
	return v1.PreemptLowerPriority
}

// Members returns the list of Pods that belong to this preemptor.
func (p *podGroupPreemptor) Members() []*v1.Pod {
	return p.pods
}

// GetPreemptionPolicy returns a preemption policy of this preemptor.
func (p *podGroupPreemptor) GetPreemptionPolicy() v1.PreemptionPolicy {
	return p.preemptionPolicy
}

// domain represents the boundary or scope within which the preemption logic is evaluated.
// It abstracts the scheduling domain, which can range from a single Node (for standard Pod preemption)
// to a group of Nodes or the entire Cluster (for PodGroup preemption).
type domain struct {
	nodes              map[string]fwk.NodeInfo
	name               string
	allPossibleVictims []*DomainVictim
}

// Nodes returns a map of NodeInfo objects by node name that belong to this domain.
// The preemption logic uses this to check feasibility and resource availability
// within the specific scope.
func (d *domain) Nodes() map[string]fwk.NodeInfo {
	return d.nodes
}

// GetAllPossibleVictims returns all potential victims running within this domain (individual Pods or PodGroups).
func (d *domain) GetAllPossibleVictims() []*DomainVictim {
	return d.allPossibleVictims
}

// GetName returns a unique identifier for the domain.
// This is primarily used for logging and debugging purposes.
func (d *domain) GetName() string {
	return d.name
}

// getHighestAllAncestor returns the key of the highest ancestor in the hierarchy that has disruption mode All.
// It returns (key, true) if found, or (empty, false) if not found.
// TODO: log/return an error if there is a gap in the hierarchy.
func getHighestAllAncestor(pod *v1.Pod, pgLister fwk.PodGroupLister, cpgLister fwk.CompositePodGroupLister) (fwk.EntityKey, bool) {
	if pod.Spec.SchedulingGroup == nil || pgLister == nil {
		return fwk.EntityKey{}, false
	}
	startKey := fwk.PodGroupKey(pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
	var highestAllKey fwk.EntityKey
	var hasAll bool

	for gpg := range traverseHierarchyUp(pod.Namespace, startKey, pgLister, cpgLister) {
		if gpg.HasDisruptionModeAll() {
			highestAllKey = gpg.GetKey()
			hasAll = true
		}
	}

	return highestAllKey, hasAll
}

// createDomainVictims converts deduplicated Victim entries into DomainVictim objects
// by enriching them with node blast-radius metadata from the snapshot, allowing the preemption
// evaluator to assess the cluster-wide impact of evicting each candidate.
func createDomainVictims(snapshot fwk.SharedLister, victims []Victim) ([]*DomainVictim, error) {
	var allPossibleVictims []*DomainVictim
	for _, vi := range victims {
		v, err := newDomainVictim(snapshot, vi.Pods(), vi.Priority(), vi.Type())
		if err != nil {
			return nil, err
		}
		allPossibleVictims = append(allPossibleVictims, v)
	}
	return allPossibleVictims, nil
}

// newDomainForWorkloadPreemption creates a new domain for workload preemption.
// The domain is the whole cluster and it contains victims that are computed based
// on the pods and their scheduling groups.
// Pods that are part of a pod group or composite pod group with disruption mode All are grouped
// together into a single victim. Otherwise, they are treated as individual victims.
// In both cases, the priority of the victim is determined by the pod group or composite pod group priority.
func newDomainForWorkloadPreemption(logger klog.Logger, snapshot fwk.SharedLister, podGroupSnapshot fwk.PodGroupLister, compositePodGroupSnapshot fwk.CompositePodGroupLister, name string) (*domain, error) {
	nodes, err := snapshot.NodeInfos().List()
	if err != nil {
		return nil, err
	}

	allPossibleVictims, err := getCrossNodesVictims(logger, snapshot, podGroupSnapshot, compositePodGroupSnapshot, nodes)
	if err != nil {
		return nil, err
	}

	nodesMap := make(map[string]fwk.NodeInfo, len(nodes))
	for _, nodeInfo := range nodes {
		if nodeInfo != nil && nodeInfo.Node() != nil {
			nodesMap[nodeInfo.Node().Name] = nodeInfo
		}
	}

	return &domain{
		nodes:              nodesMap,
		allPossibleVictims: allPossibleVictims,
		name:               name,
	}, nil
}

// getCrossNodesVictims aggregates pods across the provided nodes into cluster-wide preemption candidates.
// When a pod belongs to a group hierarchy where any ancestor has disruption mode All, it groups all scheduled pods of that hierarchy
// across the cluster into a single victim so that preemption evaluates the total cluster-wide cost
// and blast radius of evicting the entire group.
func getCrossNodesVictims(logger klog.Logger, snapshot fwk.SharedLister, podGroupSnapshot fwk.PodGroupLister, compositePodGroupSnapshot fwk.CompositePodGroupLister, nodes []fwk.NodeInfo) ([]*DomainVictim, error) {
	existing := sets.New[fwk.EntityKey]()
	var victims []Victim
	for _, node := range nodes {
		for _, podInfo := range node.GetPods() {
			p := podInfo.GetPod()

			highestAllKey, hasAll := getHighestAllAncestor(p, podGroupSnapshot, compositePodGroupSnapshot)
			if !hasAll {
				itemKey := fwk.PodKey(p.Namespace, p.Name)
				if !existing.Has(itemKey) {
					existing.Insert(itemKey)
					victims = append(victims, NewPodVictim(podInfo, podGroupSnapshot, compositePodGroupSnapshot))
				}
				continue
			}

			if !existing.Has(highestAllKey) {
				existing.Insert(highestAllKey)
				victims = append(victims, searchCrossNodesVictimPods(logger, highestAllKey, podInfo, snapshot, podGroupSnapshot, compositePodGroupSnapshot))
			}
		}
	}

	return createDomainVictims(snapshot, victims)
}

// searchCrossNodesVictimPods searches and collects all scheduled pods belonging to the leaf pod groups
// of the hierarchy starting from pgKey, and aggregates them into a single Victim.
// If no scheduled pods are found in the leaf pod groups, it falls back to creating a PodVictim for the given podInfo.
func searchCrossNodesVictimPods(
	logger klog.Logger,
	pgKey fwk.EntityKey,
	podInfo fwk.PodInfo,
	snapshot fwk.SharedLister,
	podGroupSnapshot fwk.PodGroupLister,
	compositePodGroupSnapshot fwk.CompositePodGroupLister,
) Victim {
	var allPods []*v1.Pod
	if compositePodGroupSnapshot == nil {
		pgState, err := snapshot.PodGroupStates().Get(podInfo.GetPod().GetNamespace(), pgKey.Name)
		if err != nil {
			// Assuming this is guaranteed to succeed if feature is on and pods exist.
			// If it fails, we keep the local pods only.
			return NewPodVictim(podInfo, podGroupSnapshot, compositePodGroupSnapshot)
		}
		allPods = append(allPods, pgState.ScheduledPods()...)
	} else {
		for pgState, err := range helper.GetPodGroupStates(snapshot, pgKey) {
			if err != nil {
				logger.Error(err, "Failed to get pod group state during preemption victim collection", "rootKey", pgKey)
				continue
			}
			if pgState != nil {
				allPods = append(allPods, pgState.ScheduledPods()...)
			}
		}
	}

	if len(allPods) == 0 {
		// This should never happen since podInfo is sourced from a node in the snapshot and
		// should be present in ScheduledPods() of its leaf PodGroup. If leaf traversal or
		// PodGroupState snapshot lookups fail, treat the pod as an individual victim.
		return NewPodVictim(podInfo, podGroupSnapshot, compositePodGroupSnapshot)
	}

	podInfos := make([]fwk.PodInfo, len(allPods))
	for i, p := range allPods {
		// pods from ScheduledPods() already passed Filter/Reserve, and cannot error here.
		podInfos[i], _ = framework.NewPodInfo(p)
	}

	priority := getPodPriority(podInfo.GetPod(), podGroupSnapshot, compositePodGroupSnapshot)
	// It can only return an error for empty podInfos, which is guaranteed not to be empty here.
	victim, _ := NewVictim(podInfos, priority, pgKey.Type)
	return victim
}

// Victim represents an atomic entity that can be preempted (a victim).
// It abstracts individual Pods and PodGroup, ensuring that
// atomic entities are treated as a single unit during eviction.
type Victim interface {
	// Priority returns the priority of the preemption unit.
	// For a single Pod, this is the Pod's priority.
	// For a PodGroup, this is the priority of the PodGroup.
	Priority() int32

	// Pods returns the list of all Pods that belong to this preemption unit.
	// Evicting this unit implies evicting all Pods in this list.
	Pods() []fwk.PodInfo

	// EarliestStartTime returns the earliest start time of all Pods in this preemption unit.
	EarliestStartTime() *metav1.Time

	// IsGroup returns true if the preemption unit represents a PodGroup or a CompositePodGroup.
	// This function should be executed only when GenericWorkload feature is enabled.
	IsGroup() bool

	// Type returns the type of the preemption unit.
	Type() fwk.EntityKeyType
}

// victim is the concrete implementation of the Victim interface, bundling pods with their
// collective priority and earliest start time so individual pods and gang pod groups can be
// evaluated and sorted uniformly during preemption candidate selection.
type victim struct {
	pods              []fwk.PodInfo
	priority          int32
	earliestStartTime *metav1.Time
	keyType           fwk.EntityKeyType
}

var _ Victim = &victim{}

// Pods returns the list of all Pods that belong to this preemption unit.
// Evicting this unit implies evicting all Pods in this list.
func (v *victim) Pods() []fwk.PodInfo {
	return v.pods
}

// Priority returns the priority of the preemption unit.
// For a single Pod, this is the Pod's priority.
// For a PodGroup, this is the priority of the PodGroup.
func (v *victim) Priority() int32 {
	return v.priority
}

// EarliestStartTime returns the earliest start time of all Pods in this victim.
func (v *victim) EarliestStartTime() *metav1.Time {
	return v.earliestStartTime
}

// IsGroup returns true if the preemption unit represents a PodGroup or a CompositePodGroup.
// This function should be executed only when GenericWorkload feature is enabled.
func (v *victim) IsGroup() bool {
	return v.keyType == fwk.PodGroupKeyType || v.keyType == fwk.CompositePodGroupKeyType
}

// Type returns the type of the preemption unit.
func (v *victim) Type() fwk.EntityKeyType {
	return v.keyType
}

// NewPodVictim creates a new Victim representing a single Pod.
// It calculates the priority of the pod, taking into account its scheduling group if applicable.
// It ignores the error from NewVictim internally as it is guaranteed to succeed for a single valid pod.
// TODO: what we do below is not ideal:
//   - From the victim's importance POV, individual Pods should always have the PodKeyType.
//   - On the other hand, we need to store the information that the Pod victim belongs to the "single"
//     disruption mode PodGroup due to WAP-related metrics bookkeeping.
//   - Ideally, we should store another bit of information denoting the type of top-level group the victim
//     Pod belongs to.
//
// We should fix this on the occasion of adding support for CompositePodGroup WAP-related metrics.
func NewPodVictim(podInfo fwk.PodInfo, pgLister fwk.PodGroupLister, cpgLister fwk.CompositePodGroupLister) Victim {
	priority := getPodPriority(podInfo.GetPod(), pgLister, cpgLister)
	keyType := fwk.PodKeyType
	if podInfo.GetPod().Spec.SchedulingGroup != nil && pgLister != nil {
		keyType = fwk.PodGroupKeyType
	}
	vi, _ := NewVictim([]fwk.PodInfo{podInfo}, priority, keyType)
	return vi
}

// NewVictim creates a new Victim representing a set of Pods (or a PodGroup) that can be preempted together.
// It calculates the earliest start time among all provided Pods
func NewVictim(pods []fwk.PodInfo, priority int32, keyType fwk.EntityKeyType) (Victim, error) {
	if len(pods) == 0 {
		return nil, fmt.Errorf("no pods provided")
	}
	if keyType == "" {
		return nil, fmt.Errorf("keyType cannot be empty")
	}

	var earliest *metav1.Time
	for _, pInfo := range pods {
		t := schedutil.GetPodStartTime(pInfo.GetPod())
		if earliest == nil || (t != nil && t.Before(earliest)) {
			earliest = t
		}
	}

	return &victim{
		priority:          priority,
		pods:              pods,
		earliestStartTime: earliest,
		keyType:           keyType,
	}, nil
}

// DomainVictim extends Victim to include information about the nodes affected by its eviction.
// It represents a preemption unit within a specific scheduling domain and allows
// the preemption logic to understand the blast radius of the eviction across multiple nodes.
type DomainVictim struct {
	Victim
	// affectedNodes tracks every node hosting at least one pod belonging to this victim,
	// enabling the scheduler to assess node-specific impacts across the entire eviction blast radius.
	affectedNodes map[string]fwk.NodeInfo
}

// AffectedNodes returns a map of all nodes currently hosting Pods that belong to this victim.
func (dv *DomainVictim) AffectedNodes() map[string]fwk.NodeInfo {
	return dv.affectedNodes
}

// newDomainVictim creates a DomainVictim from the given pods and priority.
// It retrieves the NodeInfo for each pod from the snapshot and stores
// in the affectedNodes map to represent the nodes affected by evicting these pods.
func newDomainVictim(snapshot fwk.SharedLister, pods []fwk.PodInfo, priority int32, keyType fwk.EntityKeyType) (*DomainVictim, error) {
	nodeSnapshot := snapshot.NodeInfos()
	nodes := make(map[string]fwk.NodeInfo)
	for _, pInfo := range pods {
		nodeName := pInfo.GetPod().Spec.NodeName
		if _, ok := nodes[nodeName]; ok {
			continue
		}

		nodeInfo, err := nodeSnapshot.Get(nodeName)
		if err != nil {
			return nil, fmt.Errorf("failed to get node info for node %q from snapshot: %w", nodeName, err)
		}
		nodes[nodeName] = nodeInfo
	}

	victim, err := NewVictim(pods, priority, keyType)
	if err != nil {
		return nil, err
	}

	return &DomainVictim{
		Victim:        victim,
		affectedNodes: nodes,
	}, nil
}

// Candidate represents a nominated node on which the preemptor can be scheduled,
// along with the list of victims that should be evicted for the preemptor to fit the node.
type Candidate interface {
	// Victims wraps a list of to-be-preempted Pods and the number of PDB violation.
	Victims() *extenderv1.Victims
	// Name returns the target domain(for pod group)/node name where the preemptor gets nominated to run.
	Name() string
	// NumPodGroupDisruptions returns the number of preemption units that affect pod groups.
	// A single preemption unit can be all pods in a pod group (for DisruptionMode=all) or a single pod (for DisruptionMode=single).
	NumPodGroupDisruptions() int
}

type candidate struct {
	victims                *extenderv1.Victims
	numPodGroupDisruptions int
	name                   string
}

// Victims returns s.victims.
func (s *candidate) Victims() *extenderv1.Victims {
	return s.victims
}

// Name returns s.name.
func (s *candidate) Name() string {
	return s.name
}

// NumPodGroupDisruptions returns s.numPodGroupDisruptions.
func (s *candidate) NumPodGroupDisruptions() int {
	return s.numPodGroupDisruptions
}

type candidateList struct {
	idx   int32
	items []Candidate
}

// newCandidateList creates a new candidate list with the given capacity.
func newCandidateList(capacity int32) *candidateList {
	return &candidateList{idx: -1, items: make([]Candidate, capacity)}
}

// add adds a new candidate to the internal array atomically.
// Note: in case the list has reached its capacity, the candidate is disregarded
// and not added to the internal array.
func (cl *candidateList) add(c *candidate) {
	if idx := atomic.AddInt32(&cl.idx, 1); idx < int32(len(cl.items)) {
		cl.items[idx] = c
	}
}

// size returns the number of candidate stored. Note that some add() operations
// might still be executing when this is called, so care must be taken to
// ensure that all add() operations complete before accessing the elements of
// the list.
func (cl *candidateList) size() int32 {
	return min(atomic.LoadInt32(&cl.idx)+1, int32(len(cl.items)))
}

// get returns the internal candidate array. This function is NOT atomic and
// assumes that all add() operations have been completed.
func (cl *candidateList) get() []Candidate {
	return cl.items[:cl.size()]
}

// ViolatingVictim represents a victim whose preemption would violate one or more PodDisruptionBudgets.
// It tracks the victim itself and the number of pods within it that would cause PDB violations on eviction.
type ViolatingVictim[T Victim] struct {
	Victim       T
	ViolateCount int
}
