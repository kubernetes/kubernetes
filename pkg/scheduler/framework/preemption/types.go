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
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

type podGroupPreemptor struct {
	priority         int32
	pods             []*v1.Pod
	podGroup         *schedulingapi.PodGroup
	preemptionPolicy schedulingapi.PreemptionPolicy
}

func newPodGroupPreemptor(pg *schedulingapi.PodGroup, pods []*v1.Pod, enablePodGroupPreemptionPolicy bool) *podGroupPreemptor {
	return &podGroupPreemptor{
		priority:         util.PodGroupPriority(pg),
		pods:             pods,
		podGroup:         pg,
		preemptionPolicy: resolvePreemptionPolicy(pg, pods, enablePodGroupPreemptionPolicy),
	}
}

func resolvePreemptionPolicy(pg *schedulingapi.PodGroup, pods []*v1.Pod, enablePodGroupPreemptionPolicy bool) schedulingapi.PreemptionPolicy {
	if enablePodGroupPreemptionPolicy {
		// If the PodGroup was created with PodGroupPreemptionPolicy feature disabled, the PreemptionPolicy field will be nil.
		// In this case the default policy value should be returned.
		if pg.Spec.PreemptionPolicy != nil {
			return *pg.Spec.PreemptionPolicy
		}
	} else {
		for _, pod := range pods {
			if p := pod.Spec.PreemptionPolicy; p != nil && *p == v1.PreemptNever {
				return schedulingapi.PreemptNever
			}
		}
	}
	return schedulingapi.PreemptLowerPriority
}

// Priority returns the scheduling priority of the preemptor.
// This value is used to identify potential victims (which must have lower priority).
func (p *podGroupPreemptor) Priority() int32 {
	return p.priority
}

// Members returns the list of Pods that belong to this preemptor.
func (p *podGroupPreemptor) Members() []*v1.Pod {
	return p.pods
}

// PodGroup returns a pod group connected with this preemptor.
func (p *podGroupPreemptor) PodGroup() *schedulingapi.PodGroup {
	return p.podGroup
}

// PreemptionPolicy returns a preemption policy of this preemptor.
func (p *podGroupPreemptor) PreemptionPolicy() schedulingapi.PreemptionPolicy {
	return p.preemptionPolicy
}

// domain represents the boundary or scope within which the preemption logic is evaluated.
// It abstracts the scheduling domain, which can range from a single Node (for standard Pod preemption)
// to a group of Nodes or the entire Cluster (for PodGroup preemption).
type domain struct {
	nodes              []fwk.NodeInfo
	name               string
	allPossibleVictims []*DomainVictim
}

// Nodes returns a list of NodeInfo objects that belong to this domain.
// The preemption logic uses this to check feasibility and resource availability
// within the specific scope.
func (d *domain) Nodes() []fwk.NodeInfo {
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

// getPodGroup checks if a pod specifies a scheduling group and returns the corresponding PodGroup object if found.
// This function should be executed only when GenericWorkload feature is enabled.
func getPodGroup(p *v1.Pod, podGroupSnapshot fwk.PodGroupLister) *schedulingapi.PodGroup {
	if podGroupSnapshot == nil || p.Spec.SchedulingGroup == nil {
		return nil
	}
	pgName := p.Spec.SchedulingGroup.PodGroupName
	pg, err := podGroupSnapshot.Get(p.Namespace, *pgName)
	if err != nil {
		return nil
	}
	return pg
}

// isDisruptionModeAll checks if the PodGroup disruption mode is set to All.
func isDisruptionModeAll(pg *schedulingapi.PodGroup) bool {
	return pg != nil && pg.Spec.DisruptionMode != nil && pg.Spec.DisruptionMode.All != nil
}

// createDomainVictims converts deduplicated Victim entries into DomainVictim objects
// by enriching them with node blast-radius metadata from the snapshot, allowing the preemption
// evaluator to assess the cluster-wide impact of evicting each candidate.
func createDomainVictims(snapshot fwk.SharedLister, victims []Victim) ([]*DomainVictim, error) {
	var allPossibleVictims []*DomainVictim
	for _, vi := range victims {
		v, err := newDomainVictim(snapshot, vi.Pods(), vi.Priority())
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
// Pods that are part of a pod group with the PodGroup disruption mode are grouped
// together into a single victim. Otherwise, they are treated as individual victims.
// In both cases, the priority of the victim is determined by the PodGroup priority
// if the pod belongs to a PodGroup.
func newDomainForWorkloadPreemption(snapshot fwk.SharedLister, podGroupSnapshot fwk.PodGroupLister, name string) (*domain, error) {
	nodes, err := snapshot.NodeInfos().List()
	if err != nil {
		return nil, err
	}

	allPossibleVictims, err := getCrossNodesVictims(snapshot, podGroupSnapshot, nodes)
	if err != nil {
		return nil, err
	}

	return &domain{
		nodes:              nodes,
		allPossibleVictims: allPossibleVictims,
		name:               name,
	}, nil
}

// getCrossNodesVictims aggregates pods across the provided nodes into cluster-wide preemption candidates.
// When a pod belongs to a PodGroup with disruption mode All, it groups all scheduled pods of that gang
// across the cluster into a single victim so that preemption evaluates the total cluster-wide cost
// and blast radius of evicting the entire gang.
func getCrossNodesVictims(snapshot fwk.SharedLister, podGroupSnapshot fwk.PodGroupLister, nodes []fwk.NodeInfo) ([]*DomainVictim, error) {
	searchCrossNodesVictimPods := func(pg *schedulingapi.PodGroup, podInfo fwk.PodInfo) Victim {
		pgState, err := snapshot.PodGroupStates().Get(podInfo.GetPod().GetNamespace(), pg.Name)
		if err != nil {
			// Assuming this is guaranteed to succeed if feature is on and pods exist.
			// If it fails, we keep the local pods only.
			return NewPodVictim(podInfo, podGroupSnapshot)
		}

		pods := pgState.ScheduledPods()
		podInfos := make([]fwk.PodInfo, len(pods))
		for i, p := range pods {
			// pods from ScheduledPods() already passed Filter/Reserve, and cannot error here.
			podInfos[i], _ = framework.NewPodInfo(p)
		}

		// It can only return an error for empty podInfos, which is guaranteed not to be empty here.
		victim, _ := NewVictim(podInfos, util.PodGroupPriority(pg))

		return victim
	}

	existing := sets.New[types.UID]()
	var victims []Victim
	for _, node := range nodes {
		for _, podInfo := range node.GetPods() {
			p := podInfo.GetPod()

			pg := getPodGroup(p, podGroupSnapshot)
			if pg == nil || !isDisruptionModeAll(pg) {
				if !existing.Has(p.UID) {
					existing.Insert(p.UID)
					victims = append(victims, NewPodVictim(podInfo, podGroupSnapshot))
				}
				continue
			}

			if !existing.Has(pg.UID) {
				existing.Insert(pg.UID)
				victims = append(victims, searchCrossNodesVictimPods(pg, podInfo))
			}
		}
	}

	return createDomainVictims(snapshot, victims)
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

	// IsPodGroup returns true if the preemption unit represents a PodGroup.
	// This function should be executed only when GenericWorkload feature is enabled.
	IsPodGroup() bool
}

// victim is the concrete implementation of the Victim interface, bundling pods with their
// collective priority and earliest start time so individual pods and gang pod groups can be
// evaluated and sorted uniformly during preemption candidate selection.
type victim struct {
	pods              []fwk.PodInfo
	priority          int32
	earliestStartTime *metav1.Time
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

// IsPodGroup returns true if the preemption unit represents a PodGroup.
// This function should be executed only when GenericWorkload feature is enabled.
func (v *victim) IsPodGroup() bool {
	return len(v.pods) > 0 && v.pods[0].GetPod().Spec.SchedulingGroup != nil
}

// NewPodVictim creates a new Victim representing a single Pod.
// It calculates the priority of the pod, taking into account its scheduling group if applicable.
// It ignores the error from NewVictim internally as it is guaranteed to succeed for a single valid pod.
func NewPodVictim(podInfo fwk.PodInfo, pgLister fwk.PodGroupLister) Victim {
	priority := GetPodPriority(podInfo.GetPod(), pgLister)
	vi, _ := NewVictim([]fwk.PodInfo{podInfo}, priority)
	return vi
}

// NewVictim creates a new Victim representing a set of Pods (or a PodGroup) that can be preempted together.
// It calculates the earliest start time among all provided Pods
func NewVictim(pods []fwk.PodInfo, priority int32) (Victim, error) {
	if len(pods) == 0 {
		return nil, fmt.Errorf("no pods provided")
	}

	var earliest *metav1.Time
	for _, pInfo := range pods {
		t := util.GetPodStartTime(pInfo.GetPod())
		if earliest == nil || (t != nil && t.Before(earliest)) {
			earliest = t
		}
	}

	return &victim{
		priority:          priority,
		pods:              pods,
		earliestStartTime: earliest,
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
func newDomainVictim(snapshot fwk.SharedLister, pods []fwk.PodInfo, priority int32) (*DomainVictim, error) {
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

	victim, err := NewVictim(pods, priority)
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
}

type candidate struct {
	victims *extenderv1.Victims
	name    string
}

// Victims returns s.victims.
func (s *candidate) Victims() *extenderv1.Victims {
	return s.victims
}

// Name returns s.name.
func (s *candidate) Name() string {
	return s.name
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
