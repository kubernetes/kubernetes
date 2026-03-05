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
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// Preemptor abstracts the entity that initiates preemption.
// It acts as a unified interface for single Pod or a PodGroup cases,
// allowing the preemption logic to treat them polymorphically.
type Preemptor interface {
	// Priority returns the scheduling priority of the preemptor.
	// This value is used to identify potential victims (which must have lower priority).
	Priority() int32

	// IsPodGroup returns true if the preemptor represents a PodGroup
	// that must be scheduled atomically, or false if it represents a single Pod.
	IsPodGroup() bool

	// Members returns the list of Pods that belong to this preemptor.
	// For a single Pod preemptor, this returns a slice containing only that Pod.
	// For a PodGroup, this returns all Pods in the group.
	Members() []*v1.Pod

	// CycleStates returns the CycleState for each member Pod.
	// The slice index corresponds to the index in Members().
	CycleStates() []fwk.CycleState

	// Snapshot creates a copy of the Preemptor with deep-copied CycleStates.
	// This is used for dry-run simulations where we need to mutate the state
	// without affecting the original preemption context.
	Snapshot() Preemptor
}

type preemptor struct {
	priority         int32
	pods             []*v1.Pod
	preemptionPolicy *v1.PreemptionPolicy
	isPodGroup       bool
	states           []fwk.CycleState
}

var _ Preemptor = &preemptor{}

func NewPodPreemptor(p *v1.Pod, state fwk.CycleState) Preemptor {
	return &preemptor{
		priority:         corev1helpers.PodPriority(p),
		pods:             []*v1.Pod{p},
		preemptionPolicy: p.Spec.PreemptionPolicy,
		isPodGroup:       false,
		states:           []fwk.CycleState{state},
	}
}

func (p *preemptor) Priority() int32 {
	return p.priority
}

func (p *preemptor) IsPodGroup() bool {
	return p.isPodGroup
}

func (p *preemptor) Members() []*v1.Pod {
	return p.pods
}

func (p *preemptor) CycleStates() []fwk.CycleState {
	return p.states
}

func (p *preemptor) Snapshot() Preemptor {
	newStates := make([]fwk.CycleState, len(p.states))
	for i, state := range p.states {
		newStates[i] = state.Clone()
	}

	return &preemptor{
		priority:         p.priority,
		pods:             p.pods,
		preemptionPolicy: p.preemptionPolicy,
		isPodGroup:       p.isPodGroup,
		states:           newStates,
	}
}

// Domain represents the boundary or scope within which the preemption logic is evaluated.
// It abstracts the scheduling domain, which can range from a single Node (for standard Pod preemption)
// to a group of Nodes or the entire Cluster (for Workload preemption).
type Domain interface {
	// Nodes returns a list of NodeInfo objects that belong to this domain.
	// The preemption logic uses this to check feasibility and resource availability
	// within the specific scope.
	Nodes() []fwk.NodeInfo

	// GetAllPossibleVictims returns all potential victims (PreemptionUnits) running within this domain. (single Pod or a PodGroup)
	GetAllPossibleVictims() []PreemptionUnit

	// GetName returns a unique identifier or name for the domain.
	// This is primarily used for logging and debugging purposes.
	GetName() string
}

type domain struct {
	nodes              []fwk.NodeInfo
	name               string
	allPossibleVictims []PreemptionUnit
}

var _ Domain = &domain{}

func NewDomainForPodByPodPreemption(node fwk.NodeInfo, name string) Domain {
	allPossibleVictims := make([]PreemptionUnit, 0, len(node.GetPods()))
	for _, p := range node.GetPods() {
		allPossibleVictims = append(allPossibleVictims, NewPreemptionUnit([]fwk.PodInfo{p}, corev1helpers.PodPriority(p.GetPod()), []fwk.NodeInfo{node}))
	}

	return &domain{
		nodes:              []fwk.NodeInfo{node},
		allPossibleVictims: allPossibleVictims,
		name:               name,
	}
}

func (d *domain) Nodes() []fwk.NodeInfo {
	return d.nodes
}

func (d *domain) GetAllPossibleVictims() []PreemptionUnit {
	return d.allPossibleVictims
}

func (d *domain) GetName() string {
	return d.name
}

// PreemptionUnit represents an atomic entity that can be preempted (a victim).
// It abstracts individual Pods and PodGroup, ensuring that
// atomic entities are treated as a single unit during eviction.
type PreemptionUnit interface {
	// Priority returns the priority of the preemption unit.
	// For a single Pod, this is the Pod's priority.
	// For a PodGroup, this is typically the priority of the PodGroup (or its members).
	Priority() int32

	// AffectedNodes returns a map of Node names to NodeInfo for all nodes
	// where members of this preemption unit are currently running.
	// This allows the preemption logic to identify the blast radius of evicting this unit.
	AffectedNodes() map[string]fwk.NodeInfo

	// Pods returns the list of all Pods that belong to this preemption unit.
	// Evicting this unit implies evicting all Pods in this list.
	Pods() []fwk.PodInfo

	// IsPodGroup returns true if the preemption unit represents a PodGroup
	// that must be scheduled atomically, or false if it represents a single Pod.
	IsPodGroup() bool
}

type preemptionUnit struct {
	pods          []fwk.PodInfo
	priority      int32
	affectedNodes map[string]fwk.NodeInfo
}

var _ PreemptionUnit = &preemptionUnit{}

func NewPreemptionUnit(pods []fwk.PodInfo, priority int32, nodeInfos []fwk.NodeInfo) PreemptionUnit {
	nodes := make(map[string]fwk.NodeInfo)
	for _, node := range nodeInfos {
		nodes[node.Node().Name] = node
	}

	return &preemptionUnit{
		affectedNodes: nodes,
		priority:      priority,
		pods:          pods,
	}
}

func (pu *preemptionUnit) Pods() []fwk.PodInfo {
	return pu.pods
}

func (pu *preemptionUnit) Priority() int32 {
	return pu.priority
}

func (pu *preemptionUnit) AffectedNodes() map[string]fwk.NodeInfo {
	return pu.affectedNodes
}

func (pu *preemptionUnit) IsPodGroup() bool {
	return len(pu.pods) > 0 && (len(pu.pods) > 1 || pu.pods[0].GetPod().Spec.WorkloadRef != nil)
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
