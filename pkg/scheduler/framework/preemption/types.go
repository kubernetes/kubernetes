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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// Victim represents an atomic entity that can be preempted (a victim).
// It abstracts individual Pods and PodGroup, ensuring that
// atomic entities are treated as a single unit during eviction.
type Victim interface {
	// Priority returns the priority of the preemption unit, which is the priority of its Pods.
	// For a PodGroup or CompositePodGroup this is also the group's priority: the scheduling cycle
	// guarantees that every Pod in a hierarchy shares the priority of the hierarchy's root.
	Priority() int32

	// Pods returns the list of all Pods that belong to this preemption unit.
	// Evicting this unit implies evicting all Pods in this list.
	Pods() []fwk.PodInfo

	// EarliestStartTime returns the earliest start time of all Pods in this preemption unit.
	EarliestStartTime() *metav1.Time

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

// Priority returns the priority of the preemption unit, which is the priority of its Pods.
// For a PodGroup or CompositePodGroup this is also the group's priority: the scheduling cycle
// guarantees that every Pod in a hierarchy shares the priority of the hierarchy's root.
func (v *victim) Priority() int32 {
	return v.priority
}

// EarliestStartTime returns the earliest start time of all Pods in this victim.
func (v *victim) EarliestStartTime() *metav1.Time {
	return v.earliestStartTime
}

// Type returns the type of the preemption unit.
func (v *victim) Type() fwk.EntityKeyType {
	return v.keyType
}

// NewPodVictim creates a new Victim representing a single Pod.
// It ignores the error from NewVictim internally as it is guaranteed to succeed for a single valid pod.
// The victim is always of PodKeyType, even if the pod belongs to a PodGroup or CompositePodGroup with
// disruption mode Single: it is evicted alone, so it ranks and is reported in metrics as an individual pod.
func NewPodVictim(podInfo fwk.PodInfo) Victim {
	vi, _ := NewVictim([]fwk.PodInfo{podInfo}, corev1helpers.PodPriority(podInfo.GetPod()), fwk.PodKeyType)
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

// candidate represents a nominated node on which the preemptor can be scheduled,
// along with the list of victims that should be evicted for the preemptor to fit the node.
type candidate struct {
	// victims wraps a list of to-be-preempted Pods and the number of PDB violation.
	victims *extenderv1.Victims
	// name returns the target domain(for pod group)/node name where the preemptor gets nominated to run.
	name string
	// disruptions is the number of preemption units per victim type. Used only for metrics.
	disruptions map[fwk.EntityKeyType]int
	// nodes returns a list of nodes where the preemptor pods get nominated to run.
	// For a single pod preemption it's equal Name but for pod group preemption
	// it contains a set of nodes for all preemptor pods.
	nodes []string
}

var _ fwk.PreemptionCandidate = &candidate{}

func newCandidate(victims *extenderv1.Victims, name string, disruptions map[fwk.EntityKeyType]int, nodes ...string) *candidate {
	return &candidate{
		victims:     victims,
		name:        name,
		disruptions: disruptions,
		nodes:       nodes,
	}
}

// Victims returns s.victims.
func (s *candidate) Victims() *extenderv1.Victims {
	return s.victims
}

// Name returns s.name.
func (s *candidate) Name() string {
	return s.name
}

// NumDisruptionsByVictimType returns s.disruptions.
func (s *candidate) NumDisruptionsByVictimType() map[fwk.EntityKeyType]int {
	return s.disruptions
}

// Nodes returns a list of nodes where the preemptor pods get nominated to run.
// For a single pod preemption it's equal Name but for pod group preemption
// it contains a set of nodes for all preemptor pods.
func (s *candidate) Nodes() []string {
	return s.nodes
}

type candidateList struct {
	idx   int32
	items []fwk.PreemptionCandidate
}

// newCandidateList creates a new candidate list with the given capacity.
func newCandidateList(capacity int32) *candidateList {
	return &candidateList{idx: -1, items: make([]fwk.PreemptionCandidate, capacity)}
}

// add adds a new candidate to the internal array atomically.
// Note: in case the list has reached its capacity, the candidate is disregarded
// and not added to the internal array.
func (cl *candidateList) add(c fwk.PreemptionCandidate) {
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
func (cl *candidateList) get() []fwk.PreemptionCandidate {
	return cl.items[:cl.size()]
}

// ViolatingVictim represents a victim whose preemption would violate one or more PodDisruptionBudgets.
// It tracks the victim itself and the number of pods within it that would cause PDB violations on eviction.
type ViolatingVictim[T Victim] struct {
	Victim       T
	ViolateCount int
}
