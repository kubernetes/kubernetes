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
	"maps"
	"slices"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha2"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"

	"k8s.io/kubernetes/pkg/scheduler/util"
)

type podGroupPreemptor struct {
	priority         int32
	pods             []*v1.Pod
	podGroup         *schedulingapi.PodGroup
	preemptionPolicy v1.PreemptionPolicy
}

func newPodGroupPreemptor(pg *schedulingapi.PodGroup, pods []*v1.Pod) *podGroupPreemptor {
	preemptionPolicy := v1.PreemptLowerPriority
	for _, pod := range pods {
		if p := pod.Spec.PreemptionPolicy; p != nil && *p == v1.PreemptNever {
			preemptionPolicy = *p
		}
	}
	return &podGroupPreemptor{
		priority:         util.PodGroupPriority(pg),
		pods:             pods,
		podGroup:         pg,
		preemptionPolicy: preemptionPolicy,
	}
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
func (p *podGroupPreemptor) PreemptionPolicy() v1.PreemptionPolicy {
	return p.preemptionPolicy
}

// domain represents the boundary or scope within which the preemption logic is evaluated.
type domain struct {
	nodes              []fwk.NodeInfo
	name               string
	allPossibleVictims []*victim
}

// isPodGroupPreemptiblePod checks if a pod is a part of a pod group that should
// be treated as a single unit for preemption purposes.
// If the pod is a part of such a pod group, it returns the pod group and true.
// In all other cases, it returns nil and false.
func isPodGroupPreemptiblePod(p *v1.Pod, pgLister schedulinglisters.PodGroupLister) (*schedulingapi.PodGroup, bool) {
	if p.Spec.SchedulingGroup == nil {
		return nil, false
	}
	pgName := p.Spec.SchedulingGroup.PodGroupName
	pg, err := pgLister.PodGroups(p.Namespace).Get(*pgName)
	if err != nil {
		return nil, false
	}
	if mode := pg.Spec.DisruptionMode; mode == nil || *mode != schedulingapi.DisruptionModePodGroup {
		return nil, false
	}
	return pg, true
}

// newDomainForWorkloadPreemption creates a new domain for workload preemption.
// The domain is the whole cluster and it contains victims that are computed based
// on the pods and their scheduling groups.
// Pods that are part of a pod group with the PodGroup disruption mode are grouped
// together into a single victim. Otherwise, they are treated as individual victims.
func newDomainForWorkloadPreemption(nodes []fwk.NodeInfo, pgLister schedulinglisters.PodGroupLister, name string) *domain {
	victimMap := map[types.UID]*victim{}
	for _, node := range nodes {
		for _, p := range node.GetPods() {
			// TODO: Calling the lister here is not ideal given we do this
			// for every pod in the cluster. Instead, we should be getting
			// this information from the snapshot.
			pg, ok := isPodGroupPreemptiblePod(p.GetPod(), pgLister)
			if !ok {
				victimMap[p.GetPod().UID] = newVictim([]fwk.PodInfo{p}, corev1helpers.PodPriority(p.GetPod()), []fwk.NodeInfo{node})
				continue
			}
			victim, ok := victimMap[pg.UID]
			if ok {
				victim.pods = append(victim.pods, p)
				victim.affectedNodes[node.Node().Name] = node
				continue
			}
			victimMap[pg.UID] = newVictim([]fwk.PodInfo{p}, util.PodGroupPriority(pg), []fwk.NodeInfo{node})
		}
	}

	allPossibleVictims := slices.Collect(maps.Values(victimMap))
	return &domain{
		nodes:              nodes,
		allPossibleVictims: allPossibleVictims,
		name:               name,
	}
}

// Nodes returns a list of NodeInfo objects that belong to this domain.
func (d *domain) Nodes() []fwk.NodeInfo {
	return d.nodes
}

// // GetAllPossibleVictims returns all potential victims running within this domain.
func (d *domain) GetAllPossibleVictims() []*victim {
	return d.allPossibleVictims
}

// GetName returns a unique identifier for the domain.
// This is primarily used for logging and debugging purposes.
func (d *domain) GetName() string {
	return d.name
}

// victim represents an atomic entity that can be preempted (a victim).
// It abstracts individual Pods and PodGroup, ensuring that
// atomic entities are treated as a single unit during eviction.
type victim struct {
	pods              []fwk.PodInfo
	priority          int32
	affectedNodes     map[string]fwk.NodeInfo
	earliestStartTime *metav1.Time
}

// newVictim creates a new Victim representing a set of Pods (or a PodGroup) that can be preempted together.
// It calculates the earliest start time among all provided Pods and identifies all nodes
// affected by the potential eviction of these Pods.
func newVictim(pods []fwk.PodInfo, priority int32, nodeInfos []fwk.NodeInfo) *victim {
	nodes := make(map[string]fwk.NodeInfo)
	for _, node := range nodeInfos {
		nodes[node.Node().Name] = node
	}

	var earliest *metav1.Time
	for _, pInfo := range pods {
		t := util.GetPodStartTime(pInfo.GetPod())
		if earliest == nil || (t != nil && t.Before(earliest)) {
			earliest = t
		}
	}

	return &victim{
		affectedNodes:     nodes,
		priority:          priority,
		pods:              pods,
		earliestStartTime: earliest,
	}
}

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

// AffectedNodes returns a map of Node names to NodeInfo for all nodes
// where members of this preemption unit are currently running.
// This allows the preemption logic to identify the blast radius of evicting this unit.
func (v *victim) AffectedNodes() map[string]fwk.NodeInfo {
	return v.affectedNodes
}

// EarliestStartTime returns the earliest start time of all Pods in this victim.
func (v *victim) EarliestStartTime() *metav1.Time {
	return v.earliestStartTime
}

// IsPodGroup returns true if the preemption unit represents a PodGroup.
func (v *victim) IsPodGroup() bool {
	return v.pods[0].GetPod().Spec.SchedulingGroup != nil
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
