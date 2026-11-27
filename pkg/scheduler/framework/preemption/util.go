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
	"cmp"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	policylisters "k8s.io/client-go/listers/policy/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// PodTerminatingByPreemption returns true if the pod is in the termination state caused by scheduler preemption.
func PodTerminatingByPreemption(p *v1.Pod) bool {
	if p.DeletionTimestamp == nil {
		return false
	}

	for _, condition := range p.Status.Conditions {
		if condition.Type == v1.DisruptionTarget {
			return condition.Status == v1.ConditionTrue && condition.Reason == v1.PodReasonPreemptionByScheduler
		}
	}
	return false
}

// NominatedNodeName returns the node that pod is nominated to. When fh
// implements framework.NominatedNodeNameGetter (which returns the in-memory
// nomination from PodNominator when DRAPreemption is enabled), it delegates to
// fh. Otherwise, it returns pod.Status.NominatedNodeName.
func NominatedNodeName(fh fwk.Handle, pod *v1.Pod) string {
	if nominator, ok := fh.(framework.NominatedNodeNameGetter); ok {
		return nominator.NominatedNodeName(pod)
	}
	return pod.Status.NominatedNodeName
}

// MoreImportantVictim decides which of two preemption units is considered more critical.
// It applies the following rules in order:
//
//  1. Priority:
//     Higher priority units are more important.
//
//  2. Workload Type:
//     CompositePodGroups are considered more important than PodGroups, and
//     PodGroups are more important than individual Pods to preserve group integrity.
//
//  3. Runtime / Start Time (for individual Pods):
//     For two individual Pods, the one that started earlier (longer runtime)
//     is more important. This honors "first-come, first-served".
//
//  4. Group Size (for PodGroups and CompositePodGroups):
//     If both units are of the same type, the one with more members (larger size) is considered
//     more important. This avoids the high cost of rescheduling massive jobs.
//
//  5. Start Time (Tie-breaker for group types):
//     If group types and sizes are equal, the group that started earlier (has the oldest pod)
//     is more important.
func MoreImportantVictim(vi1, vi2 Victim) bool {
	if c := compareVictimStructuralImportance(vi1, vi2); c != 0 {
		return c > 0
	}

	t1, t2 := vi1.EarliestStartTime(), vi2.EarliestStartTime()
	if t1 != nil && t2 != nil && !t1.Equal(t2) {
		return t1.Before(t2)
	}

	return podIdentityLess(vi1, vi2)
}

// podIdentityLess provides a deterministic, stable ordering of victims based on
// the representative pod's UID. The first pod in a victim's list is stable once
// the victim is constructed, so a direct comparison is sufficient.
func podIdentityLess(vi1, vi2 Victim) bool {
	return vi1.Pods()[0].GetPod().UID < vi2.Pods()[0].GetPod().UID
}

func victimRank(vi Victim) int {
	switch vi.Type() {
	case fwk.CompositePodGroupKeyType:
		return 3
	case fwk.PodGroupKeyType:
		return 2
	default:
		return 1
	}
}

// compareVictimStructuralImportance compares two victims on structural importance:
// Priority, Workload Type Rank (CompositePodGroup > PodGroup > Pod), and Group Size.
// Returns > 0 if vi1 is more important, < 0 if vi2 is more important, and 0 if equal.
func compareVictimStructuralImportance(vi1, vi2 Victim) int {
	if c := cmp.Compare(vi1.Priority(), vi2.Priority()); c != 0 {
		return c
	}
	if c := cmp.Compare(victimRank(vi1), victimRank(vi2)); c != 0 {
		return c
	}
	return cmp.Compare(len(vi1.Pods()), len(vi2.Pods()))
}

func podMatchesPDB(pod *v1.Pod, pdb *policy.PodDisruptionBudget) bool {
	if pdb.Namespace != pod.Namespace {
		return false
	}
	selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
	if err != nil || !selector.Matches(labels.Set(pod.Labels)) {
		// Either the PDB had invalid selector or it did not match the pod label.
		return false
	}
	// Existing in DisruptedPods means it has been processed in API server,
	// we don't treat it as a violating case.
	if _, exist := pdb.Status.DisruptedPods[pod.Name]; exist {
		return false
	}
	return true
}

// FilterVictimsWithPDBViolation groups "victims" into "violatingVictims"
// and "nonViolatingVictims" based on whether their PDBs will be violated if they are
// preempted.
func FilterVictimsWithPDBViolation[T Victim](victims []T, pdbs []*policy.PodDisruptionBudget) (violatingVictims []ViolatingVictim[T], nonViolatingVictims []T) {
	// pdbsAllowed holds the disruptions each PDB still allows; it decreases as victims are evaluated in order.
	pdbsAllowed := make([]int32, len(pdbs))
	podIsViolating := func(pod *v1.Pod) bool {
		violating := false
		for i, pdb := range pdbs {
			if !podMatchesPDB(pod, pdb) {
				continue
			}
			// Only decrement the matched pdb when it's not in its <DisruptedPods>;
			// otherwise we may over-decrement the budget number.
			pdbsAllowed[i]--
			// Check the pod against every PDB even after one of them is violated, because
			// subsequent pods protected by the same PDBs are evaluated against the leftover budget.
			if pdbsAllowed[i] < 0 {
				violating = true
			}
		}
		return violating
	}

	for i, pdb := range pdbs {
		pdbsAllowed[i] = pdb.Status.DisruptionsAllowed
	}

	// A Victim is atomic: if any single pod inside it would violate a PDB on
	// eviction, the entire Victim is classified as violating. This is required
	// for PodGroup victims, where partial preemption is not allowed — we cannot
	// "reprieve" only the pods that violate a PDB and keep the rest evicted.
	for _, victim := range victims {
		violatingPods := 0

		for _, pi := range victim.Pods() {
			if podIsViolating(pi.GetPod()) {
				violatingPods++
			}
		}
		if violatingPods > 0 {
			violatingVictims = append(violatingVictims, ViolatingVictim[T]{
				Victim:       victim,
				ViolateCount: violatingPods,
			})
		} else {
			nonViolatingVictims = append(nonViolatingVictims, victim)
		}
	}

	return violatingVictims, nonViolatingVictims
}

// initPDBDisruptionsAllowed sets pdbDisruptionsAllowed for every PDB protecting any pod of the given victim.
func initPDBDisruptionsAllowed(pdbDisruptionsAllowed map[*policy.PodDisruptionBudget]int32, vw *victimWithPDBViolations) {
	for _, podPDBs := range vw.pdbsPerPod {
		for _, pdb := range podPDBs {
			pdbDisruptionsAllowed[pdb] = pdb.Status.DisruptionsAllowed
		}
	}
}

// countVictimPDBViolations returns how many pods of the given victim would violate a PDB if it were
// evicted. Each pod consumes one disruption in remainingDisruptionsAllowed from every PDB that protects
// it. Once a PDB has no disruptions left, the further pods it protects are violations (a pod counts once
// even if it violates several PDBs).
func countVictimPDBViolations(vw *victimWithPDBViolations, remainingDisruptionsAllowed map[*policy.PodDisruptionBudget]int32) int {
	violatingPods := 0
	for _, podPDBs := range vw.pdbsPerPod {
		podViolates := false
		for _, pdb := range podPDBs {
			remainingDisruptionsAllowed[pdb]--
			if remainingDisruptionsAllowed[pdb] < 0 {
				podViolates = true
			}
		}
		if podViolates {
			violatingPods++
		}
	}
	return violatingPods
}

// getPDBsPerPod returns matching PDBs for each pod in victim.Pods() that is protected by at least one PDB (ignoring already disrupted pods).
func getPDBsPerPod(victim Victim, pdbs []*policy.PodDisruptionBudget) [][]*policy.PodDisruptionBudget {
	if len(pdbs) == 0 {
		return nil
	}
	var pdbsPerPod [][]*policy.PodDisruptionBudget
	for _, pi := range victim.Pods() {
		pod := pi.GetPod()
		var podPDBs []*policy.PodDisruptionBudget
		for _, pdb := range pdbs {
			if podMatchesPDB(pod, pdb) {
				podPDBs = append(podPDBs, pdb)
			}
		}
		if len(podPDBs) > 0 {
			pdbsPerPod = append(pdbsPerPod, podPDBs)
		}
	}
	return pdbsPerPod
}

// getPodDisruptionBudgets returns all pod disruption budgets from the provided lister.
func getPodDisruptionBudgets(pdbLister policylisters.PodDisruptionBudgetLister) ([]*policy.PodDisruptionBudget, error) {
	return pdbLister.List(labels.Everything())
}
