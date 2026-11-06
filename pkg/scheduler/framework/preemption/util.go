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

// CompareVictimSets compares the preemption cost of two sets of victims (each sorted
// descending by MoreImportantVictim).
// Returns < 0 if set1 is less costly (better to preempt) than set2,
// > 0 if set1 is more costly (worse to preempt) than set2,
// and 0 if both sets have equal preemption cost.
func CompareVictimSets(set1, set2 []fwk.PreemptionVictim) int {
	if len(set1) == 0 || len(set2) == 0 {
		return cmp.Compare(len(set1), len(set2))
	}

	// TODO(#142718): NumPDBViolations() is computed statically across all potential victims
	// in the domain, which is exact for DisruptionsAllowed == 0 but can miscount when
	// DisruptionsAllowed > 0 and earlier victims sharing the same PDB were not condemned.
	// Track DisruptionsAllowed dynamically in PodGroupPreemptionState.
	var pdb1, pdb2 int
	for _, v := range set1 {
		pdb1 += v.NumPDBViolations()
	}
	for _, v := range set2 {
		pdb2 += v.NumPDBViolations()
	}
	if c := cmp.Compare(pdb1, pdb2); c != 0 {
		return c
	}

	for i1, i2 := 0, 0; i1 < len(set1) && i2 < len(set2); {
		vi1, ok1 := set1[i1].(Victim)
		vi2, ok2 := set2[i2].(Victim)
		if !ok1 || !ok2 {
			break
		}
		p1, p2 := vi1.Priority(), vi2.Priority()
		if c := cmp.Compare(p1, p2); c != 0 {
			return c
		}
		var prioPods1, prioPods2 int
		for ; i1 < len(set1); i1++ {
			v, ok := set1[i1].(Victim)
			if !ok || v.Priority() != p1 {
				break
			}
			prioPods1 += len(v.Pods())
		}
		for ; i2 < len(set2); i2++ {
			v, ok := set2[i2].(Victim)
			if !ok || v.Priority() != p2 {
				break
			}
			prioPods2 += len(v.Pods())
		}
		if c := cmp.Compare(prioPods1, prioPods2); c != 0 {
			return c
		}
	}

	var pods1, pods2 int
	for _, v := range set1 {
		pods1 += len(v.Pods())
	}
	for _, v := range set2 {
		pods2 += len(v.Pods())
	}
	if c := cmp.Compare(pods1, pods2); c != 0 {
		return c
	}

	minLen := min(len(set1), len(set2))
	for i := range minLen {
		vi1, ok1 := set1[i].(Victim)
		vi2, ok2 := set2[i].(Victim)
		if ok1 && ok2 {
			if c := cmp.Compare(victimRank(vi1), victimRank(vi2)); c != 0 {
				return c
			}
			if c := cmp.Compare(len(vi1.Pods()), len(vi2.Pods())); c != 0 {
				return c
			}
		}
	}

	return cmp.Compare(len(set1), len(set2))
}

// FilterVictimsWithPDBViolation groups "victims" into "violatingVictims"
// and "nonViolatingVictims" based on whether their PDBs will be violated if they are
// preempted.
func FilterVictimsWithPDBViolation[T Victim](victims []T, pdbs []*policy.PodDisruptionBudget) (violatingVictims []ViolatingVictim[T], nonViolatingVictims []T) {
	pdbsAllowed := make([]int32, len(pdbs))
	podIsViolating := func(pod *v1.Pod) bool {
		for i, pdb := range pdbs {
			if pdb.Namespace != pod.Namespace {
				continue
			}
			selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
			if err != nil {
				// This object has an invalid selector, it does not match the pod
				continue
			}
			if !selector.Matches(labels.Set(pod.Labels)) {
				continue
			}

			// Existing in DisruptedPods means it has been processed in API server,
			// we don't treat it as a violating case.
			if _, exist := pdb.Status.DisruptedPods[pod.Name]; exist {
				continue
			}
			// Only decrement the matched pdb when it's not in its <DisruptedPods>;
			// otherwise we may over-decrement the budget number.
			pdbsAllowed[i]--
			// We have found a matching PDB.
			if pdbsAllowed[i] < 0 {
				return true
			}
		}

		return false
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

// getPodDisruptionBudgets returns all pod disruption budgets from the provided lister.
func getPodDisruptionBudgets(pdbLister policylisters.PodDisruptionBudgetLister) ([]*policy.PodDisruptionBudget, error) {
	return pdbLister.List(labels.Everything())
}
