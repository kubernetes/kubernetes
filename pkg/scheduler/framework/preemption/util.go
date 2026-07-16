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
	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
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
//  2. Workload Type (if enabled):
//     If GenericWorkload is enabled, PodGroups are considered more important
//     than individual Pods to preserve group integrity.
//
//  3. Runtime / Start Time (for individual Pods):
//     For two individual Pods, the one that started earlier (longer runtime)
//     is more important. This honors "first-come, first-served".
//
//  4. Group Size (for PodGroups):
//     If both units are PodGroups, the one with more members (larger size) is considered
//     more important. This avoids the high cost of rescheduling massive jobs.
//
//  5. Start Time (Tie-breaker for PodGroups):
//     If sizes are equal, the group that started earlier (has the oldest pod)
//     is more important.
func MoreImportantVictim(vi1, vi2 Victim, genericWorkloadEnabled bool) bool {
	if vi1.Priority() != vi2.Priority() {
		return vi1.Priority() > vi2.Priority()
	}

	if !genericWorkloadEnabled {
		return vi1.EarliestStartTime().Before(vi2.EarliestStartTime())
	}

	if vi1.IsPodGroup() != vi2.IsPodGroup() {
		return vi1.IsPodGroup()
	}

	if vi1.IsPodGroup() && len(vi1.Pods()) != len(vi2.Pods()) {
		return len(vi1.Pods()) > len(vi2.Pods())
	}

	return vi1.EarliestStartTime().Before(vi2.EarliestStartTime())
}

// GetPodPriority returns the effective preemption priority of a pod. If the pod belongs to
// a pod group, it returns the priority of the pod group.
// If podGroupLister is nil or the pod does not belong to a pod group, it returns the pod's own priority.
func GetPodPriority(p *v1.Pod, podGroupLister fwk.PodGroupLister) int32 {
	if pg := getPodGroup(p, podGroupLister); pg != nil {
		return util.PodGroupPriority(pg)
	}
	return corev1helpers.PodPriority(p)
}

// FilterVictimsWithPDBViolation groups "victims" into "violatingVictims"
// and "nonViolatingVictims" based on whether their PDBs will be violated if they are
// preempted.
func FilterVictimsWithPDBViolation[T Victim](victims []T, pdbs []*policy.PodDisruptionBudget) (violatingVictims []ViolatingVictim[T], nonViolatingVictims []T) {
	pdbsAllowed := make([]int32, len(pdbs))
	podIsViolating := func(pod *v1.Pod) bool {
		if len(pod.Labels) == 0 {
			return false
		}

		for i, pdb := range pdbs {
			if pdb.Namespace != pod.Namespace {
				continue
			}
			selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
			if err != nil {
				// This object has an invalid selector, it does not match the pod
				continue
			}
			// A PDB with a nil or empty selector matches nothing.
			if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
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
