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
	"context"
	"fmt"
	"slices"
	"sort"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	policylisters "k8s.io/client-go/listers/policy/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
)

type defaultPreemptionManager struct {
	snapshot  fwk.SharedLister
	executor  fwk.PreemptionExecutor
	pdbLister policylisters.PodDisruptionBudgetLister
	fts       feature.Features
}

var _ fwk.PreemptionManager = &defaultPreemptionManager{}

// NewPreemptionManager creates a PreemptionManager using the provided handle
// and features for victim discovery and actuation.
func NewPreemptionManager(fh fwk.Handle, fts feature.Features) fwk.PreemptionManager {
	return &defaultPreemptionManager{
		snapshot:  fh.SnapshotSharedLister(),
		executor:  NewExecutor(fh, fts),
		pdbLister: fh.SharedInformerFactory().Policy().V1().PodDisruptionBudgets().Lister(),
		fts:       fts,
	}
}

// GenerateVictims creates victims with grouping determined by the hierarchy's DisruptionPolicy and ordering determined by Priority and PDB violations.
func (m *defaultPreemptionManager) GenerateVictims(ctx context.Context, pgInfo fwk.PodGroupInfo) ([]fwk.PreemptionVictim, *fwk.Status) {
	logger := klog.FromContext(ctx)

	if getPreemptionPolicy(pgInfo, m.fts.EnablePodGroupPreemptionPolicy) == v1.PreemptNever {
		msg := "not eligible due to preemptionPolicy=Never."
		logger.V(5).Info("Preemptor is not eligible for preemption", "preemptor", klog.KObj(pgInfo), "type", pgInfo.GetType(), "reason", msg)
		return nil, fwk.NewStatus(fwk.Unschedulable, msg)
	}

	podGroupSnapshot := m.snapshot.PodGroups()
	var compositePodGroupSnapshot fwk.CompositePodGroupLister
	if m.fts.EnableCompositePodGroup {
		compositePodGroupSnapshot = m.snapshot.CompositePodGroups()
	}
	victims, err := getWorkloadPreemptionVictims(logger, m.snapshot, podGroupSnapshot, compositePodGroupSnapshot)
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to get victims: %w", err))
	}

	pdbs, err := getPodDisruptionBudgets(m.pdbLister)
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to get pod disruption budgets: %w", err))
	}

	return prepareDomainVictims(victims, pgInfo.GetPriority(), pdbs), nil
}

func (m *defaultPreemptionManager) Executor() fwk.PreemptionExecutor {
	return m.executor
}

func getPreemptionPolicy(pgInfo fwk.PodGroupInfo, enablePodGroupPreemptionPolicy bool) v1.PreemptionPolicy {
	if enablePodGroupPreemptionPolicy {
		return pgInfo.GetPreemptionPolicy()
	}
	for _, pod := range pgInfo.GetAllUnscheduledPods() {
		if p := pod.Spec.PreemptionPolicy; p != nil && *p == v1.PreemptNever {
			return *p
		}
	}
	return v1.PreemptLowerPriority
}

type victimWithPDBViolations struct {
	Victim
	numPDBViolations int
}

var _ fwk.PreemptionVictim = &victimWithPDBViolations{}

func (v *victimWithPDBViolations) NumPDBViolations() int {
	return v.numPDBViolations
}

// prepareDomainVictims filters domain victims that are eligible for preemption by preemptorPriority,
// computes PDB violations using the provided pdbs, and orders victims first by priority, and second such that
// non-violating victims precede violating victims in ascending priority order.
func prepareDomainVictims(victims []Victim, preemptorPriority int32, pdbs []*policy.PodDisruptionBudget) []fwk.PreemptionVictim {
	var potentialVictims []Victim
	for _, victim := range victims {
		if victim.Priority() < preemptorPriority {
			potentialVictims = append(potentialVictims, victim)
		}
	}

	sort.Slice(potentialVictims, func(i, j int) bool {
		return MoreImportantVictim(potentialVictims[i], potentialVictims[j])
	})

	violatingVictims, nonViolatingVictims := FilterVictimsWithPDBViolation(potentialVictims, pdbs)
	orderedVictims := make([]fwk.PreemptionVictim, 0, len(potentialVictims))
	for _, nv := range slices.Backward(nonViolatingVictims) {
		orderedVictims = append(orderedVictims, &victimWithPDBViolations{
			Victim:           nv,
			numPDBViolations: 0,
		})
	}
	for _, vv := range slices.Backward(violatingVictims) {
		orderedVictims = append(orderedVictims, &victimWithPDBViolations{
			Victim:           vv.Victim,
			numPDBViolations: vv.ViolateCount,
		})
	}
	return orderedVictims
}
