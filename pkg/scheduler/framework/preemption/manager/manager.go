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

package manager

import (
	"context"
	"fmt"

	policylisters "k8s.io/client-go/listers/policy/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kube-scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
)

type defaultPreemptionManager struct {
	snapshot  fwk.SharedLister
	executor  fwk.PreemptionExecutor
	pdbLister policylisters.PodDisruptionBudgetLister
	fts       feature.Features
}

var _ fwk.PreemptionManager = &defaultPreemptionManager{}

// New creates a PreemptionManager using the provided handle
// and features for victim discovery and actuation.
func New(fh fwk.Handle, fts feature.Features) fwk.PreemptionManager {
	return NewDefaultPreemptionManager(fh, fts)
}

// NewDefaultPreemptionManager creates a PreemptionManager using the provided handle
// and features for victim discovery and actuation.
func NewDefaultPreemptionManager(fh fwk.Handle, fts feature.Features) fwk.PreemptionManager {
	return &defaultPreemptionManager{
		snapshot:  fh.SnapshotSharedLister(),
		executor:  NewExecutor(fh, fts),
		pdbLister: fh.SharedInformerFactory().Policy().V1().PodDisruptionBudgets().Lister(),
		fts:       fts,
	}
}

// GenerateVictims creates victims with grouping determined by the hierarchy's DisruptionPolicy and ordering determined by Priority and PDB violations.
func (m *defaultPreemptionManager) GenerateVictims(ctx context.Context, pgInfo fwk.PodGroupInfo) ([]fwk.Victim, error) {
	logger := klog.FromContext(ctx)
	podGroupSnapshot := m.snapshot.PodGroups()
	compositePodGroupSnapshot := m.snapshot.CompositePodGroups()
	victims, err := preemption.GetWorkloadPreemptionVictims(logger, m.snapshot, podGroupSnapshot, compositePodGroupSnapshot)
	if err != nil {
		return nil, fmt.Errorf("failed to get victims: %w", err)
	}

	pdbs, err := preemption.GetPodDisruptionBudgets(m.pdbLister)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod disruption budgets: %w", err)
	}

	preemptorPriority := getPreemptorPriority(pgInfo)
	return preemption.PrepareDomainVictims(victims, preemptorPriority, pdbs), nil
}

func (m *defaultPreemptionManager) Executor() fwk.PreemptionExecutor {
	return m.executor
}

func getPreemptorPriority(pgInfo fwk.PodGroupInfo) int32 {
	if cpg := pgInfo.GetCompositePodGroup(); cpg != nil {
		return util.CompositePodGroupPriority(cpg)
	}
	if pg := pgInfo.GetPodGroup(); pg != nil {
		return util.PodGroupPriority(pg)
	}
	return 0
}
