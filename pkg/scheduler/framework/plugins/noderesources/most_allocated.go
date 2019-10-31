/*
Copyright 2019 The Kubernetes Authors.

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

package noderesources

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// MostAllocated is a score plugin that favors nodes with high allocation based on requested resources.
type MostAllocated struct {
	handle framework.FrameworkHandle
}

var _ = framework.ScorePlugin(&MostAllocated{})

// MostAllocatedName is the name of the plugin used in the plugin registry and configurations.
const MostAllocatedName = "NodeResourcesMostAllocated"

// Name returns name of the plugin. It is used in logs, etc.
func (ma *MostAllocated) Name() string {
	return MostAllocatedName
}

// Score invoked at the Score extension point.
func (ma *MostAllocated) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := ma.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	// MostRequestedPriorityMap does not use priority metadata, hence we pass nil here
	s, err := priorities.MostRequestedPriorityMap(pod, nil, nodeInfo)
	return s.Score, migration.ErrorToFrameworkStatus(err)
}

// ScoreExtensions of the Score plugin.
func (ma *MostAllocated) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// NewMostAllocated initializes a new plugin and returns it.
func NewMostAllocated(_ *runtime.Unknown, h framework.FrameworkHandle) (framework.Plugin, error) {
	return &MostAllocated{handle: h}, nil
}
