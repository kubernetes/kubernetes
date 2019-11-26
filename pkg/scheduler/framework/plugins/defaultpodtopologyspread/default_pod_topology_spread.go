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

package defaultpodtopologyspread

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// DefaultPodTopologySpread is a plugin that calculates selector spread priority.
type DefaultPodTopologySpread struct {
	handle                        framework.FrameworkHandle
	calculateSpreadPriorityMap    priorities.PriorityMapFunction
	calculateSpreadPriorityReduce priorities.PriorityReduceFunction
}

var _ framework.ScorePlugin = &DefaultPodTopologySpread{}

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = "DefaultPodTopologySpread"

// Name returns name of the plugin. It is used in logs, etc.
func (pl *DefaultPodTopologySpread) Name() string {
	return Name
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
func (pl *DefaultPodTopologySpread) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	meta := migration.PriorityMetadata(state)
	s, err := pl.calculateSpreadPriorityMap(pod, meta, nodeInfo)
	return s.Score, migration.ErrorToFrameworkStatus(err)
}

// NormalizeScore invoked after scoring all nodes.
func (pl *DefaultPodTopologySpread) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	meta := migration.PriorityMetadata(state)
	err := pl.calculateSpreadPriorityReduce(pod, meta, pl.handle.SnapshotSharedLister(), scores)
	return migration.ErrorToFrameworkStatus(err)
}

// ScoreExtensions of the Score plugin.
func (pl *DefaultPodTopologySpread) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	calculateSpreadPriorityMap, calculateSpreadPriorityReduce := priorities.NewSelectorSpreadPriority(
		informerFactory.Core().V1().Services().Lister(),
		informerFactory.Core().V1().ReplicationControllers().Lister(),
		informerFactory.Apps().V1().ReplicaSets().Lister(),
		informerFactory.Apps().V1().StatefulSets().Lister(),
	)

	return &DefaultPodTopologySpread{
		handle:                        handle,
		calculateSpreadPriorityMap:    calculateSpreadPriorityMap,
		calculateSpreadPriorityReduce: calculateSpreadPriorityReduce,
	}, nil
}
