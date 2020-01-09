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
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// ResourceLimits is a score plugin that increases score of input node by 1 if the node satisfies
// input pod's resource limits
type ResourceLimits struct {
	handle framework.FrameworkHandle
}

var _ = framework.PostFilterPlugin(&ResourceLimits{})
var _ = framework.ScorePlugin(&ResourceLimits{})

const (
	// ResourceLimitsName is the name of the plugin used in the plugin registry and configurations.
	ResourceLimitsName = "NodeResourceLimits"

	// postFilterStateKey is the key in CycleState to NodeResourceLimits pre-computed data.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	postFilterStateKey = "PostFilter" + ResourceLimitsName
)

// postFilterState computed at PostFilter and used at Score.
type postFilterState struct {
	podResourceRequest *schedulernodeinfo.Resource
}

// Clone the postFilter state.
func (s *postFilterState) Clone() framework.StateData {
	return s
}

// Name returns name of the plugin. It is used in logs, etc.
func (rl *ResourceLimits) Name() string {
	return ResourceLimitsName
}

// PostFilter builds and writes cycle state used by Score and NormalizeScore.
func (rl *ResourceLimits) PostFilter(
	pCtx context.Context,
	cycleState *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
	_ framework.NodeToStatusMap,
) *framework.Status {
	if len(nodes) == 0 {
		// No nodes to score.
		return nil
	}

	if rl.handle.SnapshotSharedLister() == nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("empty shared lister"))
	}
	s := &postFilterState{
		podResourceRequest: getResourceLimits(pod),
	}
	cycleState.Write(postFilterStateKey, s)
	return nil
}

func getPodResource(cycleState *framework.CycleState) (*schedulernodeinfo.Resource, error) {
	c, err := cycleState.Read(postFilterStateKey)
	if err != nil {
		return nil, fmt.Errorf("Error reading %q from cycleState: %v", postFilterStateKey, err)
	}

	s, ok := c.(*postFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to ResourceLimits.postFilterState error", c)
	}
	return s.podResourceRequest, nil
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`.
// Currently works as follows:
// If a node does not publish its allocatable resources (cpu and memory both), the node score is not affected.
// If a pod does not specify its cpu and memory limits both, the node score is not affected.
// If one or both of cpu and memory limits of the pod are satisfied, the node is assigned a score of 1.
// Rationale of choosing the lowest score of 1 is that this is mainly selected to break ties between nodes that have
// same scores assigned by one of least and most requested priority functions.
func (rl *ResourceLimits) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := rl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil || nodeInfo.Node() == nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v, node is nil: %v", nodeName, err, nodeInfo.Node() == nil))
	}
	allocatableResources := nodeInfo.AllocatableResource()
	podLimits, err := getPodResource(state)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, err.Error())
	}

	cpuScore := computeScore(podLimits.MilliCPU, allocatableResources.MilliCPU)
	memScore := computeScore(podLimits.Memory, allocatableResources.Memory)

	score := int64(0)
	if cpuScore == 1 || memScore == 1 {
		score = 1
	}
	return score, nil
}

// ScoreExtensions of the Score plugin.
func (rl *ResourceLimits) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// NewResourceLimits initializes a new plugin and returns it.
func NewResourceLimits(_ *runtime.Unknown, h framework.FrameworkHandle) (framework.Plugin, error) {
	return &ResourceLimits{handle: h}, nil
}

// getResourceLimits computes resource limits for input pod.
// The reason to create this new function is to be consistent with other
// priority functions because most or perhaps all priority functions work
// with schedulernodeinfo.Resource.
func getResourceLimits(pod *v1.Pod) *schedulernodeinfo.Resource {
	result := &schedulernodeinfo.Resource{}
	for _, container := range pod.Spec.Containers {
		result.Add(container.Resources.Limits)
	}

	// take max_resource(sum_pod, any_init_container)
	for _, container := range pod.Spec.InitContainers {
		result.SetMaxResource(container.Resources.Limits)
	}

	return result
}

// computeScore returns 1 if limit value is less than or equal to allocatable
// value, otherwise it returns 0.
func computeScore(limit, allocatable int64) int64 {
	if limit != 0 && allocatable != 0 && limit <= allocatable {
		return 1
	}
	return 0
}
