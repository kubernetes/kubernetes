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

package selectorspread

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	utilnode "k8s.io/kubernetes/pkg/util/node"
)

// SelectorSpread is a plugin that calculates selector spread priority.
type SelectorSpread struct {
	sharedLister           framework.SharedLister
	services               corelisters.ServiceLister
	replicationControllers corelisters.ReplicationControllerLister
	replicaSets            appslisters.ReplicaSetLister
	statefulSets           appslisters.StatefulSetLister
}

var _ framework.PreScorePlugin = &SelectorSpread{}
var _ framework.ScorePlugin = &SelectorSpread{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "SelectorSpread"
	// preScoreStateKey is the key in CycleState to SelectorSpread pre-computed data for Scoring.
	preScoreStateKey = "PreScore" + Name

	// When zone information is present, give 2/3 of the weighting to zone spreading, 1/3 to node spreading
	// TODO: Any way to justify this weighting?
	zoneWeighting float64 = 2.0 / 3.0
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *SelectorSpread) Name() string {
	return Name
}

// preScoreState computed at PreScore and used at Score.
type preScoreState struct {
	selector labels.Selector
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *preScoreState) Clone() framework.StateData {
	return s
}

// skipSelectorSpread returns true if the pod's TopologySpreadConstraints are specified.
// Note that this doesn't take into account default constraints defined for
// the PodTopologySpread plugin.
func skipSelectorSpread(pod *v1.Pod) bool {
	return len(pod.Spec.TopologySpreadConstraints) != 0
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
func (pl *SelectorSpread) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	if skipSelectorSpread(pod) {
		return 0, nil
	}

	c, err := state.Read(preScoreStateKey)
	if err != nil {
		return 0, framework.AsStatus(fmt.Errorf("reading %q from cycleState: %w", preScoreStateKey, err))
	}

	s, ok := c.(*preScoreState)
	if !ok {
		return 0, framework.AsStatus(fmt.Errorf("cannot convert saved state to tainttoleration.preScoreState"))
	}

	nodeInfo, err := pl.sharedLister.NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.AsStatus(fmt.Errorf("getting node %q from Snapshot: %w", nodeName, err))
	}

	count := countMatchingPods(pod.Namespace, s.selector, nodeInfo)
	return int64(count), nil
}

// NormalizeScore invoked after scoring all nodes.
// For this plugin, it calculates the score of each node
// based on the number of existing matching pods on the node
// where zone information is included on the nodes, it favors nodes
// in zones with fewer existing matching pods.
func (pl *SelectorSpread) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	if skipSelectorSpread(pod) {
		return nil
	}

	countsByZone := make(map[string]int64, 10)
	maxCountByZone := int64(0)
	maxCountByNodeName := int64(0)

	for i := range scores {
		if scores[i].Score > maxCountByNodeName {
			maxCountByNodeName = scores[i].Score
		}
		nodeInfo, err := pl.sharedLister.NodeInfos().Get(scores[i].Name)
		if err != nil {
			return framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", scores[i].Name, err))
		}
		zoneID := utilnode.GetZoneKey(nodeInfo.Node())
		if zoneID == "" {
			continue
		}
		countsByZone[zoneID] += scores[i].Score
	}

	for zoneID := range countsByZone {
		if countsByZone[zoneID] > maxCountByZone {
			maxCountByZone = countsByZone[zoneID]
		}
	}

	haveZones := len(countsByZone) != 0

	maxCountByNodeNameFloat64 := float64(maxCountByNodeName)
	maxCountByZoneFloat64 := float64(maxCountByZone)
	MaxNodeScoreFloat64 := float64(framework.MaxNodeScore)

	for i := range scores {
		// initializing to the default/max node score of maxPriority
		fScore := MaxNodeScoreFloat64
		if maxCountByNodeName > 0 {
			fScore = MaxNodeScoreFloat64 * (float64(maxCountByNodeName-scores[i].Score) / maxCountByNodeNameFloat64)
		}
		// If there is zone information present, incorporate it
		if haveZones {
			nodeInfo, err := pl.sharedLister.NodeInfos().Get(scores[i].Name)
			if err != nil {
				return framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", scores[i].Name, err))
			}

			zoneID := utilnode.GetZoneKey(nodeInfo.Node())
			if zoneID != "" {
				zoneScore := MaxNodeScoreFloat64
				if maxCountByZone > 0 {
					zoneScore = MaxNodeScoreFloat64 * (float64(maxCountByZone-countsByZone[zoneID]) / maxCountByZoneFloat64)
				}
				fScore = (fScore * (1.0 - zoneWeighting)) + (zoneWeighting * zoneScore)
			}
		}
		scores[i].Score = int64(fScore)
	}
	return nil
}

// ScoreExtensions of the Score plugin.
func (pl *SelectorSpread) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// PreScore builds and writes cycle state used by Score and NormalizeScore.
func (pl *SelectorSpread) PreScore(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodes []*v1.Node) *framework.Status {
	if skipSelectorSpread(pod) {
		return nil
	}
	var selector labels.Selector
	selector = helper.DefaultSelector(
		pod,
		pl.services,
		pl.replicationControllers,
		pl.replicaSets,
		pl.statefulSets,
	)
	state := &preScoreState{
		selector: selector,
	}
	cycleState.Write(preScoreStateKey, state)
	return nil
}

// New initializes a new plugin and returns it.
func New(_ runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	sharedLister := handle.SnapshotSharedLister()
	if sharedLister == nil {
		return nil, fmt.Errorf("SnapshotSharedLister is nil")
	}
	sharedInformerFactory := handle.SharedInformerFactory()
	if sharedInformerFactory == nil {
		return nil, fmt.Errorf("SharedInformerFactory is nil")
	}
	return &SelectorSpread{
		sharedLister:           sharedLister,
		services:               sharedInformerFactory.Core().V1().Services().Lister(),
		replicationControllers: sharedInformerFactory.Core().V1().ReplicationControllers().Lister(),
		replicaSets:            sharedInformerFactory.Apps().V1().ReplicaSets().Lister(),
		statefulSets:           sharedInformerFactory.Apps().V1().StatefulSets().Lister(),
	}, nil
}

// countMatchingPods counts pods based on namespace and matching all selectors
func countMatchingPods(namespace string, selector labels.Selector, nodeInfo *framework.NodeInfo) int {
	if len(nodeInfo.Pods) == 0 || selector.Empty() {
		return 0
	}
	count := 0
	for _, p := range nodeInfo.Pods {
		// Ignore pods being deleted for spreading purposes
		// Similar to how it is done for SelectorSpreadPriority
		if namespace == p.Pod.Namespace && p.Pod.DeletionTimestamp == nil {
			if selector.Matches(labels.Set(p.Pod.Labels)) {
				count++
			}
		}
	}
	return count
}
