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
	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	utilnode "k8s.io/kubernetes/pkg/util/node"
)

// DefaultPodTopologySpread is a plugin that calculates selector spread priority.
type DefaultPodTopologySpread struct {
	handle framework.FrameworkHandle
}

var _ framework.ScorePlugin = &DefaultPodTopologySpread{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "DefaultPodTopologySpread"
	// postFilterStateKey is the key in CycleState to DefaultPodTopologySpread pre-computed data for Scoring.
	postFilterStateKey = "PostFilter" + Name

	// When zone information is present, give 2/3 of the weighting to zone spreading, 1/3 to node spreading
	// TODO: Any way to justify this weighting?
	zoneWeighting float64 = 2.0 / 3.0
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *DefaultPodTopologySpread) Name() string {
	return Name
}

// postFilterState computed at PostFilter and used at Score.
type postFilterState struct {
	selector labels.Selector
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *postFilterState) Clone() framework.StateData {
	return s
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
func (pl *DefaultPodTopologySpread) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	c, err := state.Read(postFilterStateKey)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Error reading %q from cycleState: %v", postFilterStateKey, err))
	}

	s, ok := c.(*postFilterState)
	if !ok {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("%+v convert to tainttoleration.postFilterState error", c))
	}

	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	count := countMatchingPods(pod.Namespace, s.selector, nodeInfo)
	return int64(count), nil
}

// NormalizeScore invoked after scoring all nodes.
// For this plugin, it calculates the source of each node
// based on the number of existing matching pods on the node
// where zone information is included on the nodes, it favors nodes
// in zones with fewer existing matching pods.
func (pl *DefaultPodTopologySpread) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	countsByZone := make(map[string]int64, 10)
	maxCountByZone := int64(0)
	maxCountByNodeName := int64(0)

	for i := range scores {
		if scores[i].Score > maxCountByNodeName {
			maxCountByNodeName = scores[i].Score
		}
		nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(scores[i].Name)
		if err != nil {
			return framework.NewStatus(framework.Error, err.Error())
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
			nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(scores[i].Name)
			if err != nil {
				return framework.NewStatus(framework.Error, err.Error())
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
		if klog.V(10) {
			klog.Infof(
				"%v -> %v: SelectorSpreadPriority, Score: (%d)", pod.Name, scores[i].Name, int64(fScore),
			)
		}
	}
	return nil
}

// ScoreExtensions of the Score plugin.
func (pl *DefaultPodTopologySpread) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// PostFilter builds and writes cycle state used by Score and NormalizeScore.
func (pl *DefaultPodTopologySpread) PostFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodes []*v1.Node, _ framework.NodeToStatusMap) *framework.Status {
	var selector labels.Selector
	informerFactory := pl.handle.SharedInformerFactory()
	selector = getSelector(
		pod,
		informerFactory.Core().V1().Services().Lister(),
		informerFactory.Core().V1().ReplicationControllers().Lister(),
		informerFactory.Apps().V1().ReplicaSets().Lister(),
		informerFactory.Apps().V1().StatefulSets().Lister(),
	)
	state := &postFilterState{
		selector: selector,
	}
	cycleState.Write(postFilterStateKey, state)
	return nil
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	return &DefaultPodTopologySpread{
		handle: handle,
	}, nil
}

// countMatchingPods counts pods based on namespace and matching all selectors
func countMatchingPods(namespace string, selector labels.Selector, nodeInfo *schedulernodeinfo.NodeInfo) int {
	if len(nodeInfo.Pods()) == 0 || selector.Empty() {
		return 0
	}
	count := 0
	for _, pod := range nodeInfo.Pods() {
		// Ignore pods being deleted for spreading purposes
		// Similar to how it is done for SelectorSpreadPriority
		if namespace == pod.Namespace && pod.DeletionTimestamp == nil {
			if selector.Matches(labels.Set(pod.Labels)) {
				count++
			}
		}
	}
	return count
}

// getSelector returns a selector for the services, RCs, RSs, and SSs matching the given pod.
func getSelector(pod *v1.Pod, sl corelisters.ServiceLister, cl corelisters.ReplicationControllerLister, rsl appslisters.ReplicaSetLister, ssl appslisters.StatefulSetLister) labels.Selector {
	labelSet := make(labels.Set)
	// Since services, RCs, RSs and SSs match the pod, they won't have conflicting
	// labels. Merging is safe.

	if services, err := schedulerlisters.GetPodServices(sl, pod); err == nil {
		for _, service := range services {
			labelSet = labels.Merge(labelSet, service.Spec.Selector)
		}
	}

	if rcs, err := cl.GetPodControllers(pod); err == nil {
		for _, rc := range rcs {
			labelSet = labels.Merge(labelSet, rc.Spec.Selector)
		}
	}

	selector := labels.NewSelector()
	if len(labelSet) != 0 {
		selector = labelSet.AsSelector()
	}

	if rss, err := rsl.GetPodReplicaSets(pod); err == nil {
		for _, rs := range rss {
			if other, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector); err == nil {
				if r, ok := other.Requirements(); ok {
					selector = selector.Add(r...)
				}
			}
		}
	}

	if sss, err := ssl.GetPodStatefulSets(pod); err == nil {
		for _, ss := range sss {
			if other, err := metav1.LabelSelectorAsSelector(ss.Spec.Selector); err == nil {
				if r, ok := other.Requirements(); ok {
					selector = selector.Add(r...)
				}
			}
		}
	}

	return selector
}
