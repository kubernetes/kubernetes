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

package topologyaware

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	"k8s.io/apimachinery/pkg/runtime"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha2"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.TopologyPlacementGenerator
)

// TopologyPlacement is a plugin that generates placements for a pod group based on its topology constraints.
type TopologyPlacement struct {
	handle         fwk.Handle
	podGroupLister schedulinglisters.PodGroupLister
}

var _ fwk.PlacementGeneratePlugin = &TopologyPlacement{}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, fh fwk.Handle, fts feature.Features) (*TopologyPlacement, error) {
	return &TopologyPlacement{
		handle:         fh,
		podGroupLister: fh.SharedInformerFactory().Scheduling().V1alpha2().PodGroups().Lister(),
	}, nil
}

// Name returns name of the plugin.
func (pl *TopologyPlacement) Name() string {
	return Name
}

// GeneratePlacements generates placements for a pod group based on the topology constraints in the pod group spec.
// It uses the parent placement to find the nodes that are available for placement.
func (pl *TopologyPlacement) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	podGroupResource, err := pl.podGroupLister.PodGroups(podGroup.GetNamespace()).Get(podGroup.GetName())
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	topologyKey, ok := pl.getTopologyKey(podGroupResource)
	if !ok {
		// No topology constraints, return a single placement with no constraints.
		return &fwk.GeneratePlacementsResult{Placements: []*fwk.Placement{parentPlacement}}, nil
	}

	var requiredDomain *string
	scheduledPods, err := pl.getScheduledPods(podGroup)
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	if len(scheduledPods) > 0 {
		scheduledDomain, err := pl.getScheduledPodsTopologyDomain(topologyKey, scheduledPods)
		if err != nil {
			return nil, fwk.AsStatus(fmt.Errorf("cannot determine domain for already scheduled pods: %w", err))
		}
		requiredDomain = &scheduledDomain
	}

	nodesPerTopologyDomain := make(map[string][]fwk.NodeInfo)
	for _, node := range parentPlacement.Nodes {
		if domain, ok := node.Node().Labels[topologyKey]; ok {
			if requiredDomain == nil || *requiredDomain == domain {
				nodesPerTopologyDomain[domain] = append(nodesPerTopologyDomain[domain], node)
			}
		}
	}

	placements := make([]*fwk.Placement, 0, len(nodesPerTopologyDomain))
	for topologyDomain, nodes := range nodesPerTopologyDomain {
		if len(nodes) > 0 {
			placements = append(placements, &fwk.Placement{
				Name:  topologyDomain,
				Nodes: nodes,
			})
		}
	}

	return &fwk.GeneratePlacementsResult{Placements: placements}, nil
}

func (pl *TopologyPlacement) getScheduledPodsTopologyDomain(topologyKey string, scheduledPods []*v1.Pod) (string, error) {
	topologyDomain := ""
	for _, pod := range scheduledPods {
		node, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(pod.Spec.NodeName)
		if err != nil {
			return "", fmt.Errorf("getting node for pod %v: %w", klog.KObj(pod), err)
		}
		domain, ok := node.Node().Labels[topologyKey]
		if !ok {
			return "", fmt.Errorf("no topology domain found for pod %v", klog.KObj(pod))
		}
		if topologyDomain != "" && topologyDomain != domain {
			return "", fmt.Errorf("more than 1 domain found for pod group: %v, %v", topologyDomain, domain)
		}
		topologyDomain = domain
	}
	return topologyDomain, nil
}

// getTopologyKey returns the topology key for the pod group if there's any specified.
func (pl *TopologyPlacement) getTopologyKey(podGroupResource *schedulingapi.PodGroup) (string, bool) {
	if schedulingConstraints := podGroupResource.Spec.SchedulingConstraints; schedulingConstraints != nil && len(schedulingConstraints.Topology) > 0 {
		// Right now, we only support a single topology constraint on the API level.
		return schedulingConstraints.Topology[0].Key, true
	}
	return "", false
}

func (pl *TopologyPlacement) getScheduledPods(podGroup fwk.PodGroupInfo) ([]*v1.Pod, error) {
	name := podGroup.GetName()
	podGroupState, err := pl.handle.SnapshotSharedLister().PodGroupStates().Get(podGroup.GetNamespace(), name)
	if err != nil {
		return nil, err
	}
	return podGroupState.ScheduledPods(), nil
}
