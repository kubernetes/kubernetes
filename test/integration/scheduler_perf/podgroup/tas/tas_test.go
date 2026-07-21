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

package tas

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	_ "k8s.io/component-base/logs/json/register"
	fwk "k8s.io/kube-scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	perf "k8s.io/kubernetes/test/integration/scheduler_perf"
)

// nodePoolPlacementName is the name of the out-of-tree PlacementGenerate plugin used by the
// multi-placement workload. It must match the plugin name enabled in
// config/multi-placement.yaml.
const nodePoolPlacementName = "NodePoolPlacement"

// nodePoolPlacement is a test PlacementGenerate plugin that keeps only nodes in a fixed "pool".
// Running it alongside the in-tree TopologyPlacement plugin exercises the framework's merging of
// multiple PlacementGenerate plugins under load.
type nodePoolPlacement struct{}

func (p *nodePoolPlacement) Name() string { return nodePoolPlacementName }

func (p *nodePoolPlacement) GeneratePlacements(_ context.Context, _ fwk.PodGroupCycleState, _ fwk.PodGroupInfo, parent *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	var nodes []fwk.NodeInfo
	for _, n := range parent.Nodes {
		if n.Node().Labels["pool"] == "a" {
			nodes = append(nodes, n)
		}
	}
	if len(nodes) == 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "no nodes in pool a")
	}
	return &fwk.GeneratePlacementsResult{Placements: []*fwk.Placement{{Name: "pool-a", Nodes: nodes}}}, nil
}

func newNodePoolPlacement(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
	return &nodePoolPlacement{}, nil
}

// multiPlacementRegistry registers the extra PlacementGenerate plugin exercised by the
// multi-placement workload. It is inert for workloads that don't enable the plugin.
var multiPlacementRegistry = frameworkruntime.Registry{nodePoolPlacementName: newNodePoolPlacement}

func TestMain(m *testing.M) {
	if err := perf.InitTests(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	m.Run()
}

func TestSchedulerPerf(t *testing.T) {
	perf.RunIntegrationPerfScheduling(t, "performance-config.yaml",
		perf.WithPodsSchedulingTimeout(20*time.Minute),
		perf.WithOutOfTreePluginRegistry(multiPlacementRegistry))
}

func BenchmarkPerfScheduling(b *testing.B) {
	perf.RunBenchmarkPerfScheduling(b, "performance-config.yaml", "tas", multiPlacementRegistry, perf.WithPodsSchedulingTimeout(20*time.Minute))
}
