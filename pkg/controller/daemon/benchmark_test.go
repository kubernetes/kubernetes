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

package daemon

import (
	"context"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

// exclusionNodeAffinity mirrors the node affinity commonly carried by
// cluster-wide DaemonSets (log shippers, node agents) that must avoid a few
// special node classes: a single term with several NotIn expressions.
func exclusionNodeAffinity() *v1.Affinity {
	return &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{Key: "example.com/papaya", Operator: v1.NodeSelectorOpNotIn, Values: []string{"true"}},
						{Key: "example.com/kumquat", Operator: v1.NodeSelectorOpNotIn, Values: []string{"true"}},
						{Key: "quince", Operator: v1.NodeSelectorOpNotIn, Values: []string{"dragonfruit"}},
						{Key: "lychee", Operator: v1.NodeSelectorOpNotIn, Values: []string{"rambutan"}},
						{Key: "example.com/durian", Operator: v1.NodeSelectorOpNotIn, Values: []string{"mangosteen", "tamarind", "persimmon"}},
						{Key: "example.com/guava", Operator: v1.NodeSelectorOpNotIn, Values: []string{"starfruit"}},
					},
				}},
			},
		},
	}
}

// BenchmarkUpdateDaemonSetStatus measures the per-node evaluation loop shared
// by the DaemonSet controller's sync paths (updateDaemonSetStatus, manage,
// rollingUpdate all evaluate NodeShouldRunDaemonPod for every node in the
// cluster). The dominant per-node cost is evaluating whether the daemon pod
// fits the node, so this approximates the controller's per-sync CPU cost on
// an otherwise steady-state DaemonSet.
func BenchmarkUpdateDaemonSetStatus(b *testing.B) {
	for _, tc := range []struct {
		name     string
		affinity *v1.Affinity
	}{
		{name: "no-affinity", affinity: nil},
		{name: "exclusion-affinity", affinity: exclusionNodeAffinity()},
	} {
		for _, numNodes := range []int{1000, 10000} {
			b.Run(fmt.Sprintf("affinity=%s/nodes=%d", tc.name, numNodes), func(b *testing.B) {
				// Silence per-iteration log output so the benchmark measures
				// the evaluation loop, not the test logger.
				logger := ktesting.NewLogger(b, ktesting.NewConfig(ktesting.Verbosity(0)))
				ctx := klog.NewContext(context.Background(), logger)
				ds := newDaemonSet("bench")
				ds.Spec.Template.Spec.Affinity = tc.affinity
				// Pre-set the status to the values the loop computes so that
				// storeDaemonSetStatus short-circuits and each iteration
				// measures only the per-node evaluation loop, not the fake
				// client's status update.
				ds.Status.DesiredNumberScheduled = int32(numNodes)
				ds.Status.NumberUnavailable = int32(numNodes)
				manager, _, _, err := newTestController(ctx, ds)
				if err != nil {
					b.Fatalf("error creating DaemonSets controller: %v", err)
				}
				if err := manager.dsStore.Add(ds); err != nil {
					b.Fatal(err)
				}
				nodeList := make([]*v1.Node, 0, numNodes)
				for i := range numNodes {
					node := newNode(fmt.Sprintf("node-%d", i), map[string]string{
						"kubernetes.io/hostname":        fmt.Sprintf("node-%d", i),
						"topology.kubernetes.io/zone":   fmt.Sprintf("zone-%d", i%3),
						"topology.kubernetes.io/region": "region-1",
					})
					if err := manager.nodeStore.Add(node); err != nil {
						b.Fatal(err)
					}
					nodeList = append(nodeList, node)
				}

				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if err := manager.updateDaemonSetStatus(ctx, ds, nodeList, "hash", false); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
