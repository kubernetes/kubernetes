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

package cm

import (
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// filterNUMATopology returns a new slice of cadvisorapi.Node excluding nodes
// whose IDs appear in excludedNUMANodes. Each kept node is deep-copied so
// callers can mutate the result without affecting the original topology.
func filterNUMATopology(topology []cadvisorapi.Node, excludedNUMANodes []int) []cadvisorapi.Node {
	excludedSet := sets.New(excludedNUMANodes...)
	filtered := make([]cadvisorapi.Node, 0, len(topology))
	for _, node := range topology {
		if !excludedSet.Has(node.Id) {
			cp := cadvisorapi.Node{
				Id:     node.Id,
				Memory: node.Memory,
			}
			cp.HugePages = append([]cadvisorapi.HugePagesInfo(nil), node.HugePages...)
			cp.Cores = make([]cadvisorapi.Core, len(node.Cores))
			for i, core := range node.Cores {
				cp.Cores[i] = core
				cp.Cores[i].Threads = append([]int(nil), core.Threads...)
				cp.Cores[i].Caches = append([]cadvisorapi.Cache(nil), core.Caches...)
				cp.Cores[i].UncoreCaches = append([]cadvisorapi.Cache(nil), core.UncoreCaches...)
			}
			cp.Caches = append([]cadvisorapi.Cache(nil), node.Caches...)
			cp.Distances = append([]uint64(nil), node.Distances...)
			filtered = append(filtered, cp)
		}
	}
	return filtered
}
