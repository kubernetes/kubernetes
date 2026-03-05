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

func filterNUMATopology(topology []cadvisorapi.Node, excludedNUMANodes []int) []cadvisorapi.Node {
	if len(excludedNUMANodes) == 0 {
		return topology
	}

	excludedSet := sets.New(excludedNUMANodes...)
	filtered := make([]cadvisorapi.Node, 0, len(topology))
	for _, node := range topology {
		if !excludedSet.Has(node.Id) {
			filtered = append(filtered, node)
		}
	}
	return filtered
}
