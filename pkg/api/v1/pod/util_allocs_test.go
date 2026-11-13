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

package pod_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

// TestContainerPairsAllocs is in an external test package because inlining and escape analysis
// differ for callers in another package, which is where ContainerPairs is used.
func TestContainerPairsAllocs(t *testing.T) {
	pod := &v1.Pod{Spec: v1.PodSpec{
		InitContainers:      []v1.Container{{Name: "init"}},
		Containers:          []v1.Container{{Name: "a"}, {Name: "x"}, {Name: "c"}},
		EphemeralContainers: []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug"}}},
	}}
	oldPod := &v1.Pod{Spec: v1.PodSpec{
		InitContainers: []v1.Container{{Name: "init"}},
		Containers:     []v1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
	}}
	pairs := 0
	allocs := testing.AllocsPerRun(10, func() {
		for range podutil.ContainerPairs(pod, oldPod, podutil.AllContainers) {
			pairs++
		}
	})
	assert.Positive(t, pairs)
	assert.Zero(t, allocs)
}
