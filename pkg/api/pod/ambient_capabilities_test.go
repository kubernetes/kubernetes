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

package pod

import (
	"testing"

	"github.com/stretchr/testify/require"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropDisabledAmbientCapabilities(t *testing.T) {
	for _, tc := range []struct {
		name                    string
		enabled, oldInUse, want bool
	}{
		{"disabled", false, false, false},
		{"enabled", true, false, true},
		{"preserve existing on rollback", false, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AmbientCapabilities, tc.enabled)
			container := api.Container{SecurityContext: &api.SecurityContext{Capabilities: &api.Capabilities{Ambient: []api.Capability{"NET_BIND_SERVICE"}}}}
			spec := &api.PodSpec{Containers: []api.Container{*container.DeepCopy()}, InitContainers: []api.Container{*container.DeepCopy()}, EphemeralContainers: []api.EphemeralContainer{{EphemeralContainerCommon: api.EphemeralContainerCommon(*container.DeepCopy())}}}
			var old *api.PodSpec
			if tc.oldInUse {
				old = spec.DeepCopy()
			}
			dropDisabledFields(spec, nil, old, nil)
			visited := 0
			VisitContainers(spec, AllContainers, func(c *api.Container, _ ContainerType) bool {
				visited++
				require.Equal(t, tc.want, len(c.SecurityContext.Capabilities.Ambient) > 0)
				return true
			})
			require.Equal(t, 3, visited)
		})
	}
}
