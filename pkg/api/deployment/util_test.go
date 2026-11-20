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

package deployment

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/utils/ptr"
)

func TestWorkloadItemForDeploymentMinCountDefaulting(t *testing.T) {
	gangConfig := func(minCount *int32) *workloadbuilder.SchedulingConfig {
		return &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{Gang: &workloadbuilder.GangSchedulingPolicy{MinCount: minCount}},
		}
	}

	cases := map[string]struct {
		replicas *int32
		cfg      *workloadbuilder.SchedulingConfig
		wantMin  *int32
	}{
		"gang unset minCount defaults to replicas": {
			replicas: ptr.To[int32](5),
			cfg:      gangConfig(nil),
			wantMin:  ptr.To[int32](5),
		},
		"gang unset minCount with nil replicas clamps to 1": {
			replicas: nil,
			cfg:      gangConfig(nil),
			wantMin:  ptr.To[int32](1),
		},
		"gang unset minCount with zero replicas clamps to 1": {
			replicas: ptr.To[int32](0),
			cfg:      gangConfig(nil),
			wantMin:  ptr.To[int32](1),
		},
		"gang minCount already set is preserved": {
			replicas: ptr.To[int32](5),
			cfg:      gangConfig(ptr.To[int32](2)),
			wantMin:  ptr.To[int32](2),
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			item := WorkloadItemForDeployment("deployment", "", tc.replicas, WorkloadInput(nil, nil, nil, nil), nil)
			item.Callbacks[0](tc.cfg)
			if diff := cmp.Diff(tc.wantMin, tc.cfg.Policy.Gang.MinCount); diff != "" {
				t.Errorf("unexpected resolved MinCount (-want,+got):\n%s", diff)
			}
		})
	}

	t.Run("basic policy is left untouched", func(t *testing.T) {
		item := WorkloadItemForDeployment("deployment", "", ptr.To[int32](5), WorkloadInput(nil, nil, nil, nil), nil)
		cfg := &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{Basic: &workloadbuilder.BasicSchedulingPolicy{}},
		}
		item.Callbacks[0](cfg)
		if cfg.Policy.Gang != nil {
			t.Errorf("Gang = %+v, want nil for a basic policy", cfg.Policy.Gang)
		}
	})

	t.Run("nil config does not panic", func(t *testing.T) {
		item := WorkloadItemForDeployment("deployment", "", ptr.To[int32](5), WorkloadInput(nil, nil, nil, nil), nil)
		item.Callbacks[0](nil)
	})
}
