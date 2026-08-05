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

package job

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestDropDisabledFieldsScheduling(t *testing.T) {
	schedulingConfig := func() *batch.JobSchedulingConfiguration {
		return &batch.JobSchedulingConfiguration{SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}}}
	}

	cases := map[string]struct {
		enableWorkloadWithJob bool
		jobSpec               *batch.JobSchedulingConfiguration
		oldJobSpec            *batch.JobSchedulingConfiguration
		wantDropped           bool
		// hasOldSpec distinguishes create (nil old spec) from update.
		hasOldSpec bool
	}{
		"feature enabled, create keeps scheduling": {
			enableWorkloadWithJob: true,
			jobSpec:               schedulingConfig(),
		},
		"feature disabled, create drops scheduling": {
			enableWorkloadWithJob: false,
			jobSpec:               schedulingConfig(),
			wantDropped:           true,
		},
		"feature disabled, create without scheduling stays nil": {
			enableWorkloadWithJob: false,
			jobSpec:               nil,
		},
		"feature enabled, update keeps scheduling": {
			enableWorkloadWithJob: true,
			jobSpec:               schedulingConfig(),
			oldJobSpec:            schedulingConfig(),
			hasOldSpec:            true,
		},
		"feature disabled, update preserves scheduling already in use": {
			enableWorkloadWithJob: false,
			jobSpec:               schedulingConfig(),
			oldJobSpec:            schedulingConfig(),
			hasOldSpec:            true,
		},
		"feature disabled, update drops newly added scheduling": {
			enableWorkloadWithJob: false,
			jobSpec:               schedulingConfig(),
			oldJobSpec:            nil,
			hasOldSpec:            true,
			wantDropped:           true,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			// WorkloadWithJob depends on GenericWorkload, so toggle them together.
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: tc.enableWorkloadWithJob,
				features.WorkloadWithJob: tc.enableWorkloadWithJob,
			})

			newSpec := &batch.JobSpec{Scheduling: tc.jobSpec}
			var oldSpec *batch.JobSpec
			if tc.hasOldSpec {
				oldSpec = &batch.JobSpec{Scheduling: tc.oldJobSpec}
			}

			DropDisabledFields(newSpec, oldSpec)

			var want *batch.JobSchedulingConfiguration
			if !tc.wantDropped {
				want = tc.jobSpec
			}
			if diff := cmp.Diff(want, newSpec.Scheduling); diff != "" {
				t.Errorf("unexpected scheduling after DropDisabledFields (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestWorkloadInput(t *testing.T) {
	policy := &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
		Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](3)},
	}
	constraints := &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
		Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/rack"}},
	}
	disruption := &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
		All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
	}
	claims := []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
		{Name: "c1", ResourceClaimName: new("rc1")},
	}

	cases := map[string]struct {
		policy         *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
		constraints    *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints
		disruptionMode *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
		resourceClaims []schedulingv1alpha3.WorkloadPodGroupResourceClaim
		want           workloadbuilder.WorkloadInput
	}{
		"nil blocks still carry path elements": {
			want: workloadbuilder.WorkloadInput{
				Policy:         workloadbuilder.PolicyInput{PathElements: []string{"schedulingPolicy"}},
				Constraints:    workloadbuilder.ConstraintsInput{PathElements: []string{"schedulingConstraints"}},
				DisruptionMode: workloadbuilder.DisruptionModeInput{PathElements: []string{"disruptionMode"}},
				ResourceClaims: workloadbuilder.ResourceClaimsInput{PathElements: []string{"resourceClaims"}},
			},
		},
		"populated blocks pass through with path elements": {
			policy:         policy,
			constraints:    constraints,
			disruptionMode: disruption,
			resourceClaims: claims,
			want: workloadbuilder.WorkloadInput{
				Policy:         workloadbuilder.PolicyInput{PodGroupData: policy, PathElements: []string{"schedulingPolicy"}},
				Constraints:    workloadbuilder.ConstraintsInput{PodGroupData: constraints, PathElements: []string{"schedulingConstraints"}},
				DisruptionMode: workloadbuilder.DisruptionModeInput{PodGroupData: disruption, PathElements: []string{"disruptionMode"}},
				ResourceClaims: workloadbuilder.ResourceClaimsInput{PodGroupData: claims, PathElements: []string{"resourceClaims"}},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := WorkloadInput(tc.policy, tc.constraints, tc.disruptionMode, tc.resourceClaims)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("unexpected WorkloadInput (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestWorkloadItemForJob(t *testing.T) {
	input := WorkloadInput(nil, nil, nil, nil)

	cases := map[string]struct {
		itemName          string
		priorityClassName string
		parallelism       *int32
		input             workloadbuilder.WorkloadInput
		path              *field.Path
	}{
		"name and priority class carried onto the item": {
			itemName:          "job-pgt-0",
			priorityClassName: "high",
			parallelism:       ptr.To[int32](4),
			input:             input,
			path:              field.NewPath("spec", "scheduling"),
		},
		"empty priority class and nil parallelism": {
			itemName: "job",
			input:    input,
		},
		"nil path is carried through": {
			itemName: "job",
			input:    input,
			path:     nil,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := WorkloadItemForJob(tc.itemName, tc.priorityClassName, tc.parallelism, tc.input, tc.path)

			if got.Name != tc.itemName {
				t.Errorf("Name = %q, want %q", got.Name, tc.itemName)
			}
			if got.Path.String() != tc.path.String() {
				t.Errorf("Path = %q, want %q", got.Path.String(), tc.path.String())
			}
			wantDefault := &workloadbuilder.SchedulingConfig{
				Policy:            &workloadbuilder.SchedulingPolicy{Basic: &workloadbuilder.BasicSchedulingPolicy{}},
				PriorityClassName: tc.priorityClassName,
			}
			if diff := cmp.Diff(wantDefault, got.DefaultConfig); diff != "" {
				t.Errorf("unexpected DefaultConfig (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.input, got.Input); diff != "" {
				t.Errorf("unexpected Input (-want,+got):\n%s", diff)
			}
			if len(got.Callbacks) != 1 {
				t.Fatalf("Callbacks length = %d, want 1", len(got.Callbacks))
			}
		})
	}
}

func TestWorkloadItemForJobMinCountDefaulting(t *testing.T) {
	gangConfig := func(minCount *int32) *workloadbuilder.SchedulingConfig {
		return &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{Gang: &workloadbuilder.GangSchedulingPolicy{MinCount: minCount}},
		}
	}

	cases := map[string]struct {
		parallelism *int32
		cfg         *workloadbuilder.SchedulingConfig
		wantMin     *int32
	}{
		"gang unset minCount defaults to parallelism": {
			parallelism: ptr.To[int32](5),
			cfg:         gangConfig(nil),
			wantMin:     ptr.To[int32](5),
		},
		"gang unset minCount with nil parallelism clamps to 1": {
			parallelism: nil,
			cfg:         gangConfig(nil),
			wantMin:     ptr.To[int32](1),
		},
		"gang unset minCount with zero parallelism clamps to 1": {
			parallelism: ptr.To[int32](0),
			cfg:         gangConfig(nil),
			wantMin:     ptr.To[int32](1),
		},
		"gang minCount already set is preserved": {
			parallelism: ptr.To[int32](5),
			cfg:         gangConfig(ptr.To[int32](2)),
			wantMin:     ptr.To[int32](2),
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			item := WorkloadItemForJob("job", "", tc.parallelism, WorkloadInput(nil, nil, nil, nil), nil)
			item.Callbacks[0](tc.cfg)
			if diff := cmp.Diff(tc.wantMin, tc.cfg.Policy.Gang.MinCount); diff != "" {
				t.Errorf("unexpected resolved MinCount (-want,+got):\n%s", diff)
			}
		})
	}

	t.Run("basic policy is left untouched", func(t *testing.T) {
		item := WorkloadItemForJob("job", "", ptr.To[int32](5), WorkloadInput(nil, nil, nil, nil), nil)
		cfg := &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{Basic: &workloadbuilder.BasicSchedulingPolicy{}},
		}
		item.Callbacks[0](cfg)
		if cfg.Policy.Gang != nil {
			t.Errorf("Gang = %+v, want nil for a basic policy", cfg.Policy.Gang)
		}
	})

	t.Run("nil config does not panic", func(t *testing.T) {
		item := WorkloadItemForJob("job", "", ptr.To[int32](5), WorkloadInput(nil, nil, nil, nil), nil)
		item.Callbacks[0](nil)
	})
}
