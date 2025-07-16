/*
Copyright 2020 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

var (
	ignoreBadValueDetail = cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail")
)

func TestValidateDefaultPreemptionArgs(t *testing.T) {
	cases := map[string]struct {
		args     config.DefaultPreemptionArgs
		wantErrs field.ErrorList
	}{
		"valid args (default)": {
			args: config.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: 10,
				MinCandidateNodesAbsolute:   100,
			},
		},
		"negative minCandidateNodesPercentage": {
			args: config.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: -1,
				MinCandidateNodesAbsolute:   100,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "minCandidateNodesPercentage",
				},
			},
		},
		"minCandidateNodesPercentage over 100": {
			args: config.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: 900,
				MinCandidateNodesAbsolute:   100,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "minCandidateNodesPercentage",
				},
			},
		},
		"negative minCandidateNodesAbsolute": {
			args: config.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: 20,
				MinCandidateNodesAbsolute:   -1,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "minCandidateNodesAbsolute",
				},
			},
		},
		"all zero": {
			args: config.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: 0,
				MinCandidateNodesAbsolute:   0,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "minCandidateNodesPercentage",
				}, &field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "minCandidateNodesAbsolute",
				},
			},
		},
		"both negative": {
			args: config.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: -1,
				MinCandidateNodesAbsolute:   -1,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "minCandidateNodesPercentage",
				}, &field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "minCandidateNodesAbsolute",
				},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateDefaultPreemptionArgs(nil, &tc.args)
			if diff := cmp.Diff(tc.wantErrs.ToAggregate(), err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidateDefaultPreemptionArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateInterPodAffinityArgs(t *testing.T) {
	cases := map[string]struct {
		args    config.InterPodAffinityArgs
		wantErr error
	}{
		"valid args": {
			args: config.InterPodAffinityArgs{
				HardPodAffinityWeight: 10,
			},
		},
		"hardPodAffinityWeight less than min": {
			args: config.InterPodAffinityArgs{
				HardPodAffinityWeight: -1,
			},
			wantErr: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "hardPodAffinityWeight",
			},
		},
		"hardPodAffinityWeight more than max": {
			args: config.InterPodAffinityArgs{
				HardPodAffinityWeight: 101,
			},
			wantErr: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "hardPodAffinityWeight",
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateInterPodAffinityArgs(nil, &tc.args)
			if diff := cmp.Diff(tc.wantErr, err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidateInterPodAffinityArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidatePodTopologySpreadArgs(t *testing.T) {
	cases := map[string]struct {
		args     *config.PodTopologySpreadArgs
		wantErrs field.ErrorList
	}{
		"valid config": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.DoNotSchedule,
					},
					{
						MaxSkew:           2,
						TopologyKey:       "zone",
						WhenUnsatisfiable: v1.ScheduleAnyway,
					},
				},
				DefaultingType: config.ListDefaulting,
			},
		},
		"maxSkew less than zero": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           -1,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.DoNotSchedule,
					},
				},
				DefaultingType: config.ListDefaulting,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "defaultConstraints[0].maxSkew",
				},
			},
		},
		"empty topology key": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "",
						WhenUnsatisfiable: v1.DoNotSchedule,
					},
				},
				DefaultingType: config.ListDefaulting,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeRequired,
					Field: "defaultConstraints[0].topologyKey",
				},
			},
		},
		"whenUnsatisfiable is empty": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: "",
					},
				},
				DefaultingType: config.ListDefaulting,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeRequired,
					Field: "defaultConstraints[0].whenUnsatisfiable",
				},
			},
		},
		"whenUnsatisfiable contains unsupported action": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: "unknown action",
					},
				},
				DefaultingType: config.ListDefaulting,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeNotSupported,
					Field: "defaultConstraints[0].whenUnsatisfiable",
				},
			},
		},
		"duplicated constraints": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.DoNotSchedule,
					},
					{
						MaxSkew:           2,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.DoNotSchedule,
					},
				},
				DefaultingType: config.ListDefaulting,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeDuplicate,
					Field: "defaultConstraints[1]",
				},
			},
		},
		"label selector present": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "key",
						WhenUnsatisfiable: v1.DoNotSchedule,
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"a": "b",
							},
						},
					},
				},
				DefaultingType: config.ListDefaulting,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeForbidden,
					Field: "defaultConstraints[0].labelSelector",
				},
			},
		},
		"list default constraints, no constraints": {
			args: &config.PodTopologySpreadArgs{
				DefaultingType: config.ListDefaulting,
			},
		},
		"system default constraints": {
			args: &config.PodTopologySpreadArgs{
				DefaultingType: config.SystemDefaulting,
			},
		},
		"wrong constraints": {
			args: &config.PodTopologySpreadArgs{
				DefaultingType: "unknown",
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeNotSupported,
					Field: "defaultingType",
				},
			},
		},
		"system default constraints, but has constraints": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "key",
						WhenUnsatisfiable: v1.DoNotSchedule,
					},
				},
				DefaultingType: config.SystemDefaulting,
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "defaultingType",
				},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidatePodTopologySpreadArgs(nil, tc.args)
			if diff := cmp.Diff(tc.wantErrs.ToAggregate(), err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidatePodTopologySpreadArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateNodeResourcesBalancedAllocationArgs(t *testing.T) {
	cases := map[string]struct {
		args     *config.NodeResourcesBalancedAllocationArgs
		wantErrs field.ErrorList
	}{
		"valid config": {
			args: &config.NodeResourcesBalancedAllocationArgs{
				Resources: []config.ResourceSpec{
					{
						Name:   "cpu",
						Weight: 1,
					},
					{
						Name:   "memory",
						Weight: 1,
					},
				},
			},
		},
		"invalid config": {
			args: &config.NodeResourcesBalancedAllocationArgs{
				Resources: []config.ResourceSpec{
					{
						Name:   "cpu",
						Weight: 2,
					},
					{
						Name:   "memory",
						Weight: 1,
					},
				},
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "resources[0].weight",
				},
			},
		},
		"repeated resources": {
			args: &config.NodeResourcesBalancedAllocationArgs{
				Resources: []config.ResourceSpec{
					{
						Name:   "cpu",
						Weight: 1,
					},
					{
						Name:   "cpu",
						Weight: 1,
					},
				},
			},
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeDuplicate,
					Field: "resources[1].name",
				},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateNodeResourcesBalancedAllocationArgs(nil, tc.args)
			if diff := cmp.Diff(tc.wantErrs.ToAggregate(), err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidateNodeResourcesBalancedAllocationArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateNodeAffinityArgs(t *testing.T) {
	cases := []struct {
		name    string
		args    config.NodeAffinityArgs
		wantErr error
	}{
		{
			name: "empty",
		},
		{
			name: "valid added affinity",
			args: config.NodeAffinityArgs{
				AddedAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "label-1",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"label-1-val"},
									},
								},
							},
						},
					},
					PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
						{
							Weight: 1,
							Preference: v1.NodeSelectorTerm{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      "metadata.name",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"node-1"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "invalid added affinity",
			args: config.NodeAffinityArgs{
				AddedAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "invalid/label/key",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"label-1-val"},
									},
								},
							},
						},
					},
					PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
						{
							Weight: 1,
							Preference: v1.NodeSelectorTerm{
								MatchFields: []v1.NodeSelectorRequirement{
									{
										Key:      "metadata.name",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"node-1", "node-2"},
									},
								},
							},
						},
					},
				},
			},
			wantErr: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "addedAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key",
				},
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "addedAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].matchFields[0].values",
				},
			}.ToAggregate(),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateNodeAffinityArgs(nil, &tc.args)
			if diff := cmp.Diff(tc.wantErr, err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidatedNodeAffinityArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateVolumeBindingArgs(t *testing.T) {
	cases := []struct {
		name     string
		args     config.VolumeBindingArgs
		features map[featuregate.Feature]bool
		wantErr  error
	}{
		{
			name: "zero is a valid config",
			args: config.VolumeBindingArgs{
				BindTimeoutSeconds: 0,
			},
		},
		{
			name: "positive value is valid config",
			args: config.VolumeBindingArgs{
				BindTimeoutSeconds: 10,
			},
		},
		{
			name: "negative value is invalid config ",
			args: config.VolumeBindingArgs{
				BindTimeoutSeconds: -10,
			},
			wantErr: errors.NewAggregate([]error{&field.Error{
				Type:     field.ErrorTypeInvalid,
				Field:    "bindTimeoutSeconds",
				BadValue: int64(-10),
				Detail:   "invalid BindTimeoutSeconds, should not be a negative value",
			}}),
		},
		{
			name: "[StorageCapacityScoring=off] shape should be nil when the feature is off",
			features: map[featuregate.Feature]bool{
				features.StorageCapacityScoring: false,
			},
			args: config.VolumeBindingArgs{
				BindTimeoutSeconds: 10,
				Shape:              nil,
			},
		},
		{
			name: "[StorageCapacityScoring=off] error if the shape is not nil when the feature is off",
			features: map[featuregate.Feature]bool{
				features.StorageCapacityScoring: false,
			},
			args: config.VolumeBindingArgs{
				BindTimeoutSeconds: 10,
				Shape: []config.UtilizationShapePoint{
					{Utilization: 1, Score: 1},
					{Utilization: 3, Score: 3},
				},
			},
			wantErr: errors.NewAggregate([]error{&field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "shape",
			}}),
		},
		{
			name: "[StorageCapacityScoring=on] shape should not be empty",
			features: map[featuregate.Feature]bool{
				features.StorageCapacityScoring: true,
			},
			args: config.VolumeBindingArgs{
				BindTimeoutSeconds: 10,
				Shape:              []config.UtilizationShapePoint{},
			},
			wantErr: errors.NewAggregate([]error{&field.Error{
				Type:  field.ErrorTypeRequired,
				Field: "shape",
			}}),
		},
		{
			name: "[StorageCapacityScoring=on] shape points must be sorted in increasing order",
			features: map[featuregate.Feature]bool{
				features.StorageCapacityScoring: true,
			},
			args: config.VolumeBindingArgs{
				BindTimeoutSeconds: 10,
				Shape: []config.UtilizationShapePoint{
					{Utilization: 3, Score: 3},
					{Utilization: 1, Score: 1},
				},
			},
			wantErr: errors.NewAggregate([]error{&field.Error{
				Type:   field.ErrorTypeInvalid,
				Field:  "shape[1].utilization",
				Detail: "Invalid value: 1: utilization values must be sorted in increasing order",
			}}),
		},
		{
			name: "[StorageCapacityScoring=on] shape point: invalid utilization and score",
			features: map[featuregate.Feature]bool{
				features.StorageCapacityScoring: true,
			},
			args: config.VolumeBindingArgs{
				BindTimeoutSeconds: 10,
				Shape: []config.UtilizationShapePoint{
					{Utilization: -1, Score: 1},
					{Utilization: 10, Score: -1},
					{Utilization: 20, Score: 11},
					{Utilization: 101, Score: 1},
				},
			},
			wantErr: errors.NewAggregate([]error{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "shape[0].utilization",
				},
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "shape[1].score",
				},
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "shape[2].score",
				},
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "shape[3].utilization",
				},
			}),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for k, v := range tc.features {
				featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, k, v)
			}
			err := ValidateVolumeBindingArgs(nil, &tc.args)
			if diff := cmp.Diff(tc.wantErr, err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidateVolumeBindingArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateFitArgs(t *testing.T) {
	defaultScoringStrategy := &config.ScoringStrategy{
		Type: config.LeastAllocated,
		Resources: []config.ResourceSpec{
			{Name: "cpu", Weight: 1},
			{Name: "memory", Weight: 1},
		},
	}
	argsTest := []struct {
		name   string
		args   config.NodeResourcesFitArgs
		expect string
	}{
		{
			name: "IgnoredResources: too long value",
			args: config.NodeResourcesFitArgs{
				IgnoredResources: []string{fmt.Sprintf("longvalue%s", strings.Repeat("a", 64))},
				ScoringStrategy:  defaultScoringStrategy,
			},
			expect: "name part must be no more than 63 characters",
		},
		{
			name: "IgnoredResources: name is empty",
			args: config.NodeResourcesFitArgs{
				IgnoredResources: []string{"example.com/"},
				ScoringStrategy:  defaultScoringStrategy,
			},
			expect: "name part must be non-empty",
		},
		{
			name: "IgnoredResources: name has too many slash",
			args: config.NodeResourcesFitArgs{
				IgnoredResources: []string{"example.com/aaa/bbb"},
				ScoringStrategy:  defaultScoringStrategy,
			},
			expect: "a qualified name must consist of alphanumeric characters",
		},
		{
			name: "IgnoredResources: valid args",
			args: config.NodeResourcesFitArgs{
				IgnoredResources: []string{"example.com"},
				ScoringStrategy:  defaultScoringStrategy,
			},
		},
		{
			name: "IgnoredResourceGroups: valid args ",
			args: config.NodeResourcesFitArgs{
				IgnoredResourceGroups: []string{"example.com"},
				ScoringStrategy:       defaultScoringStrategy,
			},
		},
		{
			name: "IgnoredResourceGroups: illegal args",
			args: config.NodeResourcesFitArgs{
				IgnoredResourceGroups: []string{"example.com/"},
				ScoringStrategy:       defaultScoringStrategy,
			},
			expect: "name part must be non-empty",
		},
		{
			name: "IgnoredResourceGroups: name is too long",
			args: config.NodeResourcesFitArgs{
				IgnoredResourceGroups: []string{strings.Repeat("a", 64)},
				ScoringStrategy:       defaultScoringStrategy,
			},
			expect: "name part must be no more than 63 characters",
		},
		{
			name: "IgnoredResourceGroups: name cannot be contain slash",
			args: config.NodeResourcesFitArgs{
				IgnoredResourceGroups: []string{"example.com/aa"},
				ScoringStrategy:       defaultScoringStrategy,
			},
			expect: "resource group name can't contain '/'",
		},
		{
			name:   "ScoringStrategy: field is required",
			args:   config.NodeResourcesFitArgs{},
			expect: "ScoringStrategy field is required",
		},
		{
			name: "ScoringStrategy: type is unsupported",
			args: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type: "Invalid",
				},
			},
			expect: `Unsupported value: "Invalid"`,
		},
	}

	for _, test := range argsTest {
		t.Run(test.name, func(t *testing.T) {
			if err := ValidateNodeResourcesFitArgs(nil, &test.args); err != nil && (!strings.Contains(err.Error(), test.expect)) {
				t.Errorf("case[%v]: error details do not include %v", test.name, err)
			}
		})
	}
}

func TestValidateLeastAllocatedScoringStrategy(t *testing.T) {
	tests := []struct {
		name      string
		resources []config.ResourceSpec
		wantErrs  field.ErrorList
	}{
		{
			name:     "default config",
			wantErrs: nil,
		},
		{
			name: "multi valid resources",
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 1,
				},
				{
					Name:   "memory",
					Weight: 10,
				},
			},
			wantErrs: nil,
		},
		{
			name: "weight less than min",
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 0,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[0].weight",
				},
			},
		},
		{
			name: "weight greater than max",
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 101,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[0].weight",
				},
			},
		},
		{
			name: "multi invalid resources",
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 0,
				},
				{
					Name:   "memory",
					Weight: 101,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[0].weight",
				},
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[1].weight",
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			args := config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.LeastAllocated,
					Resources: test.resources,
				},
			}
			err := ValidateNodeResourcesFitArgs(nil, &args)
			if diff := cmp.Diff(test.wantErrs.ToAggregate(), err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidateNodeResourcesFitArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateMostAllocatedScoringStrategy(t *testing.T) {
	tests := []struct {
		name      string
		resources []config.ResourceSpec
		wantErrs  field.ErrorList
	}{
		{
			name:     "default config",
			wantErrs: nil,
		},
		{
			name: "multi valid resources",
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 1,
				},
				{
					Name:   "memory",
					Weight: 10,
				},
			},
			wantErrs: nil,
		},
		{
			name: "weight less than min",
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 0,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[0].weight",
				},
			},
		},
		{
			name: "weight greater than max",
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 101,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[0].weight",
				},
			},
		},
		{
			name: "multi invalid resources",
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 0,
				},
				{
					Name:   "memory",
					Weight: 101,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[0].weight",
				},
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[1].weight",
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			args := config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.MostAllocated,
					Resources: test.resources,
				},
			}
			err := ValidateNodeResourcesFitArgs(nil, &args)
			if diff := cmp.Diff(test.wantErrs.ToAggregate(), err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidateNodeResourcesFitArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateRequestedToCapacityRatioScoringStrategy(t *testing.T) {
	defaultShape := []config.UtilizationShapePoint{
		{
			Utilization: 30,
			Score:       3,
		},
	}
	tests := []struct {
		name      string
		resources []config.ResourceSpec
		shapes    []config.UtilizationShapePoint
		wantErrs  field.ErrorList
	}{
		{
			name:   "no shapes",
			shapes: nil,
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeRequired,
					Field: "scoringStrategy.shape",
				},
			},
		},
		{
			name:   "weight greater than max",
			shapes: defaultShape,
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 101,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[0].weight",
				},
			},
		},
		{
			name:   "weight less than min",
			shapes: defaultShape,
			resources: []config.ResourceSpec{
				{
					Name:   "cpu",
					Weight: 0,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.resources[0].weight",
				},
			},
		},
		{
			name:     "valid shapes",
			shapes:   defaultShape,
			wantErrs: nil,
		},
		{
			name: "utilization less than min",
			shapes: []config.UtilizationShapePoint{
				{
					Utilization: -1,
					Score:       3,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.shape[0].utilization",
				},
			},
		},
		{
			name: "utilization greater than max",
			shapes: []config.UtilizationShapePoint{
				{
					Utilization: 101,
					Score:       3,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.shape[0].utilization",
				},
			},
		},
		{
			name: "duplicated utilization values",
			shapes: []config.UtilizationShapePoint{
				{
					Utilization: 10,
					Score:       3,
				},
				{
					Utilization: 10,
					Score:       3,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.shape[1].utilization",
				},
			},
		},
		{
			name: "increasing utilization values",
			shapes: []config.UtilizationShapePoint{
				{
					Utilization: 10,
					Score:       3,
				},
				{
					Utilization: 20,
					Score:       3,
				},
				{
					Utilization: 30,
					Score:       3,
				},
			},
			wantErrs: nil,
		},
		{
			name: "non-increasing utilization values",
			shapes: []config.UtilizationShapePoint{
				{
					Utilization: 10,
					Score:       3,
				},
				{
					Utilization: 20,
					Score:       3,
				},
				{
					Utilization: 15,
					Score:       3,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.shape[2].utilization",
				},
			},
		},
		{
			name: "score less than min",
			shapes: []config.UtilizationShapePoint{
				{
					Utilization: 10,
					Score:       -1,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.shape[0].score",
				},
			},
		},
		{
			name: "score greater than max",
			shapes: []config.UtilizationShapePoint{
				{
					Utilization: 10,
					Score:       11,
				},
			},
			wantErrs: field.ErrorList{
				{
					Type:  field.ErrorTypeInvalid,
					Field: "scoringStrategy.shape[0].score",
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			args := config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.RequestedToCapacityRatio,
					Resources: test.resources,
					RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
						Shape: test.shapes,
					},
				},
			}
			err := ValidateNodeResourcesFitArgs(nil, &args)
			if diff := cmp.Diff(test.wantErrs.ToAggregate(), err, ignoreBadValueDetail); diff != "" {
				t.Errorf("ValidateNodeResourcesFitArgs returned err (-want,+got):\n%s", diff)
			}
		})
	}
}
