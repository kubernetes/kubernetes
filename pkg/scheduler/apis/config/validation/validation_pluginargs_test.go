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
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func TestValidateInterPodAffinityArgs(t *testing.T) {
	cases := map[string]struct {
		args    config.InterPodAffinityArgs
		wantErr string
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
			wantErr: `hardPodAffinityWeight: Invalid value: -1: not in valid range [0-100]`,
		},
		"hardPodAffinityWeight more than max": {
			args: config.InterPodAffinityArgs{
				HardPodAffinityWeight: 101,
			},
			wantErr: `hardPodAffinityWeight: Invalid value: 101: not in valid range [0-100]`,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateInterPodAffinityArgs(tc.args)
			assertErr(t, tc.wantErr, err)
		})
	}
}

func TestValidateNodeLabelArgs(t *testing.T) {
	cases := map[string]struct {
		args    config.NodeLabelArgs
		wantErr string
	}{
		"valid config": {
			args: config.NodeLabelArgs{
				PresentLabels:           []string{"present"},
				AbsentLabels:            []string{"absent"},
				PresentLabelsPreference: []string{"present-preference"},
				AbsentLabelsPreference:  []string{"absent-preference"},
			},
		},
		"labels conflict": {
			args: config.NodeLabelArgs{
				PresentLabels: []string{"label"},
				AbsentLabels:  []string{"label"},
			},
			wantErr: `detecting at least one label (e.g., "label") that exist in both the present([label]) and absent([label]) label list`,
		},
		"labels preference conflict": {
			args: config.NodeLabelArgs{
				PresentLabelsPreference: []string{"label"},
				AbsentLabelsPreference:  []string{"label"},
			},
			wantErr: `detecting at least one label (e.g., "label") that exist in both the present([label]) and absent([label]) label list`,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateNodeLabelArgs(tc.args)
			assertErr(t, tc.wantErr, err)
		})
	}
}

func TestValidatePodTopologySpreadArgs(t *testing.T) {
	cases := map[string]struct {
		args    *config.PodTopologySpreadArgs
		wantErr string
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
			},
		},
		"max skew less than zero": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           -1,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.DoNotSchedule,
					},
				},
			},
			wantErr: `defaultConstraints[0].maxSkew: Invalid value: -1: must be greater than zero`,
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
			},
			wantErr: `defaultConstraints[0].topologyKey: Required value: can not be empty`,
		},
		"WhenUnsatisfiable is empty": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: "",
					},
				},
			},
			wantErr: `defaultConstraints[0].whenUnsatisfiable: Required value: can not be empty`,
		},
		"WhenUnsatisfiable contains unsupported action": {
			args: &config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: "unknown action",
					},
				},
			},
			wantErr: `defaultConstraints[0].whenUnsatisfiable: Unsupported value: "unknown action": supported values: "DoNotSchedule", "ScheduleAnyway"`,
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
			},
			wantErr: `defaultConstraints[1]: Duplicate value: "{node, DoNotSchedule}"`,
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
			},
			wantErr: `defaultConstraints[0].labelSelector: Forbidden: constraint must not define a selector, as they deduced for each pod`,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidatePodTopologySpreadArgs(tc.args)
			assertErr(t, tc.wantErr, err)
		})
	}
}

func assertErr(t *testing.T, wantErr string, gotErr error) {
	if wantErr == "" {
		if gotErr != nil {
			t.Fatalf("wanted err to be: 'nil', got: '%s'", gotErr.Error())
		}
	} else {
		if gotErr == nil {
			t.Fatalf("wanted err to be: '%s', got: nil", wantErr)
		}
		if gotErr.Error() != wantErr {
			t.Errorf("wanted err to be: '%s', got '%s'", wantErr, gotErr.Error())
		}
	}
}
