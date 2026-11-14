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

package validation

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestValidatePodDeletionCostOption(t *testing.T) {
	testCases := []struct {
		name                            string
		oldPodMeta                      *metav1.ObjectMeta
		featureEnabled                  bool
		wantAllowInvalidPodDeletionCost bool
	}{
		{
			name:                            "CreateFeatureEnabled",
			featureEnabled:                  true,
			wantAllowInvalidPodDeletionCost: false,
		},
		{
			name:                            "CreateFeatureDisabled",
			featureEnabled:                  false,
			wantAllowInvalidPodDeletionCost: true,
		},
		{
			name:                            "UpdateFeatureDisabled",
			oldPodMeta:                      &metav1.ObjectMeta{Annotations: map[string]string{core.PodDeletionCost: "100"}},
			featureEnabled:                  false,
			wantAllowInvalidPodDeletionCost: true,
		},
		{
			name:                            "UpdateFeatureEnabledValidOldValue",
			oldPodMeta:                      &metav1.ObjectMeta{Annotations: map[string]string{core.PodDeletionCost: "100"}},
			featureEnabled:                  true,
			wantAllowInvalidPodDeletionCost: false,
		},
		{
			name:                            "UpdateFeatureEnabledValidOldValue",
			oldPodMeta:                      &metav1.ObjectMeta{Annotations: map[string]string{core.PodDeletionCost: "invalid-value"}},
			featureEnabled:                  true,
			wantAllowInvalidPodDeletionCost: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDeletionCost, tc.featureEnabled)
			// The new pod doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(nil, nil, nil, tc.oldPodMeta)
			if tc.wantAllowInvalidPodDeletionCost != gotOptions.AllowInvalidPodDeletionCost {
				t.Errorf("unexpected diff, want: %v, got: %v", tc.wantAllowInvalidPodDeletionCost, gotOptions.AllowInvalidPodDeletionCost)
			}
		})
	}
}

func TestValidateTopologySpreadConstraintLabelSelectorOption(t *testing.T) {
	testCases := []struct {
		name       string
		oldPodSpec *core.PodSpec
		wantOption bool
	}{
		{
			name:       "Create",
			wantOption: false,
		},
		{
			name: "UpdateInvalidLabelSelector",
			oldPodSpec: &core.PodSpec{
				TopologySpreadConstraints: []core.TopologySpreadConstraint{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "foo"},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "UpdateValidLabelSelector",
			oldPodSpec: &core.PodSpec{
				TopologySpreadConstraints: []core.TopologySpreadConstraint{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"foo": "foo"},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "UpdateEmptyLabelSelector",
			oldPodSpec: &core.PodSpec{
				TopologySpreadConstraints: []core.TopologySpreadConstraint{
					{
						LabelSelector: nil,
					},
				},
			},
			wantOption: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Pod meta doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&core.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowInvalidTopologySpreadConstraintLabelSelector {
				t.Errorf("Got AllowInvalidLabelValueInSelector=%t, want %t", gotOptions.AllowInvalidTopologySpreadConstraintLabelSelector, tc.wantOption)
			}
		})
	}
}

func TestOldPodViolatesMatchLabelKeysValidationOption(t *testing.T) {
	testCases := []struct {
		name                               string
		oldPodSpec                         *core.PodSpec
		matchLabelKeysEnabled              bool
		matchLabelKeysSelectorMergeEnabled bool
		wantOption                         bool
	}{
		{
			name:                               "Create",
			oldPodSpec:                         &core.PodSpec{},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: true,
			wantOption:                         false,
		},
		{
			name: "UpdateInvalidMatchLabelKeys",
			oldPodSpec: &core.PodSpec{
				TopologySpreadConstraints: []core.TopologySpreadConstraint{{
					MatchLabelKeys: []string{"foo"},
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{"value1"},
							},
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{"value2", "value3"},
							},
						},
					},
				}},
			},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: true,
			wantOption:                         true,
		},
		{
			name: "UpdateValidMatchLabelKeys",
			oldPodSpec: &core.PodSpec{
				TopologySpreadConstraints: []core.TopologySpreadConstraint{{
					MatchLabelKeys: []string{"foo"},
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{"value1"},
							},
						},
					},
				}},
			},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: true,
			wantOption:                         false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.MatchLabelKeysInPodTopologySpread:              tc.matchLabelKeysEnabled,
				features.MatchLabelKeysInPodTopologySpreadSelectorMerge: tc.matchLabelKeysSelectorMergeEnabled,
			})
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&core.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.OldPodViolatesMatchLabelKeysValidation {
				t.Errorf("Got OldPodViolatesMatchLabelKeysValidation=%t, want %t", gotOptions.OldPodViolatesMatchLabelKeysValidation, tc.wantOption)
			}
		})
	}
}

func TestOldPodViolatesLegacyMatchLabelKeysValidationOption(t *testing.T) {
	testCases := []struct {
		name                               string
		oldPodSpec                         *core.PodSpec
		matchLabelKeysEnabled              bool
		matchLabelKeysSelectorMergeEnabled bool
		wantOption                         bool
	}{
		{
			name:                               "Create",
			oldPodSpec:                         &core.PodSpec{},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: false,
			wantOption:                         false,
		},
		{
			name: "UpdateInvalidMatchLabelKeys",
			oldPodSpec: &core.PodSpec{
				TopologySpreadConstraints: []core.TopologySpreadConstraint{{
					MatchLabelKeys: []string{"foo"},
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{"value1"},
							},
						},
					},
				}},
			},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: false,
			wantOption:                         true,
		},
		{
			name: "UpdateValidMatchLabelKeys",
			oldPodSpec: &core.PodSpec{
				TopologySpreadConstraints: []core.TopologySpreadConstraint{{
					MatchLabelKeys: []string{"foo"},
					LabelSelector:  &metav1.LabelSelector{},
				}},
			},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: false,
			wantOption:                         false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.MatchLabelKeysInPodTopologySpread:              tc.matchLabelKeysEnabled,
				features.MatchLabelKeysInPodTopologySpreadSelectorMerge: tc.matchLabelKeysSelectorMergeEnabled,
			})
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&core.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.OldPodViolatesLegacyMatchLabelKeysValidation {
				t.Errorf("Got OldPodViolatesLegacyMatchLabelKeysValidation=%t, want %t", gotOptions.OldPodViolatesLegacyMatchLabelKeysValidation, tc.wantOption)
			}
		})
	}
}

func TestValidateAllowNonLocalProjectedTokenPathOption(t *testing.T) {
	testCases := []struct {
		name       string
		oldPodSpec *core.PodSpec
		wantOption bool
	}{
		{
			name:       "Create",
			wantOption: false,
		},
		{
			name: "UpdateInvalidProjectedTokenPath",
			oldPodSpec: &core.PodSpec{
				Volumes: []core.Volume{
					{
						Name: "foo",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{
									{
										ServiceAccountToken: &core.ServiceAccountTokenProjection{
											Path: "../foo",
										},
									},
								},
							},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "UpdateValidProjectedTokenPath",
			oldPodSpec: &core.PodSpec{
				Volumes: []core.Volume{
					{
						Name: "foo",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{
									{
										ServiceAccountToken: &core.ServiceAccountTokenProjection{
											Path: "foo",
										},
									},
								},
							},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "UpdateEmptyProjectedTokenPath",
			oldPodSpec: &core.PodSpec{
				Volumes: []core.Volume{
					{
						Name: "foo",
						VolumeSource: core.VolumeSource{
							Projected: nil,
							HostPath:  &core.HostPathVolumeSource{Path: "foo"},
						},
					},
				},
			},
			wantOption: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Pod meta doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&core.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowNonLocalProjectedTokenPath {
				t.Errorf("Got AllowNonLocalProjectedTokenPath=%t, want %t", gotOptions.AllowNonLocalProjectedTokenPath, tc.wantOption)
			}
		})
	}
}

func TestValidateInvalidLabelValueInNodeSelectorOption(t *testing.T) {
	testCases := []struct {
		name       string
		oldPodSpec *core.PodSpec
		wantOption bool
	}{
		{
			name:       "Create",
			wantOption: false,
		},
		{
			name: "UpdateInvalidLabelSelector",
			oldPodSpec: &core.PodSpec{
				Affinity: &core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{{
								MatchExpressions: []core.NodeSelectorRequirement{{
									Key:      "foo",
									Operator: core.NodeSelectorOpIn,
									Values:   []string{"-1"},
								}},
							}},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "UpdateValidLabelSelector",
			oldPodSpec: &core.PodSpec{
				Affinity: &core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{{
								MatchExpressions: []core.NodeSelectorRequirement{{
									Key:      "foo",
									Operator: core.NodeSelectorOpIn,
									Values:   []string{"bar"},
								}},
							}},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "UpdateEmptyLabelSelector",
			oldPodSpec: &core.PodSpec{
				Affinity: &core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{},
						},
					},
				},
			},
			wantOption: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Pod meta doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&core.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowInvalidLabelValueInRequiredNodeAffinity {
				t.Errorf("Got AllowInvalidLabelValueInRequiredNodeAffinity=%t, want %t", gotOptions.AllowInvalidLabelValueInRequiredNodeAffinity, tc.wantOption)
			}
		})
	}
}

func TestValidateAllowIndivisibleHugePagesValuesOption(t *testing.T) {
	// 2^64 bytes overflows int64, so the divisibility check rejects it; validators
	// that read it as zero accepted it, so a stored object can carry it and the
	// option has to admit it.
	hugePages := core.ResourceName(core.ResourceHugePagesPrefix + "2Mi")
	indivisible := core.ResourceList{hugePages: resource.MustParse("18446744073709551616")}

	testCases := []struct {
		name       string
		oldPodSpec *core.PodSpec
		wantOption bool
	}{
		{
			name:       "NoOldPodSpec",
			oldPodSpec: nil,
			wantOption: false,
		},
		{
			name: "DivisibleContainerLimit",
			oldPodSpec: &core.PodSpec{Containers: []core.Container{{Resources: core.ResourceRequirements{
				Limits: core.ResourceList{hugePages: resource.MustParse("4Mi")},
			}}}},
			wantOption: false,
		},
		{
			name: "ContainerLimit",
			oldPodSpec: &core.PodSpec{Containers: []core.Container{{Resources: core.ResourceRequirements{
				Limits: indivisible,
			}}}},
			wantOption: true,
		},
		{
			name:       "PodLevelLimit",
			oldPodSpec: &core.PodSpec{Resources: &core.ResourceRequirements{Limits: indivisible}},
			wantOption: true,
		},
		{
			name:       "PodLevelRequest",
			oldPodSpec: &core.PodSpec{Resources: &core.ResourceRequirements{Requests: indivisible}},
			wantOption: true,
		},
		{
			name:       "Overhead",
			oldPodSpec: &core.PodSpec{Overhead: indivisible},
			wantOption: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&core.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowIndivisibleHugePagesValues {
				t.Errorf("Got AllowIndivisibleHugePagesValues=%t, want %t", gotOptions.AllowIndivisibleHugePagesValues, tc.wantOption)
			}
		})
	}
}

// A stored pod-level hugepage value that no longer passes the divisibility
// check survives an update that leaves it unchanged, while a create rejects it.
func TestPodLevelIndivisibleHugePagesValueSurvivesUpdate(t *testing.T) {
	hugePages := core.ResourceName(core.ResourceHugePagesPrefix + "2Mi")
	pod := func(value string) *core.Pod {
		list := core.ResourceList{
			hugePages:           resource.MustParse(value),
			core.ResourceMemory: resource.MustParse("64Mi"),
		}
		return &core.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns", ResourceVersion: "1"},
			Spec: core.PodSpec{
				RestartPolicy:                 core.RestartPolicyAlways,
				DNSPolicy:                     core.DNSClusterFirst,
				TerminationGracePeriodSeconds: ptr.To[int64](30),
				Containers: []core.Container{{
					Name:                     "ctr",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: core.TerminationMessageReadFile,
				}},
				Resources: &core.ResourceRequirements{Limits: list, Requests: list},
			},
		}
	}
	createErrors := func(created *core.Pod) field.ErrorList {
		opts := GetValidationOptionsFromPodSpecAndMeta(&created.Spec, nil, &created.ObjectMeta, nil)
		opts.PodLevelResourcesEnabled = true
		return ValidatePodSpec(&created.Spec, &created.ObjectMeta, field.NewPath("spec"), opts)
	}

	// The fixture has to be valid apart from the hugepage value, or the empty
	// error list below proves nothing.
	if errs := createErrors(pod("4Mi")); len(errs) != 0 {
		t.Fatalf("the fixture is not valid on its own: %v", errs)
	}

	// 2^64 bytes: rejected by the divisibility check, present in objects stored
	// by validators that read it as zero.
	errs := createErrors(pod("18446744073709551616"))
	if len(errs) != 2 {
		t.Errorf("create: got %v, want one error for requests and one for limits", errs)
	}
	for _, err := range errs {
		if !strings.Contains(err.Error(), "not positive integer multiple") {
			t.Errorf("create: unexpected error %v", err)
		}
	}

	stored := pod("18446744073709551616")
	updated := stored.DeepCopy()
	updated.Labels = map[string]string{"touched": "yes"}
	opts := GetValidationOptionsFromPodSpecAndMeta(&updated.Spec, &stored.Spec, &updated.ObjectMeta, &stored.ObjectMeta)
	opts.PodLevelResourcesEnabled = true
	if errs := ValidatePodUpdate(updated, stored, opts); len(errs) != 0 {
		t.Errorf("update that only changes a label: %v", errs)
	}
}

func TestValidateContainerRestartRulesOption(t *testing.T) {
	policyNever := core.ContainerRestartPolicyNever
	testCases := []struct {
		name           string
		oldPodSpec     *core.PodSpec
		featureEnabled bool
		want           bool
	}{
		{
			name:           "feature enabled",
			featureEnabled: true,
			want:           true,
		},
		{
			name:           "feature disabled",
			featureEnabled: false,
			want:           false,
		},
		{
			name: "old pod spec has regular containers with restart policy",
			oldPodSpec: &core.PodSpec{
				Containers: []core.Container{{
					RestartPolicy: &policyNever,
				}},
			},
			featureEnabled: false,
			want:           true,
		},
		{
			name: "old pod spec has regular containers with restart policy rules",
			oldPodSpec: &core.PodSpec{
				Containers: []core.Container{{
					RestartPolicy: &policyNever,
					RestartPolicyRules: []core.ContainerRestartRule{{
						Action: core.ContainerRestartRuleActionRestart,
						ExitCodes: &core.ContainerRestartRuleOnExitCodes{
							Operator: core.ContainerRestartRuleOnExitCodesOpIn,
							Values:   []int32{42},
						},
					}},
				}},
			},
			featureEnabled: false,
			want:           true,
		},
		{
			name: "old pod spec has init containers with restart policy",
			oldPodSpec: &core.PodSpec{
				InitContainers: []core.Container{{
					RestartPolicy: &policyNever,
				}},
			},
			featureEnabled: false,
			want:           true,
		},
		{
			name: "old pod spec has regular containers with restart policy rules",
			oldPodSpec: &core.PodSpec{
				InitContainers: []core.Container{{
					RestartPolicy: &policyNever,
					RestartPolicyRules: []core.ContainerRestartRule{{
						Action: core.ContainerRestartRuleActionRestart,
						ExitCodes: &core.ContainerRestartRuleOnExitCodes{
							Operator: core.ContainerRestartRuleOnExitCodesOpIn,
							Values:   []int32{42},
						},
					}},
				}},
			},
			featureEnabled: false,
			want:           true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerRestartRules, tc.featureEnabled)
			// The new pod doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(nil, tc.oldPodSpec, nil, nil)
			if tc.want != gotOptions.AllowContainerRestartPolicyRules {
				t.Errorf("unexpected diff, want: %v, got: %v", tc.want, gotOptions.AllowInvalidPodDeletionCost)
			}
		})
	}
}

func TestHasUserNamespacesWithVolumeDevices(t *testing.T) {
	falseVar := false
	trueVar := true

	tests := []struct {
		name     string
		spec     *core.PodSpec
		expected bool
	}{
		{
			name: "hostUsers=nil",
			spec: &core.PodSpec{},
		}, {
			name: "hostUsers=false & no volume devices",
			spec: &core.PodSpec{
				HostUsers: &falseVar,
			},
		}, {
			name: "hostUsers=true & container volumeDevice",
			spec: &core.PodSpec{
				HostUsers: &trueVar,
				Containers: []core.Container{{
					Name: "test-container",
					VolumeDevices: []core.VolumeDevice{{
						Name:       "test-volume",
						DevicePath: "/dev/test-device",
					}},
				}},
			},
		}, {
			name:     "hostUsers=false & container volumeDevice",
			expected: true,
			spec: &core.PodSpec{
				HostUsers: &falseVar,
				Containers: []core.Container{{
					Name: "test-container",
					VolumeDevices: []core.VolumeDevice{{
						Name:       "test-volume",
						DevicePath: "/dev/test-device",
					}},
				}},
			},
		}, {

			name:     "hostUsers=false & initContainer volumeDevice",
			expected: true,
			spec: &core.PodSpec{
				HostUsers: &falseVar,
				InitContainers: []core.Container{{
					Name: "test-container",
					VolumeDevices: []core.VolumeDevice{{
						Name:       "test-volume",
						DevicePath: "/dev/test-device",
					}},
				}},
			},
		}, {
			name:     "hostUsers=false & ephemeralContainer volumeDevice",
			expected: true,
			spec: &core.PodSpec{
				HostUsers: &falseVar,
				EphemeralContainers: []core.EphemeralContainer{{
					EphemeralContainerCommon: core.EphemeralContainerCommon{
						Name: "test-container",
						VolumeDevices: []core.VolumeDevice{{
							Name:       "test-volume",
							DevicePath: "/dev/test-device",
						}}},
				}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := hasUserNamespacesWithVolumeDevices(test.spec)
			if test.expected != actual {
				t.Errorf("expected %v, got %v", test.expected, actual)
			}
		})
	}
}

func TestTaintTolerationComparisonOperatorsInUse(t *testing.T) {
	tests := []struct {
		name     string
		podSpec  *core.PodSpec
		expected bool
	}{
		{
			name:     "nil pod spec",
			podSpec:  nil,
			expected: false,
		},
		{
			name: "no tolerations",
			podSpec: &core.PodSpec{
				Containers: []core.Container{{Name: "test"}},
			},
			expected: false,
		},
		{
			name: "only Equal operator",
			podSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpEqual, Value: "value1"},
				},
			},
			expected: false,
		},
		{
			name: "only Exists operator",
			podSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpExists},
				},
			},
			expected: false,
		},
		{
			name: "Lt operator present",
			podSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpLt, Value: "100"},
				},
			},
			expected: true,
		},
		{
			name: "Gt operator present",
			podSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpGt, Value: "50"},
				},
			},
			expected: true,
		},
		{
			name: "mixed operators with Lt",
			podSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpEqual, Value: "value1"},
					{Key: "key2", Operator: core.TolerationOpLt, Value: "100"},
					{Key: "key3", Operator: core.TolerationOpExists},
				},
			},
			expected: true,
		},
		{
			name: "mixed operators with Gt",
			podSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpExists},
					{Key: "key2", Operator: core.TolerationOpGt, Value: "200"},
				},
			},
			expected: true,
		},
		{
			name: "both Lt and Gt operators",
			podSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpLt, Value: "100"},
					{Key: "key2", Operator: core.TolerationOpGt, Value: "50"},
				},
			},
			expected: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := taintTolerationComparisonOperatorsInUse(test.podSpec)
			if test.expected != actual {
				t.Errorf("expected %v, got %v", test.expected, actual)
			}
		})
	}
}

func TestAllowTaintTolerationComparisonOperators(t *testing.T) {
	tests := []struct {
		name           string
		featureEnabled bool
		oldPodSpec     *core.PodSpec
		expected       bool
	}{
		{
			name:           "feature gate enabled, nil old pod spec",
			featureEnabled: true,
			oldPodSpec:     nil,
			expected:       true,
		},
		{
			name:           "feature gate enabled, old pod spec without comparison operators",
			featureEnabled: true,
			oldPodSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpEqual, Value: "value1"},
				},
			},
			expected: true,
		},
		{
			name:           "feature gate enabled, old pod spec with Lt operator",
			featureEnabled: true,
			oldPodSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpLt, Value: "100"},
				},
			},
			expected: true,
		},
		{
			name:           "feature gate disabled, nil old pod spec",
			featureEnabled: false,
			oldPodSpec:     nil,
			expected:       false,
		},
		{
			name:           "feature gate disabled, old pod spec without comparison operators",
			featureEnabled: false,
			oldPodSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpEqual, Value: "value1"},
					{Key: "key2", Operator: core.TolerationOpExists},
				},
			},
			expected: false,
		},
		{
			name:           "feature gate disabled, old pod spec with Lt operator",
			featureEnabled: false,
			oldPodSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpLt, Value: "100"},
				},
			},
			expected: true,
		},
		{
			name:           "feature gate disabled, old pod spec with Gt operator",
			featureEnabled: false,
			oldPodSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpGt, Value: "50"},
				},
			},
			expected: true,
		},
		{
			name:           "feature gate disabled, old pod spec with mixed operators including Lt",
			featureEnabled: false,
			oldPodSpec: &core.PodSpec{
				Tolerations: []core.Toleration{
					{Key: "key1", Operator: core.TolerationOpEqual, Value: "value1"},
					{Key: "key2", Operator: core.TolerationOpLt, Value: "100"},
					{Key: "key3", Operator: core.TolerationOpExists},
				},
			},
			expected: true,
		},
		{
			name:           "feature gate disabled, empty old pod spec",
			featureEnabled: false,
			oldPodSpec:     &core.PodSpec{},
			expected:       false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TaintTolerationComparisonOperators, test.featureEnabled)
			actual := allowTaintTolerationComparisonOperators(test.oldPodSpec)
			if test.expected != actual {
				t.Errorf("expected %v, got %v", test.expected, actual)
			}
		})
	}
}

func TestValidateRestartAllContainersOption(t *testing.T) {
	policyAlways := core.ContainerRestartPolicyAlways
	testCases := []struct {
		name           string
		oldPodSpec     *core.PodSpec
		featureEnabled bool
		want           bool
	}{
		{
			name:           "feature enabled",
			featureEnabled: true,
			want:           true,
		},
		{
			name:           "feature disabled",
			featureEnabled: false,
			want:           false,
		},
		{
			name: "old pod spec has container without action",
			oldPodSpec: &core.PodSpec{
				Containers: []core.Container{{
					Name: "container",
				}},
			},
			featureEnabled: false,
			want:           false,
		},
		{
			name: "old pod spec has container with action",
			oldPodSpec: &core.PodSpec{
				Containers: []core.Container{{
					Name: "container",
					RestartPolicyRules: []core.ContainerRestartRule{{
						Action: core.ContainerRestartRuleActionRestartAllContainers,
						ExitCodes: &core.ContainerRestartRuleOnExitCodes{
							Operator: core.ContainerRestartRuleOnExitCodesOpIn,
							Values:   []int32{42},
						},
					}},
				}},
			},
			featureEnabled: false,
			want:           true,
		},
		{
			name: "old pod spec has sidecar containers with rules",
			oldPodSpec: &core.PodSpec{
				InitContainers: []core.Container{{
					RestartPolicy: &policyAlways,
					RestartPolicyRules: []core.ContainerRestartRule{{
						Action: core.ContainerRestartRuleActionRestartAllContainers,
						ExitCodes: &core.ContainerRestartRuleOnExitCodes{
							Operator: core.ContainerRestartRuleOnExitCodesOpIn,
							Values:   []int32{42},
						},
					}},
				}},
			},
			featureEnabled: false,
			want:           true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.ContainerRestartRules:                tc.featureEnabled,
				features.RestartAllContainersOnContainerExits: tc.featureEnabled,
			})
			// The new pod doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(nil, tc.oldPodSpec, nil, nil)
			if tc.want != gotOptions.AllowRestartAllContainers {
				t.Errorf("unexpected diff, want: %v, got: %v", tc.want, gotOptions.AllowRestartAllContainers)
			}
		})
	}
}

func TestHasRestartContainerForNonSidecarInitContainer(t *testing.T) {
	tests := []struct {
		name     string
		podSpec  *core.PodSpec
		expected bool
	}{
		{
			name:     "nil pod spec",
			podSpec:  nil,
			expected: false,
		},
		{
			name:     "no init containers",
			podSpec:  &core.PodSpec{InitContainers: []core.Container{}},
			expected: false,
		},
		{
			name: "regular init container without resize policy",
			podSpec: &core.PodSpec{
				InitContainers: []core.Container{
					{Name: "init-1"},
				},
			},
			expected: false,
		},
		{
			name: "sidecar (restartable) init container with RestartContainer policy",
			podSpec: &core.PodSpec{
				InitContainers: []core.Container{
					{
						Name:          "sidecar",
						RestartPolicy: ptr.To(core.ContainerRestartPolicyAlways),
						ResizePolicy: []core.ContainerResizePolicy{
							{ResourceName: core.ResourceCPU, RestartPolicy: core.RestartContainer},
						},
					},
				},
			},
			expected: false, // Should be false because it's a sidecar
		},
		{
			name: "non-sidecar init container with NotRequired policy",
			podSpec: &core.PodSpec{
				InitContainers: []core.Container{
					{
						Name: "init-1",
						ResizePolicy: []core.ContainerResizePolicy{
							{ResourceName: core.ResourceCPU, RestartPolicy: core.NotRequired},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "non-sidecar init container with RestartContainer policy",
			podSpec: &core.PodSpec{
				InitContainers: []core.Container{
					{
						Name: "init-1",
						ResizePolicy: []core.ContainerResizePolicy{
							{ResourceName: core.ResourceMemory, RestartPolicy: core.RestartContainer},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "mix of sidecar and non-sidecar with RestartContainer policy",
			podSpec: &core.PodSpec{
				InitContainers: []core.Container{
					{
						Name:          "sidecar",
						RestartPolicy: ptr.To(core.ContainerRestartPolicyAlways),
						ResizePolicy: []core.ContainerResizePolicy{
							{ResourceName: core.ResourceCPU, RestartPolicy: core.RestartContainer},
						},
					},
					{
						Name: "init-2",
						ResizePolicy: []core.ContainerResizePolicy{
							{ResourceName: core.ResourceCPU, RestartPolicy: core.RestartContainer},
						},
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasRestartContainerForNonSidecarInitContainer(tt.podSpec)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestGetValidationOptionsAllowSysAdminWhenPrivilegeEscalationFalse(t *testing.T) {
	testCases := []struct {
		name       string
		oldPodSpec *core.PodSpec
		wantOption bool
	}{
		{
			name:       "Create pod",
			oldPodSpec: nil,
			wantOption: false,
		},
		{
			name: "Update pod with CAP_SYS_ADMIN and not AllowPrivilegeEscalation",
			oldPodSpec: &core.PodSpec{
				Containers: []core.Container{
					{
						SecurityContext: &core.SecurityContext{
							AllowPrivilegeEscalation: new(false),
							Capabilities: &core.Capabilities{
								Add: []core.Capability{"CAP_SYS_ADMIN"},
							},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "Update pod with CAP_SYS_ADMIN and nil AllowPrivilegeEscalation",
			oldPodSpec: &core.PodSpec{
				Containers: []core.Container{
					{
						SecurityContext: &core.SecurityContext{
							AllowPrivilegeEscalation: nil,
							Capabilities: &core.Capabilities{
								Add: []core.Capability{"CAP_SYS_ADMIN"},
							},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "Update pod with CAP_SYS_ADMIN and true AllowPrivilegeEscalation",
			oldPodSpec: &core.PodSpec{
				Containers: []core.Container{
					{
						SecurityContext: &core.SecurityContext{
							AllowPrivilegeEscalation: new(true),
							Capabilities: &core.Capabilities{
								Add: []core.Capability{"CAP_SYS_ADMIN"},
							},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "Update pod without CAP_SYS_ADMIN",
			oldPodSpec: &core.PodSpec{
				Containers: []core.Container{
					{
						SecurityContext: &core.SecurityContext{
							AllowPrivilegeEscalation: new(false),
							Capabilities: &core.Capabilities{
								Add: []core.Capability{"CAP_NET_ADMIN"},
							},
						},
					},
				},
			},
			wantOption: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&core.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowSysAdminWhenPrivilegeEscalationFalse {
				t.Errorf("Got AllowSysAdminWhenPrivilegeEscalationFalse=%t, want %t", gotOptions.AllowSysAdminWhenPrivilegeEscalationFalse, tc.wantOption)
			}
		})
	}
}

func TestGetValidationOptionsAllowMLDSAPodCertificateKeyTypes(t *testing.T) {
	testCases := []struct {
		name        string
		oldPodSpec  *core.PodSpec
		gateEnabled bool
		wantOption  bool
	}{
		{
			name:        "Create pod with gate disabled",
			oldPodSpec:  nil,
			gateEnabled: false,
			wantOption:  false,
		},
		{
			name:        "Create pod with gate enabled",
			oldPodSpec:  nil,
			gateEnabled: true,
			wantOption:  true,
		},
		{
			name: "Update pod with gate disabled but previously uses an MLDSA44 key type in pod certificate projected volume",
			oldPodSpec: &core.PodSpec{
				Volumes: []core.Volume{
					{
						Name: "volume",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{
									{
										PodCertificate: &core.PodCertificateProjection{
											KeyType: "MLDSA44",
										},
									},
								},
							},
						},
					},
				},
			},
			gateEnabled: false,
			wantOption:  true,
		},
		{
			name: "Update pod with gate disabled but previously uses an MLDSA65 key type in pod certificate projected volume",
			oldPodSpec: &core.PodSpec{
				Volumes: []core.Volume{
					{
						Name: "volume",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{
									{
										PodCertificate: &core.PodCertificateProjection{
											KeyType: "MLDSA65",
										},
									},
								},
							},
						},
					},
				},
			},
			gateEnabled: false,
			wantOption:  true,
		},
		{
			name: "Update pod with gate disabled but previously uses an MLDSA87 key type in pod certificate projected volume",
			oldPodSpec: &core.PodSpec{
				Volumes: []core.Volume{
					{
						Name: "volume",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{
									{
										PodCertificate: &core.PodCertificateProjection{
											KeyType: "MLDSA87",
										},
									},
								},
							},
						},
					},
				},
			},
			gateEnabled: false,
			wantOption:  true,
		},
		{
			name: "Update pod with gate disabled previously uses a non-MLDSA key type in pod certificate projected volume",
			oldPodSpec: &core.PodSpec{
				Volumes: []core.Volume{
					{
						Name: "volume",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{
									{
										PodCertificate: &core.PodCertificateProjection{
											KeyType: "RSA4096",
										},
									},
								},
							},
						},
					},
				},
			},
			gateEnabled: false,
			wantOption:  false,
		},
		{
			name: "Update pod with gate enabled, previously uses a non-MLDSA key type in pod certificate projected volume",
			oldPodSpec: &core.PodSpec{
				Volumes: []core.Volume{
					{
						Name: "volume",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{
									{
										PodCertificate: &core.PodCertificateProjection{
											KeyType: "RSA4096",
										},
									},
								},
							},
						},
					},
				},
			},
			gateEnabled: true,
			wantOption:  true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodCertificateMLDSA, tc.gateEnabled)
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&core.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowMLDSAPodCertificateKeyTypes {
				t.Errorf("Got AllowMLDSAPodCertifcateKeyTypes=%t, want %t", gotOptions.AllowMLDSAPodCertificateKeyTypes, tc.wantOption)
			}
		})
	}
}

// TestSecurityProfileOCIValidationOption checks that nothing drops the oci
// field and that the option follows the gate or the old pod's use of OCI.
func TestSecurityProfileOCIValidationOption(t *testing.T) {
	ociProfile := func() *core.SeccompProfile {
		return &core.SeccompProfile{
			Type: core.SeccompProfileTypeOCI,
			OCI:  &core.SecurityProfileOCI{Ref: "registry.example.com/profile@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
		}
	}
	podWithPodOCI := func() *core.Pod {
		return &core.Pod{Spec: core.PodSpec{
			SecurityContext: &core.PodSecurityContext{SeccompProfile: ociProfile()},
			Containers:      []core.Container{{Name: "c"}},
		}}
	}
	podWithContainerOCI := func() *core.Pod {
		return &core.Pod{Spec: core.PodSpec{
			InitContainers: []core.Container{{Name: "init", SecurityContext: &core.SecurityContext{SeccompProfile: ociProfile()}}},
			Containers:     []core.Container{{Name: "c", SecurityContext: &core.SecurityContext{SeccompProfile: ociProfile()}}},
		}}
	}
	podWithoutOCI := func() *core.Pod {
		return &core.Pod{Spec: core.PodSpec{
			SecurityContext: &core.PodSecurityContext{SeccompProfile: &core.SeccompProfile{Type: core.SeccompProfileTypeRuntimeDefault}},
			Containers:      []core.Container{{Name: "c"}},
		}}
	}

	pods := map[string]struct {
		pod    func() *core.Pod
		hasOCI bool
	}{
		"pod-level OCI":       {pod: podWithPodOCI, hasOCI: true},
		"container-level OCI": {pod: podWithContainerOCI, hasOCI: true},
		"no OCI":              {pod: podWithoutOCI},
		"nil":                 {pod: func() *core.Pod { return nil }},
	}

	for _, enabled := range []bool{true, false} {
		for oldName, oldInfo := range pods {
			for newName, newInfo := range pods {
				if newName == "nil" {
					continue
				}
				t.Run(fmt.Sprintf("enabled=%v, old=%s, new=%s", enabled, oldName, newName), func(t *testing.T) {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SecurityProfileOCI, enabled)

					oldPod := oldInfo.pod()
					newPod := newInfo.pod()
					podutil.DropDisabledPodFields(newPod, oldPod)

					// The old pod is never changed.
					if oldPod != nil && !reflect.DeepEqual(oldPod, oldInfo.pod()) {
						t.Errorf("old pod changed: %v", cmp.Diff(oldInfo.pod(), oldPod))
					}

					if !reflect.DeepEqual(newPod, newInfo.pod()) {
						t.Errorf("new pod changed: %v", cmp.Diff(newInfo.pod(), newPod))
					}
					keep := enabled || oldInfo.hasOCI

					var oldSpec *core.PodSpec
					if oldPod != nil {
						oldSpec = &oldPod.Spec
					}
					opts := GetValidationOptionsFromPodSpecAndMeta(&newPod.Spec, oldSpec, nil, nil)
					if opts.AllowSecurityProfileOCI != keep {
						t.Errorf("expected AllowSecurityProfileOCI=%v, got %v", keep, opts.AllowSecurityProfileOCI)
					}
				})
			}
		}
	}
}
