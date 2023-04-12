/*
Copyright 2023 The Kubernetes Authors.

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

package admissionregistration

import (
	"testing"

	"github.com/stretchr/testify/require"

	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestDropDisabledMutatingWebhookConfigurationFields(t *testing.T) {
	tests := []struct {
		name               string
		old                *MutatingWebhookConfiguration
		new                *MutatingWebhookConfiguration
		featureGateEnabled bool
		expected           []MatchCondition
	}{
		{
			name: "create with no match conditions, feature gate off",
			old:  nil,
			new: &MutatingWebhookConfiguration{
				Webhooks: []MutatingWebhook{
					{},
				},
			},
			featureGateEnabled: false,
			expected:           nil,
		},
		{
			name: "create with match conditions, feature gate off",
			old:  nil,
			new: &MutatingWebhookConfiguration{
				Webhooks: []MutatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			featureGateEnabled: false,
			expected:           nil,
		},
		{
			name: "create with no match conditions, feature gate on",
			old:  nil,
			new: &MutatingWebhookConfiguration{
				Webhooks: []MutatingWebhook{
					{},
					{},
				},
			},
			featureGateEnabled: true,
			expected:           nil,
		},
		{
			name: "create with match conditions, feature gate on",
			old:  nil,
			new: &MutatingWebhookConfiguration{
				Webhooks: []MutatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			featureGateEnabled: true,
			expected: []MatchCondition{
				{
					Name: "test1",
				},
			},
		},
		{
			name: "update with old has match conditions feature gate on",
			old: &MutatingWebhookConfiguration{
				Webhooks: []MutatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			new: &MutatingWebhookConfiguration{
				Webhooks: []MutatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			featureGateEnabled: true,
			expected: []MatchCondition{
				{
					Name: "test1",
				},
			},
		},
		{
			name: "update with old has match conditions feature gate off",
			old: &MutatingWebhookConfiguration{
				Webhooks: []MutatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			new: &MutatingWebhookConfiguration{
				Webhooks: []MutatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			featureGateEnabled: false,
			expected: []MatchCondition{
				{
					Name: "test1",
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AdmissionWebhookMatchConditions, test.featureGateEnabled)()
			DropDisabledMutatingWebhookConfigurationFields(test.new, test.old)

			for _, hook := range test.new.Webhooks {
				if test.expected == nil {
					if hook.MatchConditions != nil {
						t.Error("expected all hooks matchConditions to be nil")
					}
				} else {
					require.Equal(t, len(test.expected), len(hook.MatchConditions))
					for i, matchCondition := range hook.MatchConditions {
						require.Equal(t, test.expected[i], matchCondition)
					}
				}
			}
		})
	}
}

func TestDropDisabledValidatingWebhookConfigurationFields(t *testing.T) {
	tests := []struct {
		name               string
		old                *ValidatingWebhookConfiguration
		new                *ValidatingWebhookConfiguration
		featureGateEnabled bool
		expected           []MatchCondition
	}{
		{
			name: "create with no match conditions, feature gate off",
			old:  nil,
			new: &ValidatingWebhookConfiguration{
				Webhooks: []ValidatingWebhook{
					{},
				},
			},
			featureGateEnabled: false,
			expected:           nil,
		},
		{
			name: "create with match conditions, feature gate off",
			old:  nil,
			new: &ValidatingWebhookConfiguration{
				Webhooks: []ValidatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			featureGateEnabled: false,
			expected:           nil,
		},
		{
			name: "create with no match conditions, feature gate on",
			old:  nil,
			new: &ValidatingWebhookConfiguration{
				Webhooks: []ValidatingWebhook{
					{},
					{},
				},
			},
			featureGateEnabled: true,
			expected:           nil,
		},
		{
			name: "create with match conditions, feature gate on",
			old:  nil,
			new: &ValidatingWebhookConfiguration{
				Webhooks: []ValidatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			featureGateEnabled: true,
			expected: []MatchCondition{
				{
					Name: "test1",
				},
			},
		},
		{
			name: "update with old has match conditions feature gate on",
			old: &ValidatingWebhookConfiguration{
				Webhooks: []ValidatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			new: &ValidatingWebhookConfiguration{
				Webhooks: []ValidatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			featureGateEnabled: true,
			expected: []MatchCondition{
				{
					Name: "test1",
				},
			},
		},
		{
			name: "update with old has match conditions feature gate off",
			old: &ValidatingWebhookConfiguration{
				Webhooks: []ValidatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			new: &ValidatingWebhookConfiguration{
				Webhooks: []ValidatingWebhook{
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
					{
						MatchConditions: []MatchCondition{
							{
								Name: "test1",
							},
						},
					},
				},
			},
			featureGateEnabled: false,
			expected: []MatchCondition{
				{
					Name: "test1",
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AdmissionWebhookMatchConditions, test.featureGateEnabled)()
			DropDisabledValidatingWebhookConfigurationFields(test.new, test.old)

			for _, hook := range test.new.Webhooks {
				if test.expected == nil {
					if hook.MatchConditions != nil {
						t.Error("expected all hooks matchConditions to be nil")
					}
				} else {
					require.Equal(t, len(test.expected), len(hook.MatchConditions))
					for i, matchCondition := range hook.MatchConditions {
						require.Equal(t, test.expected[i], matchCondition)
					}
				}
			}
		})
	}
}
