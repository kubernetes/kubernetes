/*
Copyright 2019 The Kubernetes Authors.

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

package rest

import (
	"github.com/stretchr/testify/require"
	"testing"

	"github.com/stretchr/testify/assert"
	flowcontrolv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/client-go/kubernetes/fake"
	flowcontrolapisv1alpha1 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1alpha1"
)

func TestShouldEnsurePredefinedSettings(t *testing.T) {
	testCases := []struct {
		name                  string
		existingPriorityLevel *flowcontrolv1alpha1.PriorityLevelConfiguration
		expected              bool
	}{
		{
			name:                  "should ensure if exempt priority-level is absent",
			existingPriorityLevel: nil,
			expected:              true,
		},
		{
			name:                  "should not ensure if exempt priority-level is present",
			existingPriorityLevel: bootstrap.MandatoryPriorityLevelConfigurationExempt,
			expected:              false,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			c := fake.NewSimpleClientset()
			if testCase.existingPriorityLevel != nil {
				c.FlowcontrolV1alpha1().PriorityLevelConfigurations().Create(testCase.existingPriorityLevel)
			}
			should, err := lastMandatoryExists(c.FlowcontrolV1alpha1())
			assert.NoError(t, err)
			assert.Equal(t, testCase.expected, should)
		})
	}
}

func TestFlowSchemaHasWrongSpec(t *testing.T) {
	fs1 := &flowcontrolv1alpha1.FlowSchema{
		Spec: flowcontrolv1alpha1.FlowSchemaSpec{},
	}
	fs2 := &flowcontrolv1alpha1.FlowSchema{
		Spec: flowcontrolv1alpha1.FlowSchemaSpec{
			MatchingPrecedence: 1,
		},
	}
	fs1Defaulted := &flowcontrolv1alpha1.FlowSchema{
		Spec: flowcontrolv1alpha1.FlowSchemaSpec{
			MatchingPrecedence: flowcontrolapisv1alpha1.FlowSchemaDefaultMatchingPrecedence,
		},
	}
	testCases := []struct {
		name         string
		expected     *flowcontrolv1alpha1.FlowSchema
		actual       *flowcontrolv1alpha1.FlowSchema
		hasWrongSpec bool
	}{
		{
			name:         "identical flow-schemas should work",
			expected:     bootstrap.MandatoryFlowSchemaCatchAll,
			actual:       bootstrap.MandatoryFlowSchemaCatchAll,
			hasWrongSpec: false,
		},
		{
			name:         "defaulted flow-schemas should work",
			expected:     fs1,
			actual:       fs1Defaulted,
			hasWrongSpec: false,
		},
		{
			name:         "non-defaulted flow-schema has wrong spec",
			expected:     fs1,
			actual:       fs2,
			hasWrongSpec: true,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			w, err := flowSchemaHasWrongSpec(testCase.expected, testCase.actual)
			require.NoError(t, err)
			assert.Equal(t, testCase.hasWrongSpec, w)
		})
	}
}

func TestPriorityLevelHasWrongSpec(t *testing.T) {
	pl1 := &flowcontrolv1alpha1.PriorityLevelConfiguration{
		Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
				LimitResponse: flowcontrolv1alpha1.LimitResponse{
					Type: flowcontrolv1alpha1.LimitResponseTypeReject,
				},
			},
		},
	}
	pl2 := &flowcontrolv1alpha1.PriorityLevelConfiguration{
		Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: 1,
			},
		},
	}
	pl1Defaulted := &flowcontrolv1alpha1.PriorityLevelConfiguration{
		Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: flowcontrolapisv1alpha1.PriorityLevelConfigurationDefaultAssuredConcurrencyShares,
				LimitResponse: flowcontrolv1alpha1.LimitResponse{
					Type: flowcontrolv1alpha1.LimitResponseTypeReject,
				},
			},
		},
	}
	testCases := []struct {
		name         string
		expected     *flowcontrolv1alpha1.PriorityLevelConfiguration
		actual       *flowcontrolv1alpha1.PriorityLevelConfiguration
		hasWrongSpec bool
	}{
		{
			name:         "identical priority-level should work",
			expected:     bootstrap.MandatoryPriorityLevelConfigurationCatchAll,
			actual:       bootstrap.MandatoryPriorityLevelConfigurationCatchAll,
			hasWrongSpec: false,
		},
		{
			name:         "defaulted priority-level should work",
			expected:     pl1,
			actual:       pl1Defaulted,
			hasWrongSpec: false,
		},
		{
			name:         "non-defaulted priority-level has wrong spec",
			expected:     pl1,
			actual:       pl2,
			hasWrongSpec: true,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			w, err := priorityLevelHasWrongSpec(testCase.expected, testCase.actual)
			require.NoError(t, err)
			assert.Equal(t, testCase.hasWrongSpec, w)
		})
	}
}
