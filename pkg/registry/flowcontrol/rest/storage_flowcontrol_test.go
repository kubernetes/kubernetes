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
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	flowcontrolv1beta1 "k8s.io/api/flowcontrol/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/client-go/kubernetes/fake"
	flowcontrolapisv1beta1 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta1"
)

func TestShouldEnsurePredefinedSettings(t *testing.T) {
	testCases := []struct {
		name                  string
		existingPriorityLevel *flowcontrolv1beta1.PriorityLevelConfiguration
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
				c.FlowcontrolV1beta1().PriorityLevelConfigurations().Create(context.TODO(), testCase.existingPriorityLevel, metav1.CreateOptions{})
			}
			should, err := shouldEnsureSuggested(c.FlowcontrolV1beta1())
			assert.NoError(t, err)
			assert.Equal(t, testCase.expected, should)
		})
	}
}

func TestFlowSchemaHasWrongSpec(t *testing.T) {
	fs1 := &flowcontrolv1beta1.FlowSchema{
		Spec: flowcontrolv1beta1.FlowSchemaSpec{},
	}
	fs2 := &flowcontrolv1beta1.FlowSchema{
		Spec: flowcontrolv1beta1.FlowSchemaSpec{
			MatchingPrecedence: 1,
		},
	}
	fs1Defaulted := &flowcontrolv1beta1.FlowSchema{
		Spec: flowcontrolv1beta1.FlowSchemaSpec{
			MatchingPrecedence: flowcontrolapisv1beta1.FlowSchemaDefaultMatchingPrecedence,
		},
	}
	testCases := []struct {
		name         string
		expected     *flowcontrolv1beta1.FlowSchema
		actual       *flowcontrolv1beta1.FlowSchema
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
	pl1 := &flowcontrolv1beta1.PriorityLevelConfiguration{
		Spec: flowcontrolv1beta1.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1beta1.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1beta1.LimitedPriorityLevelConfiguration{
				LimitResponse: flowcontrolv1beta1.LimitResponse{
					Type: flowcontrolv1beta1.LimitResponseTypeReject,
				},
			},
		},
	}
	pl2 := &flowcontrolv1beta1.PriorityLevelConfiguration{
		Spec: flowcontrolv1beta1.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1beta1.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1beta1.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: 1,
			},
		},
	}
	pl1Defaulted := &flowcontrolv1beta1.PriorityLevelConfiguration{
		Spec: flowcontrolv1beta1.PriorityLevelConfigurationSpec{
			Type: flowcontrolv1beta1.PriorityLevelEnablementLimited,
			Limited: &flowcontrolv1beta1.LimitedPriorityLevelConfiguration{
				AssuredConcurrencyShares: flowcontrolapisv1beta1.PriorityLevelConfigurationDefaultAssuredConcurrencyShares,
				LimitResponse: flowcontrolv1beta1.LimitResponse{
					Type: flowcontrolv1beta1.LimitResponseTypeReject,
				},
			},
		},
	}
	testCases := []struct {
		name         string
		expected     *flowcontrolv1beta1.PriorityLevelConfiguration
		actual       *flowcontrolv1beta1.PriorityLevelConfiguration
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
