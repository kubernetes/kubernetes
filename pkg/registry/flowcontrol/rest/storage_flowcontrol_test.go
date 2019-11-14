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
	"testing"

	"github.com/stretchr/testify/assert"
	flowcontrol "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/client-go/kubernetes/fake"
)

func TestShouldEnsurePredefinedSettings(t *testing.T) {
	testCases := []struct {
		name                  string
		existingPriorityLevel *flowcontrol.PriorityLevelConfiguration
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
			should, err := shouldEnsureAllPredefined(c.FlowcontrolV1alpha1())
			assert.NoError(t, err)
			assert.Equal(t, testCase.expected, should)
		})
	}
}
