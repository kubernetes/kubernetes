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

package metrics

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestDefaultStabilityLevel(t *testing.T) {
	var tests = []struct {
		name        string
		inputValue  StabilityLevel
		expectValue StabilityLevel
		expectPanic bool
	}{
		{
			name:        "empty should take ALPHA by default",
			inputValue:  "",
			expectValue: ALPHA,
			expectPanic: false,
		},
		{
			name:        "ALPHA remain unchanged",
			inputValue:  ALPHA,
			expectValue: ALPHA,
			expectPanic: false,
		},
		{
			name:        "STABLE remain unchanged",
			inputValue:  STABLE,
			expectValue: STABLE,
			expectPanic: false,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			var stability = tc.inputValue

			stability.setDefaults()
			assert.Equalf(t, tc.expectValue, stability, "Got %s, expected: %v ", stability, tc.expectValue)
		})
	}
}

func TestConstrainToAllowedList(t *testing.T) {
	allowList := &MetricLabelAllowList{
		labelToAllowList: map[string]sets.String{
			"label_a": sets.NewString("allow_value1", "allow_value2"),
		},
	}
	labelNameList := []string{"label_a", "label_b"}
	var tests = []struct {
		name                 string
		inputLabelValueList  []string
		outputLabelValueList []string
	}{
		{
			"no unexpected value",
			[]string{"allow_value1", "label_b_value"},
			[]string{"allow_value1", "label_b_value"},
		},
		{
			"with unexpected value",
			[]string{"not_allowed", "label_b_value"},
			[]string{"unexpected", "label_b_value"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			allowList.ConstrainToAllowedList(labelNameList, test.inputLabelValueList)
			if !reflect.DeepEqual(test.inputLabelValueList, test.outputLabelValueList) {
				t.Errorf("Got %v, expected %v", test.inputLabelValueList, test.outputLabelValueList)
			}
		})
	}
}

func TestConstrainLabelMap(t *testing.T) {
	allowList := &MetricLabelAllowList{
		labelToAllowList: map[string]sets.String{
			"label_a": sets.NewString("allow_value1", "allow_value2"),
		},
	}
	var tests = []struct {
		name           string
		inputLabelMap  map[string]string
		outputLabelMap map[string]string
	}{
		{
			"no unexpected value",
			map[string]string{
				"label_a": "allow_value1",
				"label_b": "label_b_value",
			},
			map[string]string{
				"label_a": "allow_value1",
				"label_b": "label_b_value",
			},
		},
		{
			"with unexpected value",
			map[string]string{
				"label_a": "not_allowed",
				"label_b": "label_b_value",
			},
			map[string]string{
				"label_a": "unexpected",
				"label_b": "label_b_value",
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			allowList.ConstrainLabelMap(test.inputLabelMap)
			if !reflect.DeepEqual(test.inputLabelMap, test.outputLabelMap) {
				t.Errorf("Got %v, expected %v", test.inputLabelMap, test.outputLabelMap)
			}
		})
	}
}
