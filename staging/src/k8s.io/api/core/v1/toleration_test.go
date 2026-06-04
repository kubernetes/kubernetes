/*
Copyright 2017 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/klog/v2/ktesting"
	"testing"
)

func TestTolerationToleratesTaint(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description                                string
		toleration                                 Toleration
		taint                                      Taint
		expectTolerated                            bool
		expectError                                bool
		enableTaintTolerationComparisonOperatorsFG bool
	}{
		{
			description: "toleration and taint have the same key and effect, and operator is Exists, and taint has no value, expect tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpExists,
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key and effect, and operator is Exists, and taint has some value, expect tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpExists,
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same effect, toleration has empty key and operator is Exists, means match all taints, expect tolerated",
			toleration: Toleration{
				Key:      "",
				Operator: TolerationOpExists,
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key, effect and value, and operator is Equal, expect tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key and effect, but different values, and operator is Equal, expect not tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "value1",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "value2",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: false,
		},
		{
			description: "toleration and taint have the same key and value, but different effects, and operator is Equal, expect not tolerated",
			toleration: Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "foo",
				Value:  "bar",
				Effect: TaintEffectNoExecute,
			},
			expectTolerated: false,
		},
		{
			description: "toleration with Gt operator - taint value less than toleration value, expect not tolerated",
			toleration: Toleration{
				Key:      "node.kubernetes.io/sla",
				Operator: TolerationOpGt,
				Value:    "950",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "node.kubernetes.io/sla",
				Value:  "800",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: false,
			enableTaintTolerationComparisonOperatorsFG: true,
		},
		{
			description: "toleration with Gt operator - taint value greater than toleration value, expect tolerated",
			toleration: Toleration{
				Key:      "node.kubernetes.io/sla",
				Operator: TolerationOpGt,
				Value:    "750",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "node.kubernetes.io/sla",
				Value:  "950",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
			enableTaintTolerationComparisonOperatorsFG: true,
		},
		{
			description: "toleration with Lt operator - taint value greater than toleration value, expect not tolerated",
			toleration: Toleration{
				Key:      "node.kubernetes.io/sla",
				Operator: TolerationOpLt,
				Value:    "800",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "node.kubernetes.io/sla",
				Value:  "950",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: false,
			enableTaintTolerationComparisonOperatorsFG: true,
		},
		{
			description: "toleration with Lt operator - taint value less than toleration value, expect tolerated",
			toleration: Toleration{
				Key:      "node.kubernetes.io/sla",
				Operator: TolerationOpLt,
				Value:    "950",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "node.kubernetes.io/sla",
				Value:  "800",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: true,
			enableTaintTolerationComparisonOperatorsFG: true,
		},
		{
			description: "toleration with Gt operator and taint with equal numeric value, expect not tolerated",
			toleration: Toleration{
				Key:      "node.kubernetes.io/sla",
				Operator: TolerationOpGt,
				Value:    "950",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "node.kubernetes.io/sla",
				Value:  "950",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: false,
			enableTaintTolerationComparisonOperatorsFG: true,
		},
		{
			description: "toleration with Gt operator and taint with non-numeric value, expect not tolerated",
			toleration: Toleration{
				Key:      "node.kubernetes.io/sla",
				Operator: TolerationOpGt,
				Value:    "950",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "node.kubernetes.io/sla",
				Value:  "high",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: false,
			expectError:     true,
			enableTaintTolerationComparisonOperatorsFG: true,
		},
		{
			description: "toleration with Gt operator and negative numeric values - taint value less than threshold, expect not tolerated",
			toleration: Toleration{
				Key:      "test-key",
				Operator: TolerationOpGt,
				Value:    "-100",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "test-key",
				Value:  "-200",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: false,
			enableTaintTolerationComparisonOperatorsFG: true,
		},
		{
			description: "toleration with Gt operator and large int64 values - taint value less than threshold, expect not tolerated",
			toleration: Toleration{
				Key:      "test-key",
				Operator: TolerationOpGt,
				Value:    "9223372036854775806",
				Effect:   TaintEffectNoSchedule,
			},
			taint: Taint{
				Key:    "test-key",
				Value:  "100",
				Effect: TaintEffectNoSchedule,
			},
			expectTolerated: false,
			enableTaintTolerationComparisonOperatorsFG: true,
		},
	}
	for _, tc := range testCases {
		if tolerated := tc.toleration.ToleratesTaint(logger, &tc.taint, tc.enableTaintTolerationComparisonOperatorsFG); tc.expectTolerated != tolerated {
			t.Errorf("[%s] expect %v, got %v: toleration %+v, taint %s", tc.description, tc.expectTolerated, tolerated, tc.toleration, tc.taint.ToString())
		}
	}
}

func TestCompareNumericValues(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description    string
		tolerationVal  string
		taintVal       string
		operator       TolerationOperator
		expectedResult bool
	}{
		// Valid Gt operator cases
		{
			description:    "Gt operator - taint value greater than toleration value, expect true",
			tolerationVal:  "100",
			taintVal:       "200",
			operator:       TolerationOpGt,
			expectedResult: true,
		},
		{
			description:    "Gt operator - taint value less than toleration value, expect false",
			tolerationVal:  "200",
			taintVal:       "100",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - taint value equal to toleration value, expect false",
			tolerationVal:  "100",
			taintVal:       "100",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - negative numbers, taint greater, expect true",
			tolerationVal:  "-100",
			taintVal:       "-50",
			operator:       TolerationOpGt,
			expectedResult: true,
		},
		{
			description:    "Gt operator - negative numbers, taint less, expect false",
			tolerationVal:  "-50",
			taintVal:       "-100",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - zero and positive, expect true",
			tolerationVal:  "0",
			taintVal:       "1",
			operator:       TolerationOpGt,
			expectedResult: true,
		},
		{
			description:    "Gt operator - large int64 values, taint greater, expect true",
			tolerationVal:  "9223372036854775806",
			taintVal:       "9223372036854775807",
			operator:       TolerationOpGt,
			expectedResult: true,
		},

		// Valid Lt operator cases
		{
			description:    "Lt operator - taint value less than toleration value, expect true",
			tolerationVal:  "200",
			taintVal:       "100",
			operator:       TolerationOpLt,
			expectedResult: true,
		},
		{
			description:    "Lt operator - taint value greater than toleration value, expect false",
			tolerationVal:  "100",
			taintVal:       "200",
			operator:       TolerationOpLt,
			expectedResult: false,
		},
		{
			description:    "Lt operator - taint value equal to toleration value, expect false",
			tolerationVal:  "100",
			taintVal:       "100",
			operator:       TolerationOpLt,
			expectedResult: false,
		},
		{
			description:    "Lt operator - negative numbers, taint less, expect true",
			tolerationVal:  "-50",
			taintVal:       "-100",
			operator:       TolerationOpLt,
			expectedResult: true,
		},
		{
			description:    "Lt operator - negative numbers, taint greater, expect false",
			tolerationVal:  "-100",
			taintVal:       "-50",
			operator:       TolerationOpLt,
			expectedResult: false,
		},
		{
			description:    "Lt operator - zero and negative, expect true",
			tolerationVal:  "0",
			taintVal:       "-1",
			operator:       TolerationOpLt,
			expectedResult: true,
		},

		// Invalid toleration values - should return false
		{
			description:    "Gt operator - invalid toleration value (non-numeric), expect false",
			tolerationVal:  "abc",
			taintVal:       "100",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - invalid toleration value (empty string), expect false",
			tolerationVal:  "",
			taintVal:       "100",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - invalid toleration value (leading zero), expect false",
			tolerationVal:  "0100",
			taintVal:       "200",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - invalid toleration value (plus sign), expect false",
			tolerationVal:  "+100",
			taintVal:       "200",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - invalid toleration value (floating point), expect false",
			tolerationVal:  "100.5",
			taintVal:       "200",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - invalid toleration value (just minus sign), expect false",
			tolerationVal:  "-",
			taintVal:       "100",
			operator:       TolerationOpGt,
			expectedResult: false,
		},

		// Invalid taint values - should return false
		{
			description:    "Gt operator - invalid taint value (non-numeric), expect false",
			tolerationVal:  "100",
			taintVal:       "xyz",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - invalid taint value (empty string), expect false",
			tolerationVal:  "100",
			taintVal:       "",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Gt operator - invalid taint value (leading zero), expect false",
			tolerationVal:  "100",
			taintVal:       "0200",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Lt operator - invalid taint value (plus sign), expect false",
			tolerationVal:  "100",
			taintVal:       "+200",
			operator:       TolerationOpLt,
			expectedResult: false,
		},
		{
			description:    "Lt operator - invalid taint value (spaces), expect false",
			tolerationVal:  "100",
			taintVal:       " 200 ",
			operator:       TolerationOpLt,
			expectedResult: false,
		},

		// Invalid operator - should return false
		{
			description:    "Equal operator (unsupported for numeric comparison), expect false",
			tolerationVal:  "100",
			taintVal:       "100",
			operator:       TolerationOpEqual,
			expectedResult: false,
		},
		{
			description:    "Exists operator (unsupported for numeric comparison), expect false",
			tolerationVal:  "100",
			taintVal:       "100",
			operator:       TolerationOpExists,
			expectedResult: false,
		},

		// Edge cases with zero
		{
			description:    "Gt operator - both zero, expect false",
			tolerationVal:  "0",
			taintVal:       "0",
			operator:       TolerationOpGt,
			expectedResult: false,
		},
		{
			description:    "Lt operator - both zero, expect false",
			tolerationVal:  "0",
			taintVal:       "0",
			operator:       TolerationOpLt,
			expectedResult: false,
		},

		// Int64 boundary cases
		{
			description:    "Gt operator - max int64 as taint, expect true",
			tolerationVal:  "0",
			taintVal:       "9223372036854775807",
			operator:       TolerationOpGt,
			expectedResult: true,
		},
		{
			description:    "Lt operator - min int64 as taint, expect true",
			tolerationVal:  "0",
			taintVal:       "-9223372036854775808",
			operator:       TolerationOpLt,
			expectedResult: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			result := compareNumericValues(logger, tc.tolerationVal, tc.taintVal, tc.operator)
			if result != tc.expectedResult {
				t.Errorf("[%s] expected %v, got %v: tolerationVal=%q, taintVal=%q, operator=%v",
					tc.description, tc.expectedResult, result, tc.tolerationVal, tc.taintVal, tc.operator)
			}
		})
	}
}
