/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package manager

import (
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
)

type testcase struct {
	Name        string
	Actions     []autoscaler.ScalingAction
	Expectation autoscaler.ScalingAction
}

func getTestCases() []testcase {
	return []testcase{
		{
			Name: "simple-noop-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeNone},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeNone,
				ScaleBy: 0,
			},
		},
		{
			Name: "simple-scale-up-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeScaleUp,
				ScaleBy: 2,
			},
		},
		{
			Name: "simple-scale-down-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 4},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeScaleDown,
				ScaleBy: 4,
			},
		},
		{
			Name: "multi-noop-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeNone},
				{Type: api.AutoScaleActionTypeNone, ScaleBy: -1},
				{Type: api.AutoScaleActionTypeNone, ScaleBy: 1},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeNone,
				ScaleBy: 0,
			},
		},
		{
			Name: "multi-scale-up-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 3},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 4},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 3},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeScaleUp,
				ScaleBy: 4,
			},
		},
		{
			Name: "multi-scale-up-zero-negative-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 1},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: -1},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 0},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeScaleUp,
				ScaleBy: 2,
			},
		},
		{
			Name: "multi-scale-down-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 4},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 6},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 6},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 6},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 4},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeScaleDown,
				ScaleBy: 4,
			},
		},
		{
			Name: "multi-scale-down-negative-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 1},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 1},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 2},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 3},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 5},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 8},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: -1},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: -1},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: -2},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeScaleDown,
				ScaleBy: 1,
			},
		},
		{
			Name: "combo-scale-test",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeNone, ScaleBy: -1},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 22},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 11},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 5},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 4},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 3},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 1},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 42},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 21},
				{Type: api.AutoScaleActionTypeScaleUp, ScaleBy: 10},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 10},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeScaleUp,
				ScaleBy: 42,
			},
		},
		{
			Name: "combo-test-2",
			Actions: []autoscaler.ScalingAction{
				{Type: api.AutoScaleActionTypeNone, ScaleBy: -1},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 4},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 2},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 8},
				{Type: api.AutoScaleActionTypeScaleDown, ScaleBy: 6},
				{Type: api.AutoScaleActionTypeNone},
			},
			Expectation: autoscaler.ScalingAction{
				Type:    api.AutoScaleActionTypeScaleDown,
				ScaleBy: 2,
			},
		},
	}
}

func checkExpectations(result, expected autoscaler.ScalingAction, triggerCheck bool) error {
	if result.Type != expected.Type {
		return fmt.Errorf("got scale type %q, expected %q", result.Type, expected.Type)
	}

	if result.ScaleBy != expected.ScaleBy {
		return fmt.Errorf("got scale by %v, expected %v", result.ScaleBy, expected.ScaleBy)
	}

	if triggerCheck {
		if result.Trigger.Type != expected.Trigger.Type {
			return fmt.Errorf("trigger types does not match expectation")
		}

		if result.Trigger.ActionType != expected.Trigger.ActionType {
			return fmt.Errorf("trigger action types does not match expectation")
		}

		if result.Trigger.ScaleBy != expected.Trigger.ScaleBy {
			return fmt.Errorf("trigger scale by does not match expectation")
		}
	}

	return nil
}

func TestReconcileActions(t *testing.T) {
	for _, tc := range getTestCases() {
		result := ReconcileActions(tc.Actions)
		if err := checkExpectations(result, tc.Expectation, false); err != nil {
			t.Errorf("Test case %s %s", tc.Name, err)
		}
	}
}
