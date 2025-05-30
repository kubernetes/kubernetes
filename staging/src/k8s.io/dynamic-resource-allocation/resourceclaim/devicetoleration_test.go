/*
Copyright 2025 The Kubernetes Authors.

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

package resourceclaim

import (
	"testing"

	resourceapi "k8s.io/api/resource/v1beta1"
)

func TestTolerationToleratesTaint(t *testing.T) {

	testCases := []struct {
		description     string
		toleration      resourceapi.DeviceToleration
		taint           resourceapi.DeviceTaint
		expectTolerated bool
	}{
		{
			description: "toleration and taint have the same key and effect, and operator is Exists, and taint has no value, expect tolerated",
			toleration: resourceapi.DeviceToleration{
				Key:      "foo",
				Operator: resourceapi.DeviceTolerationOpExists,
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
			},
			taint: resourceapi.DeviceTaint{
				Key:    "foo",
				Effect: resourceapi.DeviceTaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key and effect, and operator is Exists, and taint has some value, expect tolerated",
			toleration: resourceapi.DeviceToleration{
				Key:      "foo",
				Operator: resourceapi.DeviceTolerationOpExists,
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
			},
			taint: resourceapi.DeviceTaint{
				Key:    "foo",
				Value:  "bar",
				Effect: resourceapi.DeviceTaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same effect, toleration has empty key and operator is Exists, means match all taints, expect tolerated",
			toleration: resourceapi.DeviceToleration{
				Key:      "",
				Operator: resourceapi.DeviceTolerationOpExists,
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
			},
			taint: resourceapi.DeviceTaint{
				Key:    "foo",
				Value:  "bar",
				Effect: resourceapi.DeviceTaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key, effect and value, and operator is Equal, expect tolerated",
			toleration: resourceapi.DeviceToleration{
				Key:      "foo",
				Operator: resourceapi.DeviceTolerationOpEqual,
				Value:    "bar",
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
			},
			taint: resourceapi.DeviceTaint{
				Key:    "foo",
				Value:  "bar",
				Effect: resourceapi.DeviceTaintEffectNoSchedule,
			},
			expectTolerated: true,
		},
		{
			description: "toleration and taint have the same key and effect, but different values, and operator is Equal, expect not tolerated",
			toleration: resourceapi.DeviceToleration{
				Key:      "foo",
				Operator: resourceapi.DeviceTolerationOpEqual,
				Value:    "value1",
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
			},
			taint: resourceapi.DeviceTaint{
				Key:    "foo",
				Value:  "value2",
				Effect: resourceapi.DeviceTaintEffectNoSchedule,
			},
			expectTolerated: false,
		},
		{
			description: "toleration and taint have the same key and value, but different effects, and operator is Equal, expect not tolerated",
			toleration: resourceapi.DeviceToleration{
				Key:      "foo",
				Operator: resourceapi.DeviceTolerationOpEqual,
				Value:    "bar",
				Effect:   resourceapi.DeviceTaintEffectNoSchedule,
			},
			taint: resourceapi.DeviceTaint{
				Key:    "foo",
				Value:  "bar",
				Effect: resourceapi.DeviceTaintEffectNoExecute,
			},
			expectTolerated: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			if tolerated := ToleratesTaint(tc.toleration, tc.taint); tc.expectTolerated != tolerated {
				t.Errorf("expect %v, got %v: toleration %+v, taint %s", tc.expectTolerated, tolerated, tc.toleration, tc.taint)
			}
		})
	}
}
