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

package core

import "testing"

func TestMatchToleration(t *testing.T) {

	tolerationSeconds := int64(5)
	tolerationToMatchSeconds := int64(3)
	testCases := []struct {
		description       string
		toleration        *Toleration
		tolerationToMatch *Toleration
		expectMatch       bool
	}{
		{
			description: "two taints with the same key,operator,value,effect should match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: "Exists",
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "foo",
				Operator: "Exists",
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			expectMatch: true,
		},
		{
			description: "two taints with the different key cannot match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: "Exists",
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "different-key",
				Operator: "Exists",
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			expectMatch: false,
		},
		{
			description: "two taints with the different operator cannot match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: "Exists",
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "foo",
				Operator: "different-operator",
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			expectMatch: false,
		},
		{
			description: "two taints with the different value cannot match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: "Exists",
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "foo",
				Operator: "Exists",
				Value:    "different-value",
				Effect:   TaintEffectNoSchedule,
			},
			expectMatch: false,
		},
		{
			description: "two taints with the different effect cannot match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: "Exists",
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "foo",
				Operator: "Exists",
				Value:    "bar",
				Effect:   TaintEffectPreferNoSchedule,
			},
			expectMatch: false,
		},
		{
			description: "two taints with the different tolerationSeconds should match",
			toleration: &Toleration{
				Key:               "foo",
				Operator:          "Exists",
				Value:             "bar",
				Effect:            TaintEffectNoSchedule,
				TolerationSeconds: &tolerationSeconds,
			},
			tolerationToMatch: &Toleration{
				Key:               "foo",
				Operator:          "Exists",
				Value:             "bar",
				Effect:            TaintEffectNoSchedule,
				TolerationSeconds: &tolerationToMatchSeconds,
			},
			expectMatch: true,
		},
	}

	for _, tc := range testCases {
		if actual := tc.toleration.MatchToleration(tc.tolerationToMatch); actual != tc.expectMatch {
			t.Errorf("[%s] expect: %v , got:  %v", tc.description, tc.expectMatch, !tc.expectMatch)
		}
	}
}
