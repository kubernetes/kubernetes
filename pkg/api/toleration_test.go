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

package api

import "testing"

func TestMatchToleration(t *testing.T) {
	testCases := []struct {
		description       string
		toleration        *Toleration
		tolerationToMatch *Toleration
		expectMatch       bool
	}{
		{
			description: "two tolerations with same key,operator,value,effect should match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			expectMatch: true,
		},
		{
			description: "two tolerations with same key,operator,value but different effect cannot match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoExecute,
			},
			expectMatch: false,
		},
		{
			description: "two tolerations with same key,value,effect but different operator cannot match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "foo",
				Operator: TolerationOpExists,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			expectMatch: false,
		},
		{
			description: "two tolerations with same key,operator,effect but different value cannot match",
			toleration: &Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "bar",
				Effect:   TaintEffectNoSchedule,
			},
			tolerationToMatch: &Toleration{
				Key:      "foo",
				Operator: TolerationOpEqual,
				Value:    "different-value",
				Effect:   TaintEffectNoSchedule,
			},
			expectMatch: false,
		},
	}
	for _, test := range testCases {
		if test.expectMatch != test.toleration.MatchToleration(test.tolerationToMatch) {
			t.Errorf("[%s] expect toleration:\n%#v\nmatch toleration:\n%#v", test.description, test.toleration, test.tolerationToMatch)
		}
	}
}
