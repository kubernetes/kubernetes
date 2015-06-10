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

package generic

import (
	"math/rand"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
	amt "github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler/advisors/test"
)

type filteredAdvisorsTestCase struct {
	Name             string
	Advisors         []autoscaler.Advisor
	ErrorExpectation bool
}

func TestNewGenericAutoScaler(t *testing.T) {
	names := []string{"as", "@uto$caler", "blues", "", "jazz", "zydeco"}

	for _, name := range names {
		gas := NewGenericAutoScaler(name)
		if nil == gas {
			t.Errorf("new instance failed for %q", name)
		}
	}
}

func TestGenericAutoScalerName(t *testing.T) {
	testCases := []struct {
		Name        string
		Expectation string
	}{
		{Name: "aye", Expectation: "aye"},
		{Name: "@uto$caler", Expectation: "@uto$caler"},
		{Name: "", Expectation: DefaultName},
		{Name: "dazed-n-confused", Expectation: "dazed-n-confused"},
		{Name: "zed", Expectation: "zed"},
	}

	for _, tc := range testCases {
		gas := NewGenericAutoScaler(tc.Name)
		if gas != nil {
			name := gas.Name()
			if name != tc.Expectation {
				t.Errorf("expected name %q, got back %q",
					tc.Name, name)
			}
		} else {
			t.Errorf("new instance failed for %q", tc.Name)
		}
	}
}

func createAutoScalerSpec(numThresholds int) api.AutoScalerSpec {
	if numThresholds < 1 {
		numThresholds = 1
	}

	intentionTypes := []api.AutoScaleIntentionType{
		api.AutoScaleIntentionTypeRequestsPerSecond,
		api.AutoScaleIntentionTypeCpuUsage,
		api.AutoScaleIntentionTypeLoadAverage,
		api.AutoScaleIntentionTypeMemoryUsage,
	}

	itlen := len(intentionTypes)

	thresholds := make([]api.AutoScaleThreshold, numThresholds)
	for idx := 0; idx < numThresholds; idx++ {
		numIntents := rand.Intn(10)
		intentions := make([]api.AutoScaleIntentionThresholdConfig, numIntents)

		for j := 0; j < numIntents; j++ {
			intent := api.AutoScaleIntentionThresholdConfig{
				Intent:   intentionTypes[rand.Intn(itlen)],
				Value:    42,
				Duration: 300,
			}

			intentions = append(intentions, intent)
		}

		t := api.AutoScaleThreshold{
			Type:       api.AutoScaleThresholdTypeIntention,
			Intentions: intentions,
			ActionType: api.AutoScaleActionTypeScaleUp,
			ScaleBy:    3,
		}
		thresholds = append(thresholds, t)
	}

	return api.AutoScalerSpec{
		Thresholds:        thresholds,
		MinAutoScaleCount: 21,
		MaxAutoScaleCount: 42,

		TargetSelector: map[string]string{"ipl": "mumbai-indians"},

		Advisors: []string{"one", "two", "three", "cha-cha-cha", "guru", "megadeath"},
	}
}

func getFilteredAdvisorsTestCases() []filteredAdvisorsTestCase {
	return []filteredAdvisorsTestCase{
		{
			Name: "filter-false-steps",
			Advisors: []autoscaler.Advisor{
				&amt.TestAdvisor{
					Tag:    "one",
					Status: true,
					Error:  "",
				},
				&amt.TestAdvisor{
					Tag:    "two",
					Status: true,
					Error:  "",
				},
				&amt.TestAdvisor{
					Tag:    "cha-cha-cha",
					Status: true,
					Error:  "",
				},
				&amt.TestAdvisor{
					Tag:    "false-step",
					Status: false,
					Error:  "should-not-use-advisor",
				},
			},
			ErrorExpectation: false,
		},
		{
			Name: "expected-filtered-advisor-error",
			Advisors: []autoscaler.Advisor{
				&amt.TestAdvisor{
					Tag:    "1",
					Status: true,
					Error:  "",
				},
				&amt.TestAdvisor{
					Tag:    "2",
					Status: true,
					Error:  "",
				},
				&amt.TestAdvisor{
					Tag:    "3 o'clock",
					Status: true,
					Error:  "",
				},
				&amt.TestAdvisor{
					Tag:    "4 o'clock rock",
					Status: true,
					Error:  "",
				},
				&amt.TestAdvisor{
					Tag:    "guru",
					Status: false,
					Error:  "should-use-this-advisor",
				},
			},
			ErrorExpectation: true,
		},
	}
}

func TestGenericAutoScalerAssess(t *testing.T) {
	truthyAdvisors := []autoscaler.Advisor{
		&amt.TestAdvisor{Tag: "one", Status: true, Error: ""},
		&amt.TestAdvisor{Tag: "two", Status: true, Error: ""},
	}

	liarAdvisors := []autoscaler.Advisor{
		&amt.TestAdvisor{Tag: "one", Status: false, Error: ""},
	}

	errorAdvisors := []autoscaler.Advisor{
		&amt.TestAdvisor{Tag: "guru", Status: false, Error: "gas gas gas"},
		&amt.TestAdvisor{Tag: "megadeath", Status: false, Error: "it's a gas"},
	}

	comboAdvisors := []autoscaler.Advisor{
		&amt.TestAdvisor{Tag: "one", Status: true, Error: ""},
		&amt.TestAdvisor{Tag: "two", Status: true, Error: ""},
		&amt.TestAdvisor{Tag: "guru", Status: false, Error: "blues"},
		&amt.TestAdvisor{Tag: "megadeath", Status: true, Error: "hey paranoid"},
	}

	multiAdvisors := []autoscaler.Advisor{
		&amt.TestAdvisor{Tag: "two", Status: true, Error: ""},
		&amt.TestAdvisor{Tag: "three", Status: true, Error: ""},
		&amt.TestAdvisor{Tag: "cha-cha-cha", Status: false, Error: ""},
		&amt.TestAdvisor{Tag: "one", Status: false, Error: ""},
		&amt.TestAdvisor{Tag: "unreferenced", Status: true, Error: ""},
	}

	testCases := []struct {
		Name             string
		Advisors         []autoscaler.Advisor
		Scale            bool
		ErrorExpectation bool
	}{
		{
			Name:             "truth or consequences",
			Advisors:         truthyAdvisors,
			Scale:            true,
			ErrorExpectation: false,
		},
		{
			Name:             "liar liar",
			Advisors:         liarAdvisors,
			Scale:            false,
			ErrorExpectation: false,
		},
		{
			Name:             "to err or not to err that ...",
			Advisors:         errorAdvisors,
			Scale:            false,
			ErrorExpectation: true,
		},
		{
			Name:             "combinatorial-explosion",
			Advisors:         comboAdvisors,
			Scale:            true,
			ErrorExpectation: true,
		},
		{
			Name:             "multi-phasic",
			Advisors:         multiAdvisors,
			Scale:            true,
			ErrorExpectation: false,
		},
	}

	specs := []api.AutoScalerSpec{
		createAutoScalerSpec(1),
		createAutoScalerSpec(2),
		createAutoScalerSpec(3),
		createAutoScalerSpec(6),
	}

	noAction := autoscaler.ScalingAction{
		Type:    api.AutoScaleActionTypeNone,
		ScaleBy: 0,
	}

	for _, tc := range testCases {
		gas := NewGenericAutoScaler(tc.Name)

		for _, spec := range specs {
			actions, err := gas.Assess(spec, tc.Advisors)
			if tc.ErrorExpectation {
				if nil == err {
					t.Errorf("test %q got no error, expected one",
						tc.Name)
				}
			}

			action := noAction
			if len(actions) > 0 {
				// For this test, use last valid action.
				for _, a := range actions {
					if api.AutoScaleActionTypeNone == a.Type {
						continue
					}
					action = a
				}
			}

			if tc.Scale && api.AutoScaleActionTypeNone == action.Type {
				t.Errorf("test %q got no scaling action, expected a scaling action",
					tc.Name)
			}

			if !tc.Scale && action.Type != api.AutoScaleActionTypeNone {
				t.Errorf("test %q got a scaling action %v, expected none",
					tc.Name, action.Type)
			}
		}
	}

	// Run filtered advisors tests.
	filteredAdvisorsTestCases := getFilteredAdvisorsTestCases()
	for _, tc := range filteredAdvisorsTestCases {
		gas := NewGenericAutoScaler(tc.Name)
		for _, spec := range specs {
			_, err := gas.Assess(spec, tc.Advisors)
			if tc.ErrorExpectation {
				if nil == err {
					t.Errorf("test %q got no error, expected one",
						tc.Name)
				}
				continue
			}

			if err != nil {
				t.Errorf("test %q got error %v, expected none",
					tc.Name)
			}
		}
	}
}
