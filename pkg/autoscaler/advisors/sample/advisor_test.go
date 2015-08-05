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

package sample

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
)

func TestNewSampleScalingAdvisor(t *testing.T) {
	testCases := []struct {
		Name    string
		Advisor autoscaler.Advisor
	}{
		{
			Name:    "truth-or-consequences",
			Advisor: NewTruthinessScalingAdvisor(),
		},
		{
			Name:    "cognito-err-go-sum-quoque",
			Advisor: NewFalsinessScalingAdvisor(),
		},
	}

	for _, tc := range testCases {
		if nil == tc.Advisor {
			t.Errorf("test case %q got nil advisor", tc.Name)
		}
	}
}

func TestSampleScalingAdvisorInitialize(t *testing.T) {
	testCases := []struct {
		Name    string
		Advisor autoscaler.Advisor
	}{
		{
			Name:    "truth-or-consequences",
			Advisor: NewTruthinessScalingAdvisor(),
		},
		{
			Name:    "cognito-err-go-sum-quoque",
			Advisor: NewFalsinessScalingAdvisor(),
		},
	}

	kubeClient := testclient.NewSimpleFake(&api.AutoScalerList{})

	for _, tc := range testCases {
		if nil == tc.Advisor {
			t.Errorf("test case %q got nil advisor, expected one", tc.Name)
			continue
		}

		err := tc.Advisor.Initialize(kubeClient)
		if err != nil {
			t.Errorf("test case %q got initialize error %q, expected no error",
				tc.Name, err)
		}
	}
}

func TestSampleScalingAdvisorName(t *testing.T) {
	testCases := []struct {
		Name         string
		Advisor      autoscaler.Advisor
		ExpectedName string
	}{
		{
			Name:         "true-dat",
			Advisor:      NewTruthinessScalingAdvisor(),
			ExpectedName: TruthinessScalingAdvisorName,
		},
		{
			Name:         "falsetto",
			Advisor:      NewFalsinessScalingAdvisor(),
			ExpectedName: FalsinessScalingAdvisorName,
		},
	}

	for _, tc := range testCases {
		if nil == tc.Advisor {
			t.Errorf("test case %q got nil advisor, expected one", tc.Name)
			continue
		}

		advisorName := tc.Advisor.Name()
		if advisorName != tc.ExpectedName {
			t.Errorf("test case %q got advisor name %q, expected %q",
				tc.Name, advisorName, tc.ExpectedName)
		}
	}
}

func getSampleThresholds() []api.AutoScaleThreshold {
	usageThresholds := []api.AutoScaleIntentionThresholdConfig{
		{
			Intent:   api.AutoScaleIntentionTypeCpuUsage,
			Value:    40,
			Duration: 300,
		},
		{
			Intent:   api.AutoScaleIntentionTypeMemoryUsage,
			Value:    655360,
			Duration: 600,
		},
	}

	customIntentions := []api.AutoScaleIntentionThresholdConfig{
		{
			Intent:   "FreeMemoryPages",
			Value:    8,
			Duration: 60,
		},
	}

	rpsThreshold := []api.AutoScaleIntentionThresholdConfig{
		{
			Intent:   api.AutoScaleIntentionTypeRequestsPerSecond,
			Value:    123,
			Duration: 900,
		},
	}

	return []api.AutoScaleThreshold{
		{
			Type:       api.AutoScaleThresholdTypeIntention,
			Intentions: usageThresholds,
			ActionType: api.AutoScaleActionTypeScaleDown,
			ScaleBy:    1,
		},
		{
			Type:       api.AutoScaleThresholdTypeIntention,
			Intentions: customIntentions,
			ActionType: api.AutoScaleActionTypeScaleUp,
			ScaleBy:    2,
		},
		{
			Type:       api.AutoScaleThresholdTypeIntention,
			Intentions: rpsThreshold,
			ActionType: api.AutoScaleActionTypeScaleUp,
			ScaleBy:    3,
		},
	}
}

func TestSampleScalingAdvisorEvaluate(t *testing.T) {
	testCases := []struct {
		Name        string
		Advisor     autoscaler.Advisor
		Thresholds  []api.AutoScaleThreshold
		Expectation bool
	}{
		{
			Name:        "construe",
			Advisor:     NewTruthinessScalingAdvisor(),
			Thresholds:  getSampleThresholds(),
			Expectation: true,
		},
		{
			Name:        "falsehood",
			Advisor:     NewFalsinessScalingAdvisor(),
			Thresholds:  getSampleThresholds(),
			Expectation: false,
		},
	}

	op := autoscaler.AggregateOperatorTypeAny

	for _, tc := range testCases {
		if nil == tc.Advisor {
			t.Errorf("test case %q got nil advisor, expected one", tc.Name)
			continue
		}

		for _, threshold := range tc.Thresholds {
			verdict, err := tc.Advisor.Evaluate(op, threshold.Intentions)
			if err != nil {
				t.Errorf("test case %q got error %q, expected no error",
					tc.Name, err)
			}

			if verdict != tc.Expectation {
				t.Errorf("test case %q evaluated %q, expected %q",
					tc.Name, verdict, tc.Expectation)
			}
		}
	}
}
