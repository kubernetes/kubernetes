/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"testing"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
)

func TestFinishedCondition(t *testing.T) {
	tests := map[string]struct {
		conditions        []batch.JobCondition
		wantJobFinished   bool
		wantConditionType batch.JobConditionType
	}{
		"Job doesn't have any conditions": {
			wantJobFinished:   false,
			wantConditionType: "",
		},
		"Job is completed and condition is true": {
			conditions: []batch.JobCondition{
				{
					Type:   batch.JobComplete,
					Status: v1.ConditionTrue,
				},
			},
			wantJobFinished:   true,
			wantConditionType: batch.JobComplete,
		},
		"Job is completed and condition is false": {
			conditions: []batch.JobCondition{
				{
					Type:   batch.JobComplete,
					Status: v1.ConditionFalse,
				},
			},
			wantJobFinished:   false,
			wantConditionType: "",
		},
		"Job is completed and condition is unknown": {
			conditions: []batch.JobCondition{
				{
					Type:   batch.JobComplete,
					Status: v1.ConditionUnknown,
				},
			},
			wantJobFinished:   false,
			wantConditionType: "",
		},
		"Job has multiple conditions, one of them being complete and condition true": {
			conditions: []batch.JobCondition{
				{
					Type:   batch.JobSuspended,
					Status: v1.ConditionFalse,
				},
				{
					Type:   batch.JobComplete,
					Status: v1.ConditionTrue,
				},
				{
					Type:   batch.JobFailed,
					Status: v1.ConditionFalse,
				},
			},
			wantJobFinished:   true,
			wantConditionType: batch.JobComplete,
		},
		"Job is failed and condition is true": {
			conditions: []batch.JobCondition{
				{
					Type:   batch.JobFailed,
					Status: v1.ConditionTrue,
				},
			},
			wantJobFinished:   true,
			wantConditionType: batch.JobFailed,
		},
		"Job is failed and condition is false": {
			conditions: []batch.JobCondition{
				{
					Type:   batch.JobFailed,
					Status: v1.ConditionFalse,
				},
			},
			wantJobFinished:   false,
			wantConditionType: "",
		},
		"Job is failed and condition is unknown": {
			conditions: []batch.JobCondition{
				{
					Type:   batch.JobFailed,
					Status: v1.ConditionUnknown,
				},
			},
			wantJobFinished:   false,
			wantConditionType: "",
		},
		"Job has multiple conditions, none of them has condition true": {
			conditions: []batch.JobCondition{
				{
					Type:   batch.JobSuspended,
					Status: v1.ConditionFalse,
				},
				{
					Type:   batch.JobComplete,
					Status: v1.ConditionFalse,
				},
				{
					Type:   batch.JobFailed,
					Status: v1.ConditionFalse,
				},
			},
			wantJobFinished:   false,
			wantConditionType: "",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			job := &batch.Job{
				Status: batch.JobStatus{
					Conditions: test.conditions,
				},
			}

			isJobFinished, conditionType := FinishedCondition(job)
			if isJobFinished != test.wantJobFinished {
				if test.wantJobFinished {
					t.Error("Expected the job to be finished")
				} else {
					t.Error("Expected the job to be unfinished")
				}
			}

			if conditionType != test.wantConditionType {
				t.Errorf("Unexpected job condition type. got: '%v', want: '%v'", conditionType, test.wantConditionType)
			}
		})
	}
}

func TestIsJobSucceeded(t *testing.T) {
	tests := map[string]struct {
		job        batch.Job
		wantResult bool
	}{
		"job doesn't have any conditions": {
			wantResult: false,
		},
		"job has Complete=True condition": {
			job: batch.Job{
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobSuspended,
							Status: v1.ConditionFalse,
						},
						{
							Type:   batch.JobComplete,
							Status: v1.ConditionTrue,
						},
					},
				},
			},
			wantResult: true,
		},
		"job has Complete=False condition": {
			job: batch.Job{
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobFailed,
							Status: v1.ConditionTrue,
						},
						{
							Type:   batch.JobComplete,
							Status: v1.ConditionFalse,
						},
					},
				},
			},
			wantResult: false,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			gotResult := IsJobSucceeded(&tc.job)
			if tc.wantResult != gotResult {
				t.Errorf("unexpected result, want=%v, got=%v", tc.wantResult, gotResult)
			}
		})
	}
}
