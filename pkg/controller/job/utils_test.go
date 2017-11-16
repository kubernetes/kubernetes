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

package job

import (
	"testing"

	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
)

func TestIsJobFinished(t *testing.T) {
	testCases := map[string]struct {
		conditionType        batch.JobConditionType
		conditionStatus      v1.ConditionStatus
		expectJobNotFinished bool
	}{
		"Job is completed and condition is true": {
			batch.JobComplete,
			v1.ConditionTrue,
			false,
		},
		"Job is completed and condition is false": {
			batch.JobComplete,
			v1.ConditionFalse,
			true,
		},
		"Job is completed and condition is unknown": {
			batch.JobComplete,
			v1.ConditionUnknown,
			true,
		},
		"Job is failed and condition is true": {
			batch.JobFailed,
			v1.ConditionTrue,
			false,
		},
		"Job is failed and condition is false": {
			batch.JobFailed,
			v1.ConditionFalse,
			true,
		},
		"Job is failed and condition is unknown": {
			batch.JobFailed,
			v1.ConditionUnknown,
			true,
		},
	}

	for name, tc := range testCases {
		job := &batch.Job{
			Status: batch.JobStatus{
				Conditions: []batch.JobCondition{{
					Type:   tc.conditionType,
					Status: tc.conditionStatus,
				}},
			},
		}

		if tc.expectJobNotFinished == IsJobFinished(job) {
			if tc.expectJobNotFinished {
				t.Errorf("test name: %s, job was not expected to be finished", name)
			} else {
				t.Errorf("test name: %s, job was expected to be finished", name)
			}
		}
	}
}
