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

	"k8s.io/kubernetes/pkg/api/v1"
	batch "k8s.io/kubernetes/pkg/apis/batch/v1"
)

func TestIsJobFinished(t *testing.T) {
	job := &batch.Job{
		Status: batch.JobStatus{
			Conditions: []batch.JobCondition{{
				Type:   batch.JobComplete,
				Status: v1.ConditionTrue,
			}},
		},
	}

	if !IsJobFinished(job) {
		t.Error("Job was expected to be finished")
	}

	job.Status.Conditions[0].Status = v1.ConditionFalse
	if IsJobFinished(job) {
		t.Error("Job was not expected to be finished")
	}

	job.Status.Conditions[0].Status = v1.ConditionUnknown
	if IsJobFinished(job) {
		t.Error("Job was not expected to be finished")
	}
}
