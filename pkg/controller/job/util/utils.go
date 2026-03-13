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
	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
)

// FinishedCondition returns true if a job is finished as well as the condition type indicating that.
// Returns false and no condition type otherwise
func FinishedCondition(j *batch.Job) (bool, batch.JobConditionType) {
	for _, c := range j.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
			return true, c.Type
		}
	}
	return false, ""
}

// IsJobFinished checks whether the given Job has finished execution.
// It does not discriminate between successful and failed terminations.
func IsJobFinished(j *batch.Job) bool {
	isFinished, _ := FinishedCondition(j)
	return isFinished
}

// IsJobSucceeded returns whether a job has completed successfully.
func IsJobSucceeded(j *batch.Job) bool {
	for _, c := range j.Status.Conditions {
		if c.Type == batch.JobComplete && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}
