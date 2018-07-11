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
	"strconv"

	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
)

func IsJobFinished(j *batch.Job) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func getCompletionsIndex(pod *v1.Pod) (int, error) {
	return strconv.Atoi(pod.Labels["job-completions-index"])
}

func addCompletionsIndexToPodTemplate(job *batch.Job, completionsIndex int) v1.PodTemplateSpec {
	if completionsIndex > 0 {
		job := job.DeepCopy()
		job.Spec.Template.Labels["job-completions-index"] = strconv.Itoa(completionsIndex)
	}
	return job.Spec.Template
}
