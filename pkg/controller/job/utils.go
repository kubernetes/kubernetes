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

const (
	CompletionsIndexName       = "job-completions-index"
	CompletionsIndexEnvArgName = "JOB_COMPLETIONS_INDEX"
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
	return strconv.Atoi(pod.Labels[CompletionsIndexName])
}

func addCompletionsIndexToPodTemplate(job *batch.Job, completionsIndex int32) v1.PodTemplateSpec {
	template := job.Spec.Template
	if completionsIndex <= 0 {
		return template
	}
	job = job.DeepCopy()
	template.Labels[CompletionsIndexName] = strconv.Itoa(int(completionsIndex))
	addEnvFunc := func(containers []v1.Container) {
		if containers == nil {
			return
		}
		for i := range containers {
			container := containers[i]
			container.Env = append(container.Env, v1.EnvVar{
				Name:  CompletionsIndexEnvArgName,
				Value: strconv.Itoa(int(completionsIndex)),
			})
			containers[i] = container
		}
	}
	addEnvFunc(template.Spec.Containers)
	addEnvFunc(template.Spec.InitContainers)
	return template
}

func getLen(pods []*v1.Pod, distinct bool) int32 {
	if distinct {
		return distinctPods(pods)
	}
	return int32(len(pods))
}

// distinctPods get distinct pod number
func distinctPods(pods []*v1.Pod) int32 {
	keyMap := make(map[int]bool, 0)
	for i := range pods {
		index, err := getCompletionsIndex(pods[i])
		if err != nil {
			continue
		}
		keyMap[index] = true
	}
	return int32(len(keyMap))
}

func getAvailableCompletionsIndexes(activePods, succeededPods []*v1.Pod, completions int32) []int32 {
	podsHasIndexes := func() map[int]bool {
		hasIndexes := make(map[int]bool, len(activePods)+len(succeededPods))
		for i := range succeededPods {
			if index, err := getCompletionsIndex(succeededPods[i]); err == nil {
				hasIndexes[index] = true
			}
		}
		for i := range activePods {
			if index, err := getCompletionsIndex(activePods[i]); err == nil {
				hasIndexes[index] = true
			}
		}
		return hasIndexes
	}()
	availableCompletionsIndexes := make([]int32, 0, completions)
	for i := 1; i <= int(completions); i++ {
		if _, ok := podsHasIndexes[i]; !ok {
			availableCompletionsIndexes = append(availableCompletionsIndexes, int32(i))
		}
	}
	return availableCompletionsIndexes
}

// getNeedStopActivePods find succeeded index in active pods and active pods duplicate index pods
func getNeedStopActivePods(activePods, succeededPods []*v1.Pod) []*v1.Pod {
	duplicateIndexActivePods := make([]*v1.Pod, 0)
	succeededPodsHasIndexes, _ := getCompletionsIndexesAndDuplicateIndexPods(succeededPods)
	for i := range activePods {
		if index, err := getCompletionsIndex(activePods[i]); err == nil {
			if _, ok := succeededPodsHasIndexes[index]; ok {
				duplicateIndexActivePods = append(duplicateIndexActivePods, activePods[i])
			}
		} else {
			// if label cont't switch to int, stop
			duplicateIndexActivePods = append(duplicateIndexActivePods, activePods[i])
		}
	}
	_, duplicateIndexPods := getCompletionsIndexesAndDuplicateIndexPods(activePods)
	return append(duplicateIndexActivePods, duplicateIndexPods...)
}

// getCompletionsIndexesAndDuplicateIndexPods get all index in pods and get duplicate index pods
func getCompletionsIndexesAndDuplicateIndexPods(pods []*v1.Pod) (map[int]bool, []*v1.Pod) {
	hasIndexes := make(map[int]bool, len(pods))
	duplicateIndexPods := make([]*v1.Pod, 0)
	for i := range pods {
		if index, err := getCompletionsIndex(pods[i]); err == nil {
			if _, ok := hasIndexes[index]; ok {
				duplicateIndexPods = append(duplicateIndexPods, pods[i])
			}
			hasIndexes[index] = true

		}
	}
	return hasIndexes, duplicateIndexPods
}
