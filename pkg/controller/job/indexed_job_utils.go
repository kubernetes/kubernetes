/*
Copyright 2021 The Kubernetes Authors.

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
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	completionIndexEnvName = "JOB_COMPLETION_INDEX"
	unknownCompletionIndex = -1
)

func isIndexedJob(job *batch.Job) bool {
	return job.Spec.CompletionMode != nil && *job.Spec.CompletionMode == batch.IndexedCompletion
}

// calculateSucceededIndexes returns a string representation of the list of
// succeeded indexes in compressed format and the number of succeeded indexes.
func calculateSucceededIndexes(pods []*v1.Pod, completions int32) (string, int32) {
	sort.Sort(byCompletionIndex(pods))
	var result strings.Builder
	var lastSucceeded int
	var count int32
	firstSucceeded := math.MinInt32
	for _, p := range pods {
		ix := getCompletionIndex(p.Annotations)
		if ix == unknownCompletionIndex {
			continue
		}
		if ix >= int(completions) {
			break
		}
		if p.Status.Phase == v1.PodSucceeded {
			if firstSucceeded == math.MinInt32 {
				firstSucceeded = ix
			} else if ix > lastSucceeded+1 {
				addSingleOrRangeStr(&result, firstSucceeded, lastSucceeded)
				count += int32(lastSucceeded - firstSucceeded + 1)
				firstSucceeded = ix
			}
			lastSucceeded = ix
		}
	}
	if firstSucceeded != math.MinInt32 {
		addSingleOrRangeStr(&result, firstSucceeded, lastSucceeded)
		count += int32(lastSucceeded - firstSucceeded + 1)
	}
	return result.String(), count
}

func addSingleOrRangeStr(builder *strings.Builder, first, last int) {
	if builder.Len() > 0 {
		builder.WriteRune(',')
	}
	builder.WriteString(strconv.Itoa(first))
	if last > first {
		if last == first+1 {
			builder.WriteRune(',')
		} else {
			builder.WriteRune('-')
		}
		builder.WriteString(strconv.Itoa(last))
	}
}

// firstPendingIndexes returns `count` indexes less than `completions` that are
// not covered by running or succeeded pods.
func firstPendingIndexes(pods []*v1.Pod, count, completions int) []int {
	if count == 0 {
		return nil
	}
	nonPending := sets.NewInt()
	for _, p := range pods {
		if p.Status.Phase == v1.PodSucceeded || controller.IsPodActive(p) {
			ix := getCompletionIndex(p.Annotations)
			if ix != unknownCompletionIndex {
				nonPending.Insert(ix)
			}
		}
	}
	result := make([]int, 0, count)
	// The following algorithm is bounded by the number of non pending pods and
	// parallelism.
	// TODO(#99368): Convert the list of non-pending pods into a set of
	// non-pending intervals from the job's .status.completedIndexes and active
	// pods.
	candidate := 0
	for _, np := range nonPending.List() {
		for ; candidate < np && candidate < completions; candidate++ {
			result = append(result, candidate)
			if len(result) == count {
				return result
			}
		}
		candidate = np + 1
	}
	for ; candidate < completions && len(result) < count; candidate++ {
		result = append(result, candidate)
	}
	return result
}

// appendDuplicatedIndexPodsForRemoval scans active `pods` for duplicated
// completion indexes. For each index, it selects n-1 pods for removal, where n
// is the number of repetitions. The pods to be removed are appended to `rm`,
// while the remaining pods are appended to `left`.
// All pods that don't have a completion index are appended to `rm`.
// All pods with index not in valid range are appended to `rm`.
func appendDuplicatedIndexPodsForRemoval(rm, left, pods []*v1.Pod, completions int) ([]*v1.Pod, []*v1.Pod) {
	sort.Sort(byCompletionIndex(pods))
	lastIndex := unknownCompletionIndex
	firstRepeatPos := 0
	countLooped := 0
	for i, p := range pods {
		ix := getCompletionIndex(p.Annotations)
		if ix >= completions {
			rm = append(rm, pods[i:]...)
			break
		}
		if ix != lastIndex {
			rm, left = appendPodsWithSameIndexForRemovalAndRemaining(rm, left, pods[firstRepeatPos:i], lastIndex)
			firstRepeatPos = i
			lastIndex = ix
		}
		countLooped += 1
	}
	return appendPodsWithSameIndexForRemovalAndRemaining(rm, left, pods[firstRepeatPos:countLooped], lastIndex)
}

func appendPodsWithSameIndexForRemovalAndRemaining(rm, left, pods []*v1.Pod, ix int) ([]*v1.Pod, []*v1.Pod) {
	if ix == unknownCompletionIndex {
		rm = append(rm, pods...)
		return rm, left
	}
	if len(pods) == 1 {
		left = append(left, pods[0])
		return rm, left
	}
	sort.Sort(controller.ActivePods(pods))
	rm = append(rm, pods[:len(pods)-1]...)
	left = append(left, pods[len(pods)-1])
	return rm, left
}

func getCompletionIndex(annotations map[string]string) int {
	if annotations == nil {
		return unknownCompletionIndex
	}
	v, ok := annotations[batch.JobCompletionIndexAnnotation]
	if !ok {
		return unknownCompletionIndex
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		return unknownCompletionIndex
	}
	if i < 0 {
		return unknownCompletionIndex
	}
	return i
}

func addCompletionIndexEnvVariables(template *v1.PodTemplateSpec) {
	for i := range template.Spec.InitContainers {
		addCompletionIndexEnvVariable(&template.Spec.InitContainers[i])
	}
	for i := range template.Spec.Containers {
		addCompletionIndexEnvVariable(&template.Spec.Containers[i])
	}
}

func addCompletionIndexEnvVariable(container *v1.Container) {
	for _, v := range container.Env {
		if v.Name == completionIndexEnvName {
			return
		}
	}
	container.Env = append(container.Env, v1.EnvVar{
		Name: completionIndexEnvName,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: fmt.Sprintf("metadata.annotations['%s']", batch.JobCompletionIndexAnnotation),
			},
		},
	})
}

func addCompletionIndexAnnotation(template *v1.PodTemplateSpec, index int) {
	if template.Annotations == nil {
		template.Annotations = make(map[string]string, 1)
	}
	template.Annotations[batch.JobCompletionIndexAnnotation] = strconv.Itoa(index)
}

func podGenerateNameWithIndex(jobName string, index int) string {
	appendIndex := "-" + strconv.Itoa(index) + "-"
	generateNamePrefix := jobName + appendIndex
	if len(generateNamePrefix) > names.MaxGeneratedNameLength {
		generateNamePrefix = generateNamePrefix[:names.MaxGeneratedNameLength-len(appendIndex)] + appendIndex
	}
	return generateNamePrefix
}

type byCompletionIndex []*v1.Pod

func (bci byCompletionIndex) Less(i, j int) bool {
	return getCompletionIndex(bci[i].Annotations) < getCompletionIndex(bci[j].Annotations)
}

func (bci byCompletionIndex) Swap(i, j int) {
	bci[i], bci[j] = bci[j], bci[i]
}

func (bci byCompletionIndex) Len() int {
	return len(bci)
}
