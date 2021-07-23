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
	"sort"
	"strconv"
	"strings"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	completionIndexEnvName = "JOB_COMPLETION_INDEX"
	unknownCompletionIndex = -1
)

func isIndexedJob(job *batch.Job) bool {
	return job.Spec.CompletionMode != nil && *job.Spec.CompletionMode == batch.IndexedCompletion
}

type interval struct {
	First int
	Last  int
}

type orderedIntervals []interval

// calculateSucceededIndexes returns the old and new list of succeeded indexes
// in compressed format (intervals).
// The old list is solely based off .status.completedIndexes, but returns an
// empty list if this Job is not tracked with finalizers. The new list includes
// the indexes that succeeded since the last sync.
func calculateSucceededIndexes(job *batch.Job, pods []*v1.Pod) (orderedIntervals, orderedIntervals) {
	var prevIntervals orderedIntervals
	withFinalizers := trackingUncountedPods(job)
	if withFinalizers {
		prevIntervals = succeededIndexesFromJob(job)
	}
	newSucceeded := sets.NewInt()
	for _, p := range pods {
		ix := getCompletionIndex(p.Annotations)
		// Succeeded Pod with valid index and, if tracking with finalizers,
		// has a finalizer (meaning that it is not counted yet).
		if p.Status.Phase == v1.PodSucceeded && ix != unknownCompletionIndex && ix < int(*job.Spec.Completions) && (!withFinalizers || hasJobTrackingFinalizer(p)) {
			newSucceeded.Insert(ix)
		}
	}
	// List returns the items of the set in order.
	result := prevIntervals.withOrderedIndexes(newSucceeded.List())
	return prevIntervals, result
}

// withOrderedIndexes returns a new list of ordered intervals that contains
// the newIndexes, provided in increasing order.
func (oi orderedIntervals) withOrderedIndexes(newIndexes []int) orderedIntervals {
	var result orderedIntervals
	i := 0
	j := 0
	var lastInterval *interval
	appendOrMergeWithLastInterval := func(thisInterval interval) {
		if lastInterval == nil || thisInterval.First > lastInterval.Last+1 {
			result = append(result, thisInterval)
			lastInterval = &result[len(result)-1]
		} else if lastInterval.Last < thisInterval.Last {
			lastInterval.Last = thisInterval.Last
		}
	}
	for i < len(oi) && j < len(newIndexes) {
		if oi[i].First < newIndexes[j] {
			appendOrMergeWithLastInterval(oi[i])
			i++
		} else {
			appendOrMergeWithLastInterval(interval{newIndexes[j], newIndexes[j]})
			j++
		}
	}
	for i < len(oi) {
		appendOrMergeWithLastInterval(oi[i])
		i++
	}
	for j < len(newIndexes) {
		appendOrMergeWithLastInterval(interval{newIndexes[j], newIndexes[j]})
		j++
	}
	return result
}

// total returns number of indexes contained in the intervals.
func (oi orderedIntervals) total() int {
	var count int
	for _, iv := range oi {
		count += iv.Last - iv.First + 1
	}
	return count
}

func (oi orderedIntervals) String() string {
	var builder strings.Builder
	for _, v := range oi {
		if builder.Len() > 0 {
			builder.WriteRune(',')
		}
		builder.WriteString(strconv.Itoa(v.First))
		if v.Last > v.First {
			if v.Last == v.First+1 {
				builder.WriteRune(',')
			} else {
				builder.WriteRune('-')
			}
			builder.WriteString(strconv.Itoa(v.Last))
		}
	}
	return builder.String()
}

func (oi orderedIntervals) has(ix int) bool {
	lo := 0
	hi := len(oi)
	// Invariant: oi[hi].Last >= ix
	for hi > lo {
		mid := lo + (hi-lo)/2
		if oi[mid].Last >= ix {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	if hi == len(oi) {
		return false
	}
	return oi[hi].First <= ix
}

func succeededIndexesFromJob(job *batch.Job) orderedIntervals {
	if job.Status.CompletedIndexes == "" {
		return nil
	}
	var result orderedIntervals
	var lastInterval *interval
	completions := int(*job.Spec.Completions)
	for _, intervalStr := range strings.Split(job.Status.CompletedIndexes, ",") {
		limitsStr := strings.Split(intervalStr, "-")
		var inter interval
		var err error
		inter.First, err = strconv.Atoi(limitsStr[0])
		if err != nil {
			klog.InfoS("Corrupted completed indexes interval, ignoring", "job", klog.KObj(job), "interval", intervalStr, "err", err)
			continue
		}
		if inter.First >= completions {
			break
		}
		if len(limitsStr) > 1 {
			inter.Last, err = strconv.Atoi(limitsStr[1])
			if err != nil {
				klog.InfoS("Corrupted completed indexes interval, ignoring", "job", klog.KObj(job), "interval", intervalStr, "err", err)
				continue
			}
			if inter.Last >= completions {
				inter.Last = completions - 1
			}
		} else {
			inter.Last = inter.First
		}
		if lastInterval != nil && lastInterval.Last == inter.First-1 {
			lastInterval.Last = inter.Last
		} else {
			result = append(result, inter)
			lastInterval = &result[len(result)-1]
		}
	}
	return result
}

// firstPendingIndexes returns `count` indexes less than `completions` that are
// not covered by `activePods` or `succeededIndexes`.
func firstPendingIndexes(activePods []*v1.Pod, succeededIndexes orderedIntervals, count, completions int) []int {
	if count == 0 {
		return nil
	}
	active := sets.NewInt()
	for _, p := range activePods {
		ix := getCompletionIndex(p.Annotations)
		if ix != unknownCompletionIndex {
			active.Insert(ix)
		}
	}
	result := make([]int, 0, count)
	nonPending := succeededIndexes.withOrderedIndexes(active.List())
	// The following algorithm is bounded by len(nonPending) and count.
	candidate := 0
	for _, sInterval := range nonPending {
		for ; candidate < completions && len(result) < count && candidate < sInterval.First; candidate++ {
			result = append(result, candidate)
		}
		if candidate < sInterval.Last+1 {
			candidate = sInterval.Last + 1
		}
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
