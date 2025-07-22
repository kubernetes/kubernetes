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
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
)

const (
	completionIndexEnvName = "JOB_COMPLETION_INDEX"
	unknownCompletionIndex = -1
)

func isIndexedJob(job *batch.Job) bool {
	return job.Spec.CompletionMode != nil && *job.Spec.CompletionMode == batch.IndexedCompletion
}

func hasBackoffLimitPerIndex(job *batch.Job) bool {
	return feature.DefaultFeatureGate.Enabled(features.JobBackoffLimitPerIndex) && job.Spec.BackoffLimitPerIndex != nil
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
func calculateSucceededIndexes(logger klog.Logger, job *batch.Job, pods []*v1.Pod) (orderedIntervals, orderedIntervals) {
	prevIntervals := parseIndexesFromString(logger, job.Status.CompletedIndexes, int(*job.Spec.Completions))
	newSucceeded := sets.New[int]()
	for _, p := range pods {
		ix := getCompletionIndex(p.Annotations)
		// Succeeded Pod with valid index and, if tracking with finalizers,
		// has a finalizer (meaning that it is not counted yet).
		if p.Status.Phase == v1.PodSucceeded && ix != unknownCompletionIndex && ix < int(*job.Spec.Completions) && hasJobTrackingFinalizer(p) {
			newSucceeded.Insert(ix)
		}
	}
	// List returns the items of the set in order.
	result := prevIntervals.withOrderedIndexes(sets.List(newSucceeded))
	return prevIntervals, result
}

// calculateFailedIndexes returns the list of failed indexes in compressed
// format (intervals). The list includes indexes already present in
// .status.failedIndexes and indexes that failed since the last sync.
func calculateFailedIndexes(logger klog.Logger, job *batch.Job, pods []*v1.Pod) *orderedIntervals {
	var prevIntervals orderedIntervals
	if job.Status.FailedIndexes != nil {
		prevIntervals = parseIndexesFromString(logger, *job.Status.FailedIndexes, int(*job.Spec.Completions))
	}
	newFailed := sets.New[int]()
	for _, p := range pods {
		ix := getCompletionIndex(p.Annotations)
		// Failed Pod with valid index and has a finalizer (meaning that it is not counted yet).
		if ix != unknownCompletionIndex && ix < int(*job.Spec.Completions) && hasJobTrackingFinalizer(p) && isIndexFailed(logger, job, p) {
			newFailed.Insert(ix)
		}
	}
	// List returns the items of the set in order.
	result := prevIntervals.withOrderedIndexes(sets.List(newFailed))
	return &result
}

func isIndexFailed(logger klog.Logger, job *batch.Job, pod *v1.Pod) bool {
	isPodFailedCounted := false
	if isPodFailed(pod, job) {
		if job.Spec.PodFailurePolicy != nil {
			_, countFailed, action := matchPodFailurePolicy(job.Spec.PodFailurePolicy, pod)
			if action != nil && *action == batch.PodFailurePolicyActionFailIndex {
				return true
			}
			isPodFailedCounted = countFailed
		} else {
			isPodFailedCounted = true
		}
	}
	return isPodFailedCounted && getIndexFailureCount(logger, pod) >= *job.Spec.BackoffLimitPerIndex
}

// withOrderedIndexes returns a new list of ordered intervals that contains
// the newIndexes, provided in increasing order.
func (oi orderedIntervals) withOrderedIndexes(newIndexes []int) orderedIntervals {
	newIndexIntervals := make(orderedIntervals, len(newIndexes))
	for i, newIndex := range newIndexes {
		newIndexIntervals[i] = interval{newIndex, newIndex}
	}
	return oi.merge(newIndexIntervals)
}

// with returns a new list of ordered intervals that contains the newOrderedIntervals.
func (oi orderedIntervals) merge(newOi orderedIntervals) orderedIntervals {
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
	for i < len(oi) && j < len(newOi) {
		if oi[i].First < newOi[j].First {
			appendOrMergeWithLastInterval(oi[i])
			i++
		} else {
			appendOrMergeWithLastInterval(newOi[j])
			j++
		}
	}
	for i < len(oi) {
		appendOrMergeWithLastInterval(oi[i])
		i++
	}
	for j < len(newOi) {
		appendOrMergeWithLastInterval(newOi[j])
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

func parseIndexesFromString(logger klog.Logger, indexesStr string, completions int) orderedIntervals {
	if indexesStr == "" {
		return nil
	}
	var result orderedIntervals
	var lastInterval *interval
	for _, intervalStr := range strings.Split(indexesStr, ",") {
		limitsStr := strings.Split(intervalStr, "-")
		var inter interval
		var err error
		inter.First, err = strconv.Atoi(limitsStr[0])
		if err != nil {
			logger.Info("Corrupted indexes interval, ignoring", "interval", intervalStr, "err", err)
			continue
		}
		if inter.First >= completions {
			break
		}
		if len(limitsStr) > 1 {
			inter.Last, err = strconv.Atoi(limitsStr[1])
			if err != nil {
				logger.Info("Corrupted indexes interval, ignoring", "interval", intervalStr, "err", err)
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
// not covered by `activePods`, `succeededIndexes` or `failedIndexes`.
// In cases of PodReplacementPolicy as Failed we will include `terminatingPods` in this list.
func firstPendingIndexes(jobCtx *syncJobCtx, count, completions int) []int {
	if count == 0 {
		return nil
	}
	active := getIndexes(jobCtx.activePods)
	result := make([]int, 0, count)
	nonPending := jobCtx.succeededIndexes.withOrderedIndexes(sets.List(active))
	if onlyReplaceFailedPods(jobCtx.job) {
		terminating := getIndexes(controller.FilterTerminatingPods(jobCtx.pods))
		nonPending = nonPending.withOrderedIndexes(sets.List(terminating))
	}
	if jobCtx.failedIndexes != nil {
		nonPending = nonPending.merge(*jobCtx.failedIndexes)
	}
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

// Returns the list of indexes corresponding to the set of pods
func getIndexes(pods []*v1.Pod) sets.Set[int] {
	result := sets.New[int]()
	for _, p := range pods {
		ix := getCompletionIndex(p.Annotations)
		if ix != unknownCompletionIndex {
			result.Insert(ix)
		}
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

// getPodsWithDelayedDeletionPerIndex returns the pod which removal is delayed
// in order to await for recreation. This map is used when BackoffLimitPerIndex
// is enabled to delay pod finalizer removal, and thus pod deletion, until the
// replacement pod is created. The pod deletion is delayed so that the
// replacement pod can have the batch.kubernetes.io/job-index-failure-count
// annotation set properly keeping track of the number of failed pods within
// the index.
func getPodsWithDelayedDeletionPerIndex(logger klog.Logger, jobCtx *syncJobCtx) map[int]*v1.Pod {
	// the failed pods corresponding to currently active indexes can be safely
	// deleted as the failure count annotation is present in the currently
	// active pods.
	activeIndexes := getIndexes(jobCtx.activePods)

	podsWithDelayedDeletionPerIndex := make(map[int]*v1.Pod)
	getValidPodsWithFilter(jobCtx, nil, func(p *v1.Pod) bool {
		if isPodFailed(p, jobCtx.job) {
			if ix := getCompletionIndex(p.Annotations); ix != unknownCompletionIndex && ix < int(*jobCtx.job.Spec.Completions) {
				if jobCtx.succeededIndexes.has(ix) || jobCtx.failedIndexes.has(ix) || activeIndexes.Has(ix) {
					return false
				}
				if lastPodWithDelayedDeletion, ok := podsWithDelayedDeletionPerIndex[ix]; ok {
					if getIndexAbsoluteFailureCount(logger, lastPodWithDelayedDeletion) <= getIndexAbsoluteFailureCount(logger, p) && !getFinishedTime(p).Before(getFinishedTime(lastPodWithDelayedDeletion)) {
						podsWithDelayedDeletionPerIndex[ix] = p
					}
				} else {
					podsWithDelayedDeletionPerIndex[ix] = p
				}
			}
		}
		return false
	})
	return podsWithDelayedDeletionPerIndex
}

func addIndexFailureCountAnnotation(logger klog.Logger, template *v1.PodTemplateSpec, job *batch.Job, podBeingReplaced *v1.Pod) {
	indexFailureCount, indexIgnoredFailureCount := getNewIndexFailureCounts(logger, job, podBeingReplaced)
	template.Annotations[batch.JobIndexFailureCountAnnotation] = strconv.Itoa(int(indexFailureCount))
	if indexIgnoredFailureCount > 0 {
		template.Annotations[batch.JobIndexIgnoredFailureCountAnnotation] = strconv.Itoa(int(indexIgnoredFailureCount))
	}
}

// getNewIndexFailureCount returns the value of the index-failure-count
// annotation for the new pod being created
func getNewIndexFailureCounts(logger klog.Logger, job *batch.Job, podBeingReplaced *v1.Pod) (int32, int32) {
	if podBeingReplaced != nil {
		indexFailureCount := parseIndexFailureCountAnnotation(logger, podBeingReplaced)
		indexIgnoredFailureCount := parseIndexFailureIgnoreCountAnnotation(logger, podBeingReplaced)
		if job.Spec.PodFailurePolicy != nil {
			_, countFailed, _ := matchPodFailurePolicy(job.Spec.PodFailurePolicy, podBeingReplaced)
			if countFailed {
				indexFailureCount++
			} else {
				indexIgnoredFailureCount++
			}
		} else {
			indexFailureCount++
		}
		return indexFailureCount, indexIgnoredFailureCount
	}
	return 0, 0
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

// getIndexFailureCount returns the value of the batch.kubernetes.io/job-index-failure-count
// annotation as int32. It fallbacks to 0 when:
//   - there is no annotation - for example the pod was created when the BackoffLimitPerIndex
//     feature was temporarily disabled, or the annotation was manually removed by the user,
//   - the value of the annotation isn't parsable as int - for example because
//     it was set by a malicious user,
//   - the value of the annotation is negative or greater by int32 - for example
//     because it was set by a malicious user.
func getIndexFailureCount(logger klog.Logger, pod *v1.Pod) int32 {
	return parseIndexFailureCountAnnotation(logger, pod)
}

func getIndexAbsoluteFailureCount(logger klog.Logger, pod *v1.Pod) int32 {
	return parseIndexFailureCountAnnotation(logger, pod) + parseIndexFailureIgnoreCountAnnotation(logger, pod)
}

func parseIndexFailureCountAnnotation(logger klog.Logger, pod *v1.Pod) int32 {
	if value, ok := pod.Annotations[batch.JobIndexFailureCountAnnotation]; ok {
		return parseInt32(logger, value)
	}
	logger.V(3).Info("There is no expected annotation", "annotationKey", batch.JobIndexFailureCountAnnotation, "pod", klog.KObj(pod), "podUID", pod.UID)
	return 0
}

func parseIndexFailureIgnoreCountAnnotation(logger klog.Logger, pod *v1.Pod) int32 {
	if value, ok := pod.Annotations[batch.JobIndexIgnoredFailureCountAnnotation]; ok {
		return parseInt32(logger, value)
	}
	return 0
}

func parseInt32(logger klog.Logger, vStr string) int32 {
	if vInt, err := strconv.Atoi(vStr); err != nil {
		logger.Error(err, "Failed to parse the value", "value", vStr)
		return 0
	} else if vInt < 0 || vInt > math.MaxInt32 {
		logger.Info("The value is invalid", "value", vInt)
		return 0
	} else {
		return int32(vInt)
	}
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
	var fieldPath string
	if feature.DefaultFeatureGate.Enabled(features.PodIndexLabel) {
		fieldPath = fmt.Sprintf("metadata.labels['%s']", batch.JobCompletionIndexAnnotation)
	} else {
		fieldPath = fmt.Sprintf("metadata.annotations['%s']", batch.JobCompletionIndexAnnotation)
	}
	container.Env = append(container.Env, v1.EnvVar{
		Name: completionIndexEnvName,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: fieldPath,
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

func addCompletionIndexLabel(template *v1.PodTemplateSpec, index int) {
	if template.Labels == nil {
		template.Labels = make(map[string]string, 1)
	}
	// For consistency, we use the annotation batch.kubernetes.io/job-completion-index for the corresponding label as well.
	template.Labels[batch.JobCompletionIndexAnnotation] = strconv.Itoa(index)
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

func completionModeStr(job *batch.Job) string {
	if job.Spec.CompletionMode != nil {
		return string(*job.Spec.CompletionMode)
	}
	return string(batch.NonIndexedCompletion)
}
