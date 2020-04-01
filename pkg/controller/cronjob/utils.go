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

package cronjob

import (
	"fmt"
	"time"

	"github.com/robfig/cron"
	"k8s.io/klog"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// Utilities for dealing with Jobs and CronJobs and time.

func inActiveList(sj batchv1beta1.CronJob, uid types.UID) bool {
	for _, j := range sj.Status.Active {
		if j.UID == uid {
			return true
		}
	}
	return false
}

func deleteFromActiveList(sj *batchv1beta1.CronJob, uid types.UID) {
	if sj == nil {
		return
	}
	newActive := []v1.ObjectReference{}
	for _, j := range sj.Status.Active {
		if j.UID != uid {
			newActive = append(newActive, j)
		}
	}
	sj.Status.Active = newActive
}

// getParentUIDFromJob extracts UID of job's parent and whether it was found
func getParentUIDFromJob(j batchv1.Job) (types.UID, bool) {
	controllerRef := metav1.GetControllerOf(&j)

	if controllerRef == nil {
		return types.UID(""), false
	}

	if controllerRef.Kind != "CronJob" {
		klog.V(4).Infof("Job with non-CronJob parent, name %s namespace %s", j.Name, j.Namespace)
		return types.UID(""), false
	}

	return controllerRef.UID, true
}

// groupJobsByParent groups jobs into a map keyed by the job parent UID (e.g. scheduledJob).
// It has no receiver, to facilitate testing.
func groupJobsByParent(js []batchv1.Job) map[types.UID][]batchv1.Job {
	jobsBySj := make(map[types.UID][]batchv1.Job)
	for _, job := range js {
		parentUID, found := getParentUIDFromJob(job)
		if !found {
			klog.V(4).Infof("Unable to get parent uid from job %s in namespace %s", job.Name, job.Namespace)
			continue
		}
		jobsBySj[parentUID] = append(jobsBySj[parentUID], job)
	}
	return jobsBySj
}

// getRecentUnmetScheduleTimes gets a slice of times (from oldest to latest) that have passed when a Job should have started but did not.
//
// If there are too many (>100) unstarted times, just give up and return an empty slice.
// If there were missed times prior to the last known start time, then those are not returned.
func getRecentUnmetScheduleTimes(sj batchv1beta1.CronJob, now time.Time) ([]time.Time, error) {
	starts := make([]time.Time, 0)
	sched, err := cron.ParseStandard(sj.Spec.Schedule)
	if err != nil {
		return starts, fmt.Errorf("unparseable schedule: %s : %s", sj.Spec.Schedule, err)
	}

	var earliestTime time.Time
	if sj.Status.LastScheduleTime != nil {
		earliestTime = sj.Status.LastScheduleTime.Time
	} else {
		// If none found, then this is either a recently created scheduledJob,
		// or the active/completed info was somehow lost (contract for status
		// in kubernetes says it may need to be recreated), or that we have
		// started a job, but have not noticed it yet (distributed systems can
		// have arbitrary delays).  In any case, use the creation time of the
		// CronJob as last known start time.
		earliestTime = sj.ObjectMeta.CreationTimestamp.Time
	}
	if earliestTime.After(now) {
		return []time.Time{}, nil
	}

	// For the sake of easy patching, try to "hide" getLatestMissedSchedule
	latestTime, missedTimes := getLatestMissedSchedule(earliestTime, now, sched)
	for i := 0; i < missedTimes; i++ {
		starts = append(starts, latestTime) // This technically breaks the function signature, but the actual consumer only cares about starts[0] and len(starts).
	}

	return starts, nil
}

// getLatestMissedSchedule returns the latest start time in the time window (inclusive), and the number of schedules in the time window (capped at 2).
// The number of missed start times conveys "none, one, multiple".
func getLatestMissedSchedule(startWindow time.Time, endWindow time.Time, schedule cron.Schedule) (time.Time, int) {
	nextSchedule := schedule.Next(startWindow)

	// If no schedules in window, return.
	if nextSchedule.After(endWindow) {
		return time.Time{}, 0

		// There is (at least one) schedule in the window. See if there are more later ones.
	} else {
		latestSchedule, found := getLatestMissedScheduleBinarySearch(nextSchedule, endWindow, schedule)
		if found && latestSchedule != nextSchedule {
			return latestSchedule, 2 // We missed at least 2 schedules. Return the latest.
		} else {
			return nextSchedule, 1
		}
	}
}

// LatestScheduleTime, foundSchedule
// getLatestMissedSchedule returns the latest start time in the specified time window (inclusive). It also returns true if a start time was found, false otherwise
// It uses a binary search with the cron.Next function, to avoid stepping through every cron schedule.
// This is less efficient than if cron.Previous existed, but is arguably simpler to review and test than a custom Previous function.
func getLatestMissedScheduleBinarySearch(startWindow time.Time, endWindow time.Time, schedule cron.Schedule) (time.Time, bool) {
	// Crons can't schedule in lower granularity than a minute.
	// If we are looking at a time window less than a minute, there is "room" for at most 1 schedule, and we can skip the binary search.
	minimumTimeUnit := time.Minute
	if endWindow.Sub(startWindow) < minimumTimeUnit {
		nextStart := schedule.Next(startWindow)
		if !nextStart.After(endWindow) {
			return nextStart, true
		} else {
			return time.Time{}, false
		}
	}

	// Break the window into 2 halves for a binary search.
	midway := startWindow.Add(endWindow.Sub(startWindow) / 2)

	// Check the second half of the window for a start time.
	// If there is a start time within this window, we can safely ignore the first half of the window.
	nextScheduleAfterMidway := schedule.Next(midway)
	if !nextScheduleAfterMidway.After(endWindow) {
		latestFound, found := getLatestMissedScheduleBinarySearch(nextScheduleAfterMidway, endWindow, schedule) // Check a sub-window between the found schedule and the end of the window.
		if found {
			return latestFound, true
		} else {
			return nextScheduleAfterMidway, true
		}

		// If there was no start time in the second half of the window, we need to check the first half of the window.
	} else {

		nextScheduleInWindow := schedule.Next(startWindow)
		if !nextScheduleInWindow.After(midway) { // Latest start time is in first half. There's no start time in the second half.
			latestFound, found := getLatestMissedScheduleBinarySearch(nextScheduleInWindow, midway, schedule) // Check the first half.
			if found {
				return latestFound, true
			} else {
				return nextScheduleInWindow, true
			}

			// If there was no schedule time in the first half of the window either,
			// then there is no missed schedule.
		} else {
			return time.Time{}, false
		}

	}
}

// getJobFromTemplate makes a Job from a CronJob
func getJobFromTemplate(sj *batchv1beta1.CronJob, scheduledTime time.Time) (*batchv1.Job, error) {
	labels := copyLabels(&sj.Spec.JobTemplate)
	annotations := copyAnnotations(&sj.Spec.JobTemplate)
	// We want job names for a given nominal start time to have a deterministic name to avoid the same job being created twice
	name := fmt.Sprintf("%s-%d", sj.Name, getTimeHash(scheduledTime))

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Labels:          labels,
			Annotations:     annotations,
			Name:            name,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(sj, controllerKind)},
		},
	}
	sj.Spec.JobTemplate.Spec.DeepCopyInto(&job.Spec)
	return job, nil
}

// getTimeHash returns Unix Epoch Time
func getTimeHash(scheduledTime time.Time) int64 {
	return scheduledTime.Unix()
}

func getFinishedStatus(j *batchv1.Job) (bool, batchv1.JobConditionType) {
	for _, c := range j.Status.Conditions {
		if (c.Type == batchv1.JobComplete || c.Type == batchv1.JobFailed) && c.Status == v1.ConditionTrue {
			return true, c.Type
		}
	}
	return false, ""
}

// IsJobFinished returns whether or not a job has completed successfully or failed.
func IsJobFinished(j *batchv1.Job) bool {
	isFinished, _ := getFinishedStatus(j)
	return isFinished
}

// byJobStartTime sorts a list of jobs by start timestamp, using their names as a tie breaker.
type byJobStartTime []batchv1.Job

func (o byJobStartTime) Len() int      { return len(o) }
func (o byJobStartTime) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byJobStartTime) Less(i, j int) bool {
	if o[i].Status.StartTime == nil && o[j].Status.StartTime != nil {
		return false
	}
	if o[i].Status.StartTime != nil && o[j].Status.StartTime == nil {
		return true
	}
	if o[i].Status.StartTime.Equal(o[j].Status.StartTime) {
		return o[i].Name < o[j].Name
	}
	return o[i].Status.StartTime.Before(o[j].Status.StartTime)
}
