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

	"github.com/golang/glog"
	"github.com/robfig/cron"

	batchv1 "k8s.io/api/batch/v1"
	batchv2alpha1 "k8s.io/api/batch/v2alpha1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ref "k8s.io/client-go/tools/reference"
	"k8s.io/kubernetes/pkg/api"
)

// Utilities for dealing with Jobs and CronJobs and time.

func inActiveList(sj batchv2alpha1.CronJob, uid types.UID) bool {
	for _, j := range sj.Status.Active {
		if j.UID == uid {
			return true
		}
	}
	return false
}

func deleteFromActiveList(sj *batchv2alpha1.CronJob, uid types.UID) {
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
		glog.V(4).Infof("Job with non-CronJob parent, name %s namespace %s", j.Name, j.Namespace)
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
			glog.V(4).Infof("Unable to get parent uid from job %s in namespace %s", job.Name, job.Namespace)
			continue
		}
		jobsBySj[parentUID] = append(jobsBySj[parentUID], job)
	}
	return jobsBySj
}

// getNextStartTimeAfter gets the latest scheduled start time that is less than "now", or an error.
func getNextStartTimeAfter(schedule string, now time.Time) (time.Time, error) {
	// Using robfig/cron for cron scheduled parsing and next runtime
	// computation. Not using the entire library because:
	// - I want to detect when we missed a runtime due to being down.
	//   - How do I set the time such that I can detect the last known runtime?
	// - I guess the functions could launch a go-routine to start the job and
	// then return.
	// How to handle concurrency control.
	// How to detect changes to schedules or deleted schedules and then
	// update the jobs?
	sched, err := cron.Parse(schedule)
	if err != nil {
		return time.Unix(0, 0), fmt.Errorf("Unparseable schedule: %s : %s", schedule, err)
	}
	return sched.Next(now), nil
}

// getRecentUnmetScheduleTimes gets a slice of times (from oldest to latest) that have passed when a Job should have started but did not.
//
// If there are too many (>100) unstarted times, just give up and return an empty slice.
// If there were missed times prior to the last known start time, then those are not returned.
func getRecentUnmetScheduleTimes(sj batchv2alpha1.CronJob, now time.Time) ([]time.Time, error) {
	starts := []time.Time{}
	sched, err := cron.ParseStandard(sj.Spec.Schedule)
	if err != nil {
		return starts, fmt.Errorf("Unparseable schedule: %s : %s", sj.Spec.Schedule, err)
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
	if sj.Spec.StartingDeadlineSeconds != nil {
		// Controller is not going to schedule anything below this point
		schedulingDeadline := now.Add(-time.Second * time.Duration(*sj.Spec.StartingDeadlineSeconds))

		if schedulingDeadline.After(earliestTime) {
			earliestTime = schedulingDeadline
		}
	}
	if earliestTime.After(now) {
		return []time.Time{}, nil
	}

	for t := sched.Next(earliestTime); !t.After(now); t = sched.Next(t) {
		starts = append(starts, t)
		// An object might miss several starts.  For example, if
		// controller gets wedged on friday at 5:01pm when everyone has
		// gone home, and someone comes in on tuesday AM and discovers
		// the problem and restarts the controller, then all the hourly
		// jobs, more than 80 of them for one hourly scheduledJob, should
		// all start running with no further intervention (if the scheduledJob
		// allows concurrency and late starts).
		//
		// However, if there is a bug somewhere, or incorrect clock
		// on controller's server or apiservers (for setting creationTimestamp)
		// then there could be so many missed start times (it could be off
		// by decades or more), that it would eat up all the CPU and memory
		// of this controller. In that case, we want to not try to list
		// all the missed start times.
		//
		// I've somewhat arbitrarily picked 100, as more than 80, but
		// but less than "lots".
		if len(starts) > 100 {
			// We can't get the most recent times so just return an empty slice
			return []time.Time{}, fmt.Errorf("Too many missed start time (> 100). Set or decrease .spec.startingDeadlineSeconds or check clock skew.")
		}
	}
	return starts, nil
}

// XXX unit test this

// getJobFromTemplate makes a Job from a CronJob
func getJobFromTemplate(sj *batchv2alpha1.CronJob, scheduledTime time.Time) (*batchv1.Job, error) {
	// TODO: consider adding the following labels:
	// nominal-start-time=$RFC_3339_DATE_OF_INTENDED_START -- for user convenience
	// scheduled-job-name=$SJ_NAME -- for user convenience
	labels := copyLabels(&sj.Spec.JobTemplate)
	annotations := copyAnnotations(&sj.Spec.JobTemplate)
	createdByRefJson, err := makeCreatedByRefJson(sj)
	if err != nil {
		return nil, err
	}
	annotations[v1.CreatedByAnnotation] = string(createdByRefJson)
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
	if err := api.Scheme.Convert(&sj.Spec.JobTemplate.Spec, &job.Spec, nil); err != nil {
		return nil, fmt.Errorf("unable to convert job template: %v", err)
	}
	return job, nil
}

// Return Unix Epoch Time
func getTimeHash(scheduledTime time.Time) int64 {
	return scheduledTime.Unix()
}

// makeCreatedByRefJson makes a json string with an object reference for use in "created-by" annotation value
func makeCreatedByRefJson(object runtime.Object) (string, error) {
	createdByRef, err := ref.GetReference(api.Scheme, object)
	if err != nil {
		return "", fmt.Errorf("unable to get controller reference: %v", err)
	}

	// TODO: this code was not safe previously - as soon as new code came along that switched to v2, old clients
	//   would be broken upon reading it. This is explicitly hardcoded to v1 to guarantee predictable deployment.
	//   We need to consistently handle this case of annotation versioning.
	codec := api.Codecs.LegacyCodec(schema.GroupVersion{Group: v1.GroupName, Version: "v1"})

	createdByRefJson, err := runtime.Encode(codec, &v1.SerializedReference{
		Reference: *createdByRef,
	})
	if err != nil {
		return "", fmt.Errorf("unable to serialize controller reference: %v", err)
	}
	return string(createdByRefJson), nil
}

func getFinishedStatus(j *batchv1.Job) (bool, batchv1.JobConditionType) {
	for _, c := range j.Status.Conditions {
		if (c.Type == batchv1.JobComplete || c.Type == batchv1.JobFailed) && c.Status == v1.ConditionTrue {
			return true, c.Type
		}
	}
	return false, ""
}

func IsJobFinished(j *batchv1.Job) bool {
	isFinished, _ := getFinishedStatus(j)
	return isFinished
}

// byJobStartTime sorts a list of jobs by start timestamp, using their names as a tie breaker.
type byJobStartTime []batchv1.Job

func (o byJobStartTime) Len() int      { return len(o) }
func (o byJobStartTime) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byJobStartTime) Less(i, j int) bool {
	if o[j].Status.StartTime == nil {
		return o[i].Status.StartTime != nil
	}

	if (*o[i].Status.StartTime).Equal(*o[j].Status.StartTime) {
		return o[i].Name < o[j].Name
	}

	return (*o[i].Status.StartTime).Before(*o[j].Status.StartTime)
}
