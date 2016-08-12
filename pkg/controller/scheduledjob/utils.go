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

package scheduledjob

import (
	"encoding/json"
	"fmt"
	"hash/adler32"
	"time"

	"github.com/golang/glog"
	"github.com/robfig/cron"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

// Utilities for dealing with Jobs and ScheduledJobs and time.

const (
	CreatedByAnnotation = "kubernetes.io/created-by"
)

func inActiveList(sj batch.ScheduledJob, uid types.UID) bool {
	for _, j := range sj.Status.Active {
		if j.UID == uid {
			return true
		}
	}
	return false
}

func deleteFromActiveList(sj *batch.ScheduledJob, uid types.UID) {
	if sj == nil {
		return
	}
	newActive := []api.ObjectReference{}
	for _, j := range sj.Status.Active {
		if j.UID != uid {
			newActive = append(newActive, j)
		}
	}
	sj.Status.Active = newActive
}

// getParentUIDFromJob extracts UID of job's parent and whether it was found
func getParentUIDFromJob(j batch.Job) (types.UID, bool) {
	creatorRefJson, found := j.ObjectMeta.Annotations[CreatedByAnnotation]
	if !found {
		glog.V(4).Infof("Job with no created-by annotation, name %s namespace %s", j.Name, j.Namespace)
		return types.UID(""), false
	}
	var sr api.SerializedReference
	err := json.Unmarshal([]byte(creatorRefJson), &sr)
	if err != nil {
		glog.V(4).Infof("Job with unparsable created-by annotation, name %s namespace %s: %v", j.Name, j.Namespace, err)
		return types.UID(""), false
	}
	if sr.Reference.Kind != "ScheduledJob" {
		glog.V(4).Infof("Job with non-ScheduledJob parent, name %s namespace %s", j.Name, j.Namespace)
		return types.UID(""), false
	}
	// Don't believe a job that claims to have a parent in a different namespace.
	if sr.Reference.Namespace != j.Namespace {
		glog.V(4).Infof("Alleged scheduledJob parent in different namespace (%s) from Job name %s namespace %s", sr.Reference.Namespace, j.Name, j.Namespace)
		return types.UID(""), false
	}

	return sr.Reference.UID, true
}

// groupJobsByParent groups jobs into a map keyed by the job parent UID (e.g. scheduledJob).
// It has no receiver, to facilitate testing.
func groupJobsByParent(sjs []batch.ScheduledJob, js []batch.Job) map[types.UID][]batch.Job {
	jobsBySj := make(map[types.UID][]batch.Job)
	for _, job := range js {
		parentUID, found := getParentUIDFromJob(job)
		if !found {
			glog.Errorf("Unable to get uid from job %s in namespace %s", job.Name, job.Namespace)
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
	tmpSched := addSeconds(schedule)
	sched, err := cron.Parse(tmpSched)
	if err != nil {
		return time.Unix(0, 0), fmt.Errorf("Unparseable schedule: %s : %s", schedule, err)
	}
	return sched.Next(now), nil
}

// getRecentUnmetScheduleTimes gets a slice of times (from oldest to latest) that have passed when a Job should have started but did not.
//
// If there are too many (>100) unstarted times, just give up and return an empty slice.
// If there were missed times prior to the last known start time, then those are not returned.
func getRecentUnmetScheduleTimes(sj batch.ScheduledJob, now time.Time) ([]time.Time, error) {
	starts := []time.Time{}
	tmpSched := addSeconds(sj.Spec.Schedule)
	sched, err := cron.Parse(tmpSched)
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
		// ScheduledJob as last known start time.
		earliestTime = sj.ObjectMeta.CreationTimestamp.Time
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
		// all the misseded start times.
		//
		// I've somewhat arbitrarily picked 100, as more than 80, but
		// but less than "lots".
		if len(starts) > 100 {
			// We can't get the most recent times so just return an empty slice
			return []time.Time{}, fmt.Errorf("Too many missed start times to list")
		}
	}
	return starts, nil
}

// TODO soltysh: this should be removed when https://github.com/robfig/cron/issues/58 is fixed
func addSeconds(schedule string) string {
	tmpSched := schedule
	if len(schedule) > 0 && schedule[0] != '@' {
		tmpSched = "0 " + schedule
	}
	return tmpSched
}

// XXX unit test this

// getJobFromTemplate makes a Job from a ScheduledJob
func getJobFromTemplate(sj *batch.ScheduledJob, scheduledTime time.Time) (*batch.Job, error) {
	// TODO: consider adding the following labels:
	// nominal-start-time=$RFC_3339_DATE_OF_INTENDED_START -- for user convenience
	// scheduled-job-name=$SJ_NAME -- for user convenience
	labels := copyLabels(&sj.Spec.JobTemplate)
	annotations := copyAnnotations(&sj.Spec.JobTemplate)
	createdByRefJson, err := makeCreatedByRefJson(sj)
	if err != nil {
		return nil, err
	}
	annotations[CreatedByAnnotation] = string(createdByRefJson)
	// We want job names for a given nominal start time to have a deterministic name to avoid the same job being created twice
	name := fmt.Sprintf("%s-%d", sj.Name, getTimeHash(scheduledTime))

	job := &batch.Job{
		ObjectMeta: api.ObjectMeta{
			Labels:      labels,
			Annotations: annotations,
			Name:        name,
		},
	}
	if err := api.Scheme.Convert(&sj.Spec.JobTemplate.Spec, &job.Spec); err != nil {
		return nil, fmt.Errorf("unable to convert job template: %v", err)
	}
	return job, nil
}

func getTimeHash(scheduledTime time.Time) uint32 {
	timeHasher := adler32.New()
	hashutil.DeepHashObject(timeHasher, scheduledTime)
	return timeHasher.Sum32()
}

// makeCreatedByRefJson makes a json string with an object reference for use in "created-by" annotation value
func makeCreatedByRefJson(object runtime.Object) (string, error) {
	createdByRef, err := api.GetReference(object)
	if err != nil {
		return "", fmt.Errorf("unable to get controller reference: %v", err)
	}

	// TODO: this code was not safe previously - as soon as new code came along that switched to v2, old clients
	//   would be broken upon reading it. This is explicitly hardcoded to v1 to guarantee predictable deployment.
	//   We need to consistently handle this case of annotation versioning.
	codec := api.Codecs.LegacyCodec(unversioned.GroupVersion{Group: api.GroupName, Version: "v1"})

	createdByRefJson, err := runtime.Encode(codec, &api.SerializedReference{
		Reference: *createdByRef,
	})
	if err != nil {
		return "", fmt.Errorf("unable to serialize controller reference: %v", err)
	}
	return string(createdByRefJson), nil
}
