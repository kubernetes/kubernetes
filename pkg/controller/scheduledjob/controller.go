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

/*
I did not use watch or expectations.  Those add a lot of corner cases, and we aren't
expecting a large volume of jobs or scheduledJobs.  (We are favoring correctness
over scalability.  If we find a single controller thread is too slow because
there are a lot of Jobs or ScheduledJobs, we we can parallelize by Namespace.
If we find the load on the API server is too high, we can use a watch and
UndeltaStore.)

Just periodically list jobs and SJs, and then reconcile them.

*/

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
)

// Utilities for dealing with Jobs and ScheduledJobs and time.

type ScheduledJobController struct {
	kubeClient *client.Client
	jobControl jobControlInterface
	sjControl  sjControlInterface
	podControl podControlInterface
	recorder   record.EventRecorder
}

func NewScheduledJobController(kubeClient *client.Client) *ScheduledJobController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	if kubeClient != nil && kubeClient.GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("scheduledjob_controller", kubeClient.GetRateLimiter())
	}

	jm := &ScheduledJobController{
		kubeClient: kubeClient,
		jobControl: realJobControl{KubeClient: kubeClient},
		sjControl:  &realSJControl{KubeClient: kubeClient},
		podControl: &realPodControl{KubeClient: kubeClient},
		recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduled-job-controller"}),
	}

	return jm
}

func NewScheduledJobControllerFromClient(kubeClient *client.Client) *ScheduledJobController {
	jm := NewScheduledJobController(kubeClient)
	return jm
}

// Run the main goroutine responsible for watching and syncing jobs.
func (jm *ScheduledJobController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	glog.Infof("Starting ScheduledJob Manager")
	// Check things every 10 second.
	go wait.Until(jm.SyncAll, 10*time.Second, stopCh)
	<-stopCh
	glog.Infof("Shutting down ScheduledJob Manager")
}

// SyncAll lists all the ScheduledJobs and Jobs and reconciles them.
func (jm *ScheduledJobController) SyncAll() {
	sjl, err := jm.kubeClient.Batch().ScheduledJobs(api.NamespaceAll).List(api.ListOptions{})
	if err != nil {
		glog.Errorf("Error listing scheduledjobs: %v", err)
		return
	}
	sjs := sjl.Items
	glog.Infof("Found %d scheduledjobs", len(sjs))

	jl, err := jm.kubeClient.Batch().Jobs(api.NamespaceAll).List(api.ListOptions{})
	if err != nil {
		glog.Errorf("Error listing jobs")
		return
	}
	js := jl.Items
	glog.Infof("Found %d jobs", len(js))

	jobsBySj := groupJobsByParent(sjs, js)
	glog.Infof("Found %d groups", len(jobsBySj))

	for _, sj := range sjs {
		SyncOne(sj, jobsBySj[sj.UID], time.Now(), jm.jobControl, jm.sjControl, jm.podControl, jm.recorder)
	}
}

// SyncOne reconciles a ScheduledJob with a list of any Jobs that it created.
// All known jobs created by "sj" should be included in "js".
// The current time is passed in to facilitate testing.
// It has no receiver, to facilitate testing.
func SyncOne(sj batch.ScheduledJob, js []batch.Job, now time.Time, jc jobControlInterface, sjc sjControlInterface, pc podControlInterface, recorder record.EventRecorder) {
	nameForLog := fmt.Sprintf("%s/%s", sj.Namespace, sj.Name)

	for _, j := range js {
		found := inActiveList(sj, j.ObjectMeta.UID)
		if !found {
			recorder.Eventf(&sj, api.EventTypeWarning, "UnexpectedJob", "Saw a job that the controller did not create or forgot: %v", j.Name)
			// We found a job object that has us as the parent, but it is not in our Active list.
			// This could happen if we crashed right after creating the Job and before updating the status,
			// or if our jobs list is newer than our sj status after a relist, or if someone intentionally created
			// a job that they wanted us to adopt.

			// TODO: maybe handle the adoption case?  Concurrency/suspend rules will not apply in that case, obviously, since we can't
			// stop users from creating jobs if they have permission.  It is assumed that if a
			// user has permission to create a job within a namespace, then they have permission to make any scheduledJob
			// in the same namespace "adopt" that job.  ReplicaSets and their Pods work the same way.
			// TBS: how to update sj.Status.LastScheduleTime if the adopted job is newer than any we knew about?
		} else {
			if job.IsJobFinished(&j) {
				deleteFromActiveList(&sj, j.ObjectMeta.UID)
				// TODO: event to call out failure vs success.
				recorder.Eventf(&sj, api.EventTypeNormal, "SawCompletedJob", "Saw completed job: %v", j.Name)
			}
		}
	}
	err := sjc.UpdateStatus(&sj)
	if err != nil {
		glog.Errorf("Unable to update status for %s: %v", nameForLog, err)
	}

	if sj.Spec.Suspend != nil && *sj.Spec.Suspend {
		glog.V(4).Infof("Not starting job for %s because it is suspended", nameForLog)
		return
	}
	times, err := getRecentUnmetScheduleTimes(sj, now)
	if err != nil {
		glog.Errorf("Cannot determine if %s needs to be started: %v", nameForLog, err)
	}
	// TODO: handle multiple unmet start times, from oldest to newest, updating status as needed.
	if len(times) == 0 {
		glog.V(4).Infof("No unmet start times for %s", nameForLog)
		return
	}
	if len(times) > 1 {
		glog.Errorf("Multiple unmet start times for %s so only starting last one", nameForLog)
	}
	scheduledTime := times[len(times)-1]
	tooLate := false
	if sj.Spec.StartingDeadlineSeconds != nil {
		tooLate = scheduledTime.Add(time.Second * time.Duration(*sj.Spec.StartingDeadlineSeconds)).Before(now)
	}
	if tooLate {
		glog.Errorf("Missed starting window for %s", nameForLog)
		// TODO: generate an event for a miss.  Use a warning level event because it indicates a
		// problem with the controller (restart or long queue), and is not expected by user either.
		// Since we don't set LastScheduleTime when not scheduling, we are going to keep noticing
		// the miss every cycle.  In order to avoid sending multiple events, and to avoid processing
		// the sj again and again, we could set a Status.LastMissedTime when we notice a miss.
		// Then, when we call getRecentUnmetScheduleTimes, we can take max(creationTimestamp,
		// Status.LastScheduleTime, Status.LastMissedTime), and then so we won't generate
		// and event the next time we process it, and also so the user looking at the status
		// can see easily that there was a missed execution.
		return
	}
	if sj.Spec.ConcurrencyPolicy == batch.ForbidConcurrent && len(sj.Status.Active) > 0 {
		// Regardless which source of information we use for the set of active jobs,
		// there is some risk that we won't see an active job when there is one.
		// (because we haven't seen the status update to the SJ or the created pod).
		// So it is theoretically possible to have concurrency with Forbid.
		// As long the as the invokations are "far enough apart in time", this usually won't happen.
		//
		// TODO: for Forbid, we could use the same name for every execution, as a lock.
		// With replace, we could use a name that is deterministic per execution time.
		// But that would mean that you could not inspect prior successes or failures of Forbid jobs.
		glog.V(4).Infof("Not starting job for %s because of prior execution still running and concurrency policy is Forbid.", nameForLog)
		return
	}
	if sj.Spec.ConcurrencyPolicy == batch.ReplaceConcurrent {
		for _, j := range sj.Status.Active {
			// TODO: this should be replaced with server side job deletion
			// currently this mimics JobReaper from pkg/kubectl/stop.go
			glog.V(4).Infof("Deleting job %s of %s s that was still running at next scheduled start time", j.Name, nameForLog)
			job, err := jc.GetJob(j.Namespace, j.Name)
			if err != nil {
				recorder.Eventf(&sj, api.EventTypeWarning, "FailedGet", "Get job: %v", err)
				return
			}
			// scale job down to 0
			if *job.Spec.Parallelism != 0 {
				zero := int32(0)
				job.Spec.Parallelism = &zero
				job, err = jc.UpdateJob(job.Namespace, job)
				if err != nil {
					recorder.Eventf(&sj, api.EventTypeWarning, "FailedUpdate", "Update job: %v", err)
					return
				}
			}
			// remove all pods...
			selector, _ := unversioned.LabelSelectorAsSelector(job.Spec.Selector)
			options := api.ListOptions{LabelSelector: selector}
			podList, err := pc.ListPods(job.Namespace, options)
			if err != nil {
				recorder.Eventf(&sj, api.EventTypeWarning, "FailedList", "List job-pods: %v", err)
			}
			errList := []error{}
			for _, pod := range podList.Items {
				if err := pc.DeletePod(pod.Namespace, pod.Name); err != nil {
					// ignores the error when the pod isn't found
					if !errors.IsNotFound(err) {
						errList = append(errList, err)
					}
				}
			}
			if len(errList) != 0 {
				recorder.Eventf(&sj, api.EventTypeWarning, "FailedDelete", "Deleted job-pods: %v", utilerrors.NewAggregate(errList))
				return
			}
			// ... and the job itself
			if err := jc.DeleteJob(job.Namespace, job.Name); err != nil {
				recorder.Eventf(&sj, api.EventTypeWarning, "FailedDelete", "Deleted job: %v", err)
				return
			}
			recorder.Eventf(&sj, api.EventTypeNormal, "SuccessfulDelete", "Deleted job %v", j.Name)
		}
	}

	jobReq, err := getJobFromTemplate(&sj, scheduledTime)
	if err != nil {
		glog.Errorf("Unable to make Job from template in %s: %v", nameForLog, err)
		return
	}
	jobResp, err := jc.CreateJob(sj.Namespace, jobReq)
	if err != nil {
		recorder.Eventf(&sj, api.EventTypeWarning, "FailedCreate", "Error creating job: %v", err)
		return
	}
	recorder.Eventf(&sj, api.EventTypeNormal, "SuccessfulCreate", "Created job %v", jobResp.Name)

	// ------------------------------------------------------------------ //

	// If this process restarts at this point (after posting a job, but
	// before updating the status), then we might try to start the job on
	// the next time.  Actually, if we relist the SJs and Jobs on the next
	// iteration of SyncAll, we might not see our own status update, and
	// then post one again.  So, we need to use the job name as a lock to
	// prevent us from making the job twice (name the job with hash of its
	// scheduled time).

	// Add the just-started job to the status list.
	ref, err := getRef(jobResp)
	if err != nil {
		glog.V(2).Infof("Unable to make object reference for job for %s", nameForLog)
	} else {
		sj.Status.Active = append(sj.Status.Active, *ref)
	}
	sj.Status.LastScheduleTime = &unversioned.Time{Time: scheduledTime}
	if err := sjc.UpdateStatus(&sj); err != nil {
		glog.Infof("Unable to update status for %s: %v", nameForLog, err)
	}

	return
}

func getRef(object runtime.Object) (*api.ObjectReference, error) {
	return api.GetReference(object)
}
