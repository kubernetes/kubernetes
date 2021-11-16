/*
Copyright 2020 The Kubernetes Authors.

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
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/robfig/cron/v3"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	batchv1informers "k8s.io/client-go/informers/batch/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	covev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	batchv1listers "k8s.io/client-go/listers/batch/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/cronjob/metrics"
)

var (
	nextScheduleDelta = 100 * time.Millisecond
)

// ControllerV2 is a controller for CronJobs.
// Refactored Cronjob controller that uses DelayingQueue and informers
type ControllerV2 struct {
	queue    workqueue.RateLimitingInterface
	recorder record.EventRecorder

	jobControl     jobControlInterface
	cronJobControl cjControlInterface

	jobLister     batchv1listers.JobLister
	cronJobLister batchv1listers.CronJobLister

	jobListerSynced     cache.InformerSynced
	cronJobListerSynced cache.InformerSynced

	// now is a function that returns current time, done to facilitate unit tests
	now func() time.Time
}

// NewControllerV2 creates and initializes a new Controller.
func NewControllerV2(jobInformer batchv1informers.JobInformer, cronJobsInformer batchv1informers.CronJobInformer, kubeClient clientset.Interface) (*ControllerV2, error) {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartStructuredLogging(0)
	eventBroadcaster.StartRecordingToSink(&covev1client.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := ratelimiter.RegisterMetricAndTrackRateLimiterUsage("cronjob_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	jm := &ControllerV2{
		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "cronjob"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, corev1.EventSource{Component: "cronjob-controller"}),

		jobControl:     realJobControl{KubeClient: kubeClient},
		cronJobControl: &realCJControl{KubeClient: kubeClient},

		jobLister:     jobInformer.Lister(),
		cronJobLister: cronJobsInformer.Lister(),

		jobListerSynced:     jobInformer.Informer().HasSynced,
		cronJobListerSynced: cronJobsInformer.Informer().HasSynced,
		now:                 time.Now,
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jm.addJob,
		UpdateFunc: jm.updateJob,
		DeleteFunc: jm.deleteJob,
	})

	cronJobsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			jm.enqueueController(obj)
		},
		UpdateFunc: jm.updateCronJob,
		DeleteFunc: func(obj interface{}) {
			jm.enqueueController(obj)
		},
	})

	metrics.Register()

	return jm, nil
}

// Run starts the main goroutine responsible for watching and syncing jobs.
func (jm *ControllerV2) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer jm.queue.ShutDown()

	klog.InfoS("Starting cronjob controller v2")
	defer klog.InfoS("Shutting down cronjob controller v2")

	if !cache.WaitForNamedCacheSync("cronjob", stopCh, jm.jobListerSynced, jm.cronJobListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(jm.worker, time.Second, stopCh)
	}

	<-stopCh
}

func (jm *ControllerV2) worker() {
	for jm.processNextWorkItem() {
	}
}

func (jm *ControllerV2) processNextWorkItem() bool {
	key, quit := jm.queue.Get()
	if quit {
		return false
	}
	defer jm.queue.Done(key)

	requeueAfter, err := jm.sync(key.(string))
	switch {
	case err != nil:
		utilruntime.HandleError(fmt.Errorf("error syncing CronJobController %v, requeuing: %v", key.(string), err))
		jm.queue.AddRateLimited(key)
	case requeueAfter != nil:
		jm.queue.Forget(key)
		jm.queue.AddAfter(key, *requeueAfter)
	}
	return true
}

func (jm *ControllerV2) sync(cronJobKey string) (*time.Duration, error) {
	ns, name, err := cache.SplitMetaNamespaceKey(cronJobKey)
	if err != nil {
		return nil, err
	}

	cronJob, err := jm.cronJobLister.CronJobs(ns).Get(name)
	switch {
	case errors.IsNotFound(err):
		// may be cronjob is deleted, don't need to requeue this key
		klog.V(4).InfoS("CronJob not found, may be it is deleted", "cronjob", klog.KRef(ns, name), "err", err)
		return nil, nil
	case err != nil:
		// for other transient apiserver error requeue with exponential backoff
		return nil, err
	}

	jobsToBeReconciled, err := jm.getJobsToBeReconciled(cronJob)
	if err != nil {
		return nil, err
	}

	cronJobCopy, requeueAfter, err := jm.syncCronJob(cronJob, jobsToBeReconciled)
	if err != nil {
		klog.V(2).InfoS("Error reconciling cronjob", "cronjob", klog.KRef(cronJob.GetNamespace(), cronJob.GetName()), "err", err)
		return nil, err
	}

	err = jm.cleanupFinishedJobs(cronJobCopy, jobsToBeReconciled)
	if err != nil {
		klog.V(2).InfoS("Error cleaning up jobs", "cronjob", klog.KRef(cronJob.GetNamespace(), cronJob.GetName()), "resourceVersion", cronJob.GetResourceVersion(), "err", err)
		return nil, err
	}

	if requeueAfter != nil {
		klog.V(4).InfoS("Re-queuing cronjob", "cronjob", klog.KRef(cronJob.GetNamespace(), cronJob.GetName()), "requeueAfter", requeueAfter)
		return requeueAfter, nil
	}
	// this marks the key done, currently only happens when the cronjob is suspended or spec has invalid schedule format
	return nil, nil
}

// resolveControllerRef returns the controller referenced by a ControllerRef,
// or nil if the ControllerRef could not be resolved to a matching controller
// of the correct Kind.
func (jm *ControllerV2) resolveControllerRef(namespace string, controllerRef *metav1.OwnerReference) *batchv1.CronJob {
	// We can't look up by UID, so look up by Name and then verify UID.
	// Don't even try to look up by Name if it's the wrong Kind.
	if controllerRef.Kind != controllerKind.Kind {
		return nil
	}
	cronJob, err := jm.cronJobLister.CronJobs(namespace).Get(controllerRef.Name)
	if err != nil {
		return nil
	}
	if cronJob.UID != controllerRef.UID {
		// The controller we found with this Name is not the same one that the
		// ControllerRef points to.
		return nil
	}
	return cronJob
}

func (jm *ControllerV2) getJobsToBeReconciled(cronJob *batchv1.CronJob) ([]*batchv1.Job, error) {
	var jobSelector labels.Selector
	if len(cronJob.Spec.JobTemplate.Labels) == 0 {
		jobSelector = labels.Everything()
	} else {
		jobSelector = labels.Set(cronJob.Spec.JobTemplate.Labels).AsSelector()
	}
	jobList, err := jm.jobLister.Jobs(cronJob.Namespace).List(jobSelector)
	if err != nil {
		return nil, err
	}

	jobsToBeReconciled := []*batchv1.Job{}

	for _, job := range jobList {
		// If it has a ControllerRef, that's all that matters.
		if controllerRef := metav1.GetControllerOf(job); controllerRef != nil && controllerRef.Name == cronJob.Name {
			// this job is needs to be reconciled
			jobsToBeReconciled = append(jobsToBeReconciled, job)
		}
	}
	return jobsToBeReconciled, nil
}

// When a job is created, enqueue the controller that manages it and update it's expectations.
func (jm *ControllerV2) addJob(obj interface{}) {
	job := obj.(*batchv1.Job)
	if job.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new job shows up in a state that
		// is already pending deletion. Prevent the job from being a creation observation.
		jm.deleteJob(job)
		return
	}

	// If it has a ControllerRef, that's all that matters.
	if controllerRef := metav1.GetControllerOf(job); controllerRef != nil {
		cronJob := jm.resolveControllerRef(job.Namespace, controllerRef)
		if cronJob == nil {
			return
		}
		jm.enqueueController(cronJob)
		return
	}
}

// updateJob figures out what CronJob(s) manage a Job when the Job
// is updated and wake them up. If the anything of the Job have changed, we need to
// awaken both the old and new CronJob. old and cur must be *batchv1.Job
// types.
func (jm *ControllerV2) updateJob(old, cur interface{}) {
	curJob := cur.(*batchv1.Job)
	oldJob := old.(*batchv1.Job)
	if curJob.ResourceVersion == oldJob.ResourceVersion {
		// Periodic resync will send update events for all known jobs.
		// Two different versions of the same jobs will always have different RVs.
		return
	}

	curControllerRef := metav1.GetControllerOf(curJob)
	oldControllerRef := metav1.GetControllerOf(oldJob)
	controllerRefChanged := !reflect.DeepEqual(curControllerRef, oldControllerRef)
	if controllerRefChanged && oldControllerRef != nil {
		// The ControllerRef was changed. Sync the old controller, if any.
		if cronJob := jm.resolveControllerRef(oldJob.Namespace, oldControllerRef); cronJob != nil {
			jm.enqueueController(cronJob)
		}
	}

	// If it has a ControllerRef, that's all that matters.
	if curControllerRef != nil {
		cronJob := jm.resolveControllerRef(curJob.Namespace, curControllerRef)
		if cronJob == nil {
			return
		}
		jm.enqueueController(cronJob)
		return
	}
}

func (jm *ControllerV2) deleteJob(obj interface{}) {
	job, ok := obj.(*batchv1.Job)

	// When a delete is dropped, the relist will notice a job in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return
		}
		job, ok = tombstone.Obj.(*batchv1.Job)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a ReplicaSet %#v", obj))
			return
		}
	}

	controllerRef := metav1.GetControllerOf(job)
	if controllerRef == nil {
		// No controller should care about orphans being deleted.
		return
	}
	cronJob := jm.resolveControllerRef(job.Namespace, controllerRef)
	if cronJob == nil {
		return
	}
	jm.enqueueController(cronJob)
}

func (jm *ControllerV2) enqueueController(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}

	jm.queue.Add(key)
}

func (jm *ControllerV2) enqueueControllerAfter(obj interface{}, t time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}

	jm.queue.AddAfter(key, t)
}

// updateCronJob re-queues the CronJob for next scheduled time if there is a
// change in spec.schedule otherwise it re-queues it now
func (jm *ControllerV2) updateCronJob(old interface{}, curr interface{}) {
	oldCJ, okOld := old.(*batchv1.CronJob)
	newCJ, okNew := curr.(*batchv1.CronJob)

	if !okOld || !okNew {
		// typecasting of one failed, handle this better, may be log entry
		return
	}
	// if the change in schedule results in next requeue having to be sooner than it already was,
	// it will be handled here by the queue. If the next requeue is further than previous schedule,
	// the sync loop will essentially be a no-op for the already queued key with old schedule.
	if oldCJ.Spec.Schedule != newCJ.Spec.Schedule {
		// schedule changed, change the requeue time
		sched, err := cron.ParseStandard(newCJ.Spec.Schedule)
		if err != nil {
			// this is likely a user error in defining the spec value
			// we should log the error and not reconcile this cronjob until an update to spec
			klog.V(2).InfoS("Unparseable schedule for cronjob", "cronjob", klog.KRef(newCJ.GetNamespace(), newCJ.GetName()), "schedule", newCJ.Spec.Schedule, "err", err)
			jm.recorder.Eventf(newCJ, corev1.EventTypeWarning, "UnParseableCronJobSchedule", "unparseable schedule for cronjob: %s", newCJ.Spec.Schedule)
			return
		}
		now := jm.now()
		t := nextScheduledTimeDuration(sched, now)

		jm.enqueueControllerAfter(curr, *t)
		return
	}

	// other parameters changed, requeue this now and if this gets triggered
	// within deadline, sync loop will work on the CJ otherwise updates will be handled
	// during the next schedule
	// TODO: need to handle the change of spec.JobTemplate.metadata.labels explicitly
	//   to cleanup jobs with old labels
	jm.enqueueController(curr)
}

// syncCronJob reconciles a CronJob with a list of any Jobs that it created.
// All known jobs created by "cj" should be included in "js".
// The current time is passed in to facilitate testing.
// It returns a copy of the CronJob that is to be used by other functions
// that mutates the object
func (jm *ControllerV2) syncCronJob(
	cj *batchv1.CronJob,
	js []*batchv1.Job) (*batchv1.CronJob, *time.Duration, error) {

	cj = cj.DeepCopy()
	now := jm.now()

	childrenJobs := make(map[types.UID]bool)
	for _, j := range js {
		childrenJobs[j.ObjectMeta.UID] = true
		found := inActiveList(*cj, j.ObjectMeta.UID)
		if !found && !IsJobFinished(j) {
			cjCopy, err := jm.cronJobControl.GetCronJob(cj.Namespace, cj.Name)
			if err != nil {
				return nil, nil, err
			}
			if inActiveList(*cjCopy, j.ObjectMeta.UID) {
				cj = cjCopy
				continue
			}
			jm.recorder.Eventf(cj, corev1.EventTypeWarning, "UnexpectedJob", "Saw a job that the controller did not create or forgot: %s", j.Name)
			// We found an unfinished job that has us as the parent, but it is not in our Active list.
			// This could happen if we crashed right after creating the Job and before updating the status,
			// or if our jobs list is newer than our cj status after a relist, or if someone intentionally created
			// a job that they wanted us to adopt.
		} else if found && IsJobFinished(j) {
			_, status := getFinishedStatus(j)
			deleteFromActiveList(cj, j.ObjectMeta.UID)
			jm.recorder.Eventf(cj, corev1.EventTypeNormal, "SawCompletedJob", "Saw completed job: %s, status: %v", j.Name, status)
		} else if IsJobFinished(j) {
			// a job does not have to be in active list, as long as it is finished, we will process the timestamp
			if cj.Status.LastSuccessfulTime == nil {
				cj.Status.LastSuccessfulTime = j.Status.CompletionTime
			}
			if j.Status.CompletionTime != nil && j.Status.CompletionTime.After(cj.Status.LastSuccessfulTime.Time) {
				cj.Status.LastSuccessfulTime = j.Status.CompletionTime
			}
		}
	}

	// Remove any job reference from the active list if the corresponding job does not exist any more.
	// Otherwise, the cronjob may be stuck in active mode forever even though there is no matching
	// job running.
	for _, j := range cj.Status.Active {
		_, found := childrenJobs[j.UID]
		if found {
			continue
		}
		// Explicitly try to get the job from api-server to avoid a slow watch not able to update
		// the job lister on time, giving an unwanted miss
		_, err := jm.jobControl.GetJob(j.Namespace, j.Name)
		switch {
		case errors.IsNotFound(err):
			// The job is actually missing, delete from active list and schedule a new one if within
			// deadline
			jm.recorder.Eventf(cj, corev1.EventTypeNormal, "MissingJob", "Active job went missing: %v", j.Name)
			deleteFromActiveList(cj, j.UID)
		case err != nil:
			return cj, nil, err
		}
		// the job is missing in the lister but found in api-server
	}

	updatedCJ, err := jm.cronJobControl.UpdateStatus(cj)
	if err != nil {
		klog.V(2).InfoS("Unable to update status for cronjob", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "resourceVersion", cj.ResourceVersion, "err", err)
		return cj, nil, err
	}
	*cj = *updatedCJ

	if cj.DeletionTimestamp != nil {
		// The CronJob is being deleted.
		// Don't do anything other than updating status.
		return cj, nil, nil
	}

	if cj.Spec.Suspend != nil && *cj.Spec.Suspend {
		klog.V(4).InfoS("Not starting job because the cron is suspended", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		return cj, nil, nil
	}

	sched, err := cron.ParseStandard(cj.Spec.Schedule)
	if err != nil {
		// this is likely a user error in defining the spec value
		// we should log the error and not reconcile this cronjob until an update to spec
		klog.V(2).InfoS("Unparseable schedule", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "schedule", cj.Spec.Schedule, "err", err)
		jm.recorder.Eventf(cj, corev1.EventTypeWarning, "UnparseableSchedule", "unparseable schedule: %q : %s", cj.Spec.Schedule, err)
		return cj, nil, nil
	}

	if strings.Contains(cj.Spec.Schedule, "TZ") {
		jm.recorder.Eventf(cj, corev1.EventTypeWarning, "UnsupportedSchedule", "CRON_TZ or TZ used in schedule %q is not officially supported, see https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/ for more details", cj.Spec.Schedule)
	}

	scheduledTime, err := getNextScheduleTime(*cj, now, sched, jm.recorder)
	if err != nil {
		// this is likely a user error in defining the spec value
		// we should log the error and not reconcile this cronjob until an update to spec
		klog.V(2).InfoS("invalid schedule", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "schedule", cj.Spec.Schedule, "err", err)
		jm.recorder.Eventf(cj, corev1.EventTypeWarning, "InvalidSchedule", "invalid schedule: %s : %s", cj.Spec.Schedule, err)
		return cj, nil, nil
	}
	if scheduledTime == nil {
		// no unmet start time, return cj,.
		// The only time this should happen is if queue is filled after restart.
		// Otherwise, the queue is always suppose to trigger sync function at the time of
		// the scheduled time, that will give atleast 1 unmet time schedule
		klog.V(4).InfoS("No unmet start times", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		t := nextScheduledTimeDuration(sched, now)
		return cj, t, nil
	}

	tooLate := false
	if cj.Spec.StartingDeadlineSeconds != nil {
		tooLate = scheduledTime.Add(time.Second * time.Duration(*cj.Spec.StartingDeadlineSeconds)).Before(now)
	}
	if tooLate {
		klog.V(4).InfoS("Missed starting window", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		jm.recorder.Eventf(cj, corev1.EventTypeWarning, "MissSchedule", "Missed scheduled time to start a job: %s", scheduledTime.UTC().Format(time.RFC1123Z))

		// TODO: Since we don't set LastScheduleTime when not scheduling, we are going to keep noticing
		// the miss every cycle.  In order to avoid sending multiple events, and to avoid processing
		// the cj again and again, we could set a Status.LastMissedTime when we notice a miss.
		// Then, when we call getRecentUnmetScheduleTimes, we can take max(creationTimestamp,
		// Status.LastScheduleTime, Status.LastMissedTime), and then so we won't generate
		// and event the next time we process it, and also so the user looking at the status
		// can see easily that there was a missed execution.
		t := nextScheduledTimeDuration(sched, now)
		return cj, t, nil
	}
	if isJobInActiveList(&batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getJobName(cj, *scheduledTime),
			Namespace: cj.Namespace,
		}}, cj.Status.Active) || cj.Status.LastScheduleTime.Equal(&metav1.Time{Time: *scheduledTime}) {
		klog.V(4).InfoS("Not starting job because the scheduled time is already processed", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "schedule", scheduledTime)
		t := nextScheduledTimeDuration(sched, now)
		return cj, t, nil
	}
	if cj.Spec.ConcurrencyPolicy == batchv1.ForbidConcurrent && len(cj.Status.Active) > 0 {
		// Regardless which source of information we use for the set of active jobs,
		// there is some risk that we won't see an active job when there is one.
		// (because we haven't seen the status update to the SJ or the created pod).
		// So it is theoretically possible to have concurrency with Forbid.
		// As long the as the invocations are "far enough apart in time", this usually won't happen.
		//
		// TODO: for Forbid, we could use the same name for every execution, as a lock.
		// With replace, we could use a name that is deterministic per execution time.
		// But that would mean that you could not inspect prior successes or failures of Forbid jobs.
		klog.V(4).InfoS("Not starting job because prior execution is still running and concurrency policy is Forbid", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		jm.recorder.Eventf(cj, corev1.EventTypeNormal, "JobAlreadyActive", "Not starting job because prior execution is running and concurrency policy is Forbid")
		t := nextScheduledTimeDuration(sched, now)
		return cj, t, nil
	}
	if cj.Spec.ConcurrencyPolicy == batchv1.ReplaceConcurrent {
		for _, j := range cj.Status.Active {
			klog.V(4).InfoS("Deleting job that was still running at next scheduled start time", "job", klog.KRef(j.Namespace, j.Name))

			job, err := jm.jobControl.GetJob(j.Namespace, j.Name)
			if err != nil {
				jm.recorder.Eventf(cj, corev1.EventTypeWarning, "FailedGet", "Get job: %v", err)
				return cj, nil, err
			}
			if !deleteJob(cj, job, jm.jobControl, jm.recorder) {
				return cj, nil, fmt.Errorf("could not replace job %s/%s", job.Namespace, job.Name)
			}
		}
	}

	jobReq, err := getJobFromTemplate2(cj, *scheduledTime)
	if err != nil {
		klog.ErrorS(err, "Unable to make Job from template", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		return cj, nil, err
	}
	jobResp, err := jm.jobControl.CreateJob(cj.Namespace, jobReq)
	switch {
	case errors.HasStatusCause(err, corev1.NamespaceTerminatingCause):
	case errors.IsAlreadyExists(err):
		// If the job is created by other actor, assume  it has updated the cronjob status accordingly
		klog.InfoS("Job already exists", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "job", klog.KRef(jobReq.GetNamespace(), jobReq.GetName()))
		return cj, nil, err
	case err != nil:
		// default error handling
		jm.recorder.Eventf(cj, corev1.EventTypeWarning, "FailedCreate", "Error creating job: %v", err)
		return cj, nil, err
	}

	metrics.CronJobCreationSkew.Observe(jobResp.ObjectMeta.GetCreationTimestamp().Sub(*scheduledTime).Seconds())
	klog.V(4).InfoS("Created Job", "job", klog.KRef(jobResp.GetNamespace(), jobResp.GetName()), "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
	jm.recorder.Eventf(cj, corev1.EventTypeNormal, "SuccessfulCreate", "Created job %v", jobResp.Name)

	// ------------------------------------------------------------------ //

	// If this process restarts at this point (after posting a job, but
	// before updating the status), then we might try to start the job on
	// the next time.  Actually, if we re-list the SJs and Jobs on the next
	// iteration of syncAll, we might not see our own status update, and
	// then post one again.  So, we need to use the job name as a lock to
	// prevent us from making the job twice (name the job with hash of its
	// scheduled time).

	// Add the just-started job to the status list.
	jobRef, err := getRef(jobResp)
	if err != nil {
		klog.V(2).InfoS("Unable to make object reference", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "err", err)
		return cj, nil, fmt.Errorf("unable to make object reference for job for %s", klog.KRef(cj.GetNamespace(), cj.GetName()))
	}
	cj.Status.Active = append(cj.Status.Active, *jobRef)
	cj.Status.LastScheduleTime = &metav1.Time{Time: *scheduledTime}
	if _, err := jm.cronJobControl.UpdateStatus(cj); err != nil {
		klog.InfoS("Unable to update status", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "resourceVersion", cj.ResourceVersion, "err", err)
		return cj, nil, fmt.Errorf("unable to update status for %s (rv = %s): %v", klog.KRef(cj.GetNamespace(), cj.GetName()), cj.ResourceVersion, err)
	}

	t := nextScheduledTimeDuration(sched, now)
	return cj, t, nil
}

func getJobName(cj *batchv1.CronJob, scheduledTime time.Time) string {
	return fmt.Sprintf("%s-%d", cj.Name, getTimeHashInMinutes(scheduledTime))
}

// nextScheduledTimeDuration returns the time duration to requeue based on
// the schedule and current time. It adds a 100ms padding to the next requeue to account
// for Network Time Protocol(NTP) time skews. If the time drifts are adjusted which in most
// realistic cases would be around 100s, scheduled cron will still be executed without missing
// the schedule.
func nextScheduledTimeDuration(sched cron.Schedule, now time.Time) *time.Duration {
	t := sched.Next(now).Add(nextScheduleDelta).Sub(now)
	return &t
}

// cleanupFinishedJobs cleanups finished jobs created by a CronJob
func (jm *ControllerV2) cleanupFinishedJobs(cj *batchv1.CronJob, js []*batchv1.Job) error {
	// If neither limits are active, there is no need to do anything.
	if cj.Spec.FailedJobsHistoryLimit == nil && cj.Spec.SuccessfulJobsHistoryLimit == nil {
		return nil
	}

	failedJobs := []*batchv1.Job{}
	successfulJobs := []*batchv1.Job{}

	for _, job := range js {
		isFinished, finishedStatus := jm.getFinishedStatus(job)
		if isFinished && finishedStatus == batchv1.JobComplete {
			successfulJobs = append(successfulJobs, job)
		} else if isFinished && finishedStatus == batchv1.JobFailed {
			failedJobs = append(failedJobs, job)
		}
	}

	if cj.Spec.SuccessfulJobsHistoryLimit != nil {
		jm.removeOldestJobs(cj,
			successfulJobs,
			*cj.Spec.SuccessfulJobsHistoryLimit)
	}

	if cj.Spec.FailedJobsHistoryLimit != nil {
		jm.removeOldestJobs(cj,
			failedJobs,
			*cj.Spec.FailedJobsHistoryLimit)
	}

	// Update the CronJob, in case jobs were removed from the list.
	_, err := jm.cronJobControl.UpdateStatus(cj)
	return err
}

func (jm *ControllerV2) getFinishedStatus(j *batchv1.Job) (bool, batchv1.JobConditionType) {
	for _, c := range j.Status.Conditions {
		if (c.Type == batchv1.JobComplete || c.Type == batchv1.JobFailed) && c.Status == corev1.ConditionTrue {
			return true, c.Type
		}
	}
	return false, ""
}

// removeOldestJobs removes the oldest jobs from a list of jobs
func (jm *ControllerV2) removeOldestJobs(cj *batchv1.CronJob, js []*batchv1.Job, maxJobs int32) {
	numToDelete := len(js) - int(maxJobs)
	if numToDelete <= 0 {
		return
	}

	klog.V(4).InfoS("Cleaning up jobs from CronJob list", "deletejobnum", numToDelete, "jobnum", len(js), "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))

	sort.Sort(byJobStartTimeStar(js))
	for i := 0; i < numToDelete; i++ {
		klog.V(4).InfoS("Removing job from CronJob list", "job", js[i].Name, "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		deleteJob(cj, js[i], jm.jobControl, jm.recorder)
	}
}

// isJobInActiveList take a job and checks if activeJobs has a job with the same
// name and namespace.
func isJobInActiveList(job *batchv1.Job, activeJobs []corev1.ObjectReference) bool {
	for _, j := range activeJobs {
		if j.Name == job.Name && j.Namespace == job.Namespace {
			return true
		}
	}
	return false
}
