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
	"time"

	"github.com/robfig/cron"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	batchv1informers "k8s.io/client-go/informers/batch/v1"
	batchv1beta1informers "k8s.io/client-go/informers/batch/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	batchv1listers "k8s.io/client-go/listers/batch/v1"
	"k8s.io/client-go/listers/batch/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
)

var (
	delta100ms                    = 100 * time.Millisecond
	retriesForEnqueingAllCronJobs = 5
)

// Refactored Cronjob controller that uses DelayingQueue and informers

// ControllerV2 is a controller for CronJobs.
type ControllerV2 struct {
	queue    workqueue.DelayingInterface
	recorder record.EventRecorder

	jobControl     jobControlInterface
	cronJobControl cjControlInterface

	jobLister     batchv1listers.JobLister
	cronJobLister v1beta1.CronJobLister

	jobListerSynced     cache.InformerSynced
	cronJobListerSynced cache.InformerSynced

	// now is a function that returns current time, done to facilitate unit tests
	now func() time.Time
}

// NewController creates and initializes a new Controller.
func NewControllerV2(jobInformer batchv1informers.JobInformer, cronJobsInformer batchv1beta1informers.CronJobInformer, kubeClient clientset.Interface) (*ControllerV2, error) {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(klog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := ratelimiter.RegisterMetricAndTrackRateLimiterUsage("cronjob_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	jm := &ControllerV2{
		queue:    workqueue.NewNamedDelayingQueue("cronjob"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cronjob-controller"}),

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

	return jm, nil
}

// Run starts the main goroutine responsible for watching and syncing jobs.
func (jm *ControllerV2) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer jm.queue.ShutDown()

	klog.Infof("Starting cronjob controller v2")
	defer klog.Infof("Shutting down cronjob controller v2")

	go jm.enqueueAllCronJobsWithRetries(retriesForEnqueingAllCronJobs)

	if !cache.WaitForNamedCacheSync("cronjob", stopCh, jm.jobListerSynced, jm.cronJobListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(jm.worker, time.Second, stopCh)
	}

	<-stopCh
}

// enqueueAllCronJobsWithRetries enqueues all the cronjobs and is thread-safe.
func (jm *ControllerV2) enqueueAllCronJobsWithRetries(retries int) {
	cronjobs := jm.listCronJobsWithRetries(retries)
	for _, cronjob := range cronjobs {
		jm.enqueueController(cronjob)
	}
	return
}

// listCronJobsWithRetries tries to list the cronjobs with retries upon hitting an error
func (jm *ControllerV2) listCronJobsWithRetries(retries int) []*batchv1beta1.CronJob {
	cronjobs := []*batchv1beta1.CronJob{}
	var err error
	for i := 0; i < retries; i++ {
		cronjobs, err = jm.cronJobLister.List(labels.Everything())
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("unable to list cronjobs at the start of cronjob controller, retrying %d more time/s, error: %v\n", retries-i-1, err))
			continue
		}
	}
	return cronjobs
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
	err, requeueAfter := jm.sync(key.(string))
	switch {
	case err != nil:
		utilruntime.HandleError(fmt.Errorf("Error syncing CronJobController %v, requeuing: %v", key.(string), err))
		jm.queue.Add(key)
	case requeueAfter != nil:
		jm.queue.AddAfter(key, *requeueAfter)
	default:
		jm.queue.Done(key)
	}
	return true
}

func (jm *ControllerV2) sync(cronJobKey string) (error, *time.Duration) {
	ns, name, err := cache.SplitMetaNamespaceKey(cronJobKey)
	if err != nil {
		return err, nil
	}

	cronJob, err := jm.cronJobLister.CronJobs(ns).Get(name)
	switch {
	case errors.IsNotFound(err):
		// may be cronjob is deleted, dont need to requeue this key
		klog.V(2).InfoS("cronjob not found, may be it is deleted", "cronjob", klog.KRef(ns, name), "err", err)
		return nil, nil
	case err != nil:
		// for other transient apiserver error requeue with exponential backoff
		return err, nil
	}

	jobList := []*batchv1.Job{}
	if len(cronJob.Spec.JobTemplate.Labels) == 0 {
		jobList, err = jm.jobLister.Jobs(ns).List(labels.Everything())
	} else {
		jobList, err = jm.jobLister.Jobs(ns).List(labels.Set(cronJob.Spec.JobTemplate.Labels).AsSelector())
	}
	if err != nil {
		return err, nil
	}

	jobsToBeReconciled := []batchv1.Job{}

	for _, job := range jobList {
		// If it has a ControllerRef, that's all that matters.
		if controllerRef := metav1.GetControllerOf(job); controllerRef != nil && controllerRef.Name == name {
			// this job is needs to be reconciled
			jobsToBeReconciled = append(jobsToBeReconciled, *job)
		}
	}

	err, requeueAfter := syncOne2(cronJob, jobsToBeReconciled, time.Now(), jm.jobControl, jm.cronJobControl, jm.recorder)
	if err != nil {
		klog.V(2).InfoS("error reconciling cronjob", "cronjob", klog.KRef(cronJob.GetNamespace(), cronJob.GetName()), "err", err)
		return err, nil
	}

	err = cleanupFinishedJobs2(cronJob, jobsToBeReconciled, jm.jobControl, jm.cronJobControl, jm.recorder)
	if err != nil {
		klog.V(2).InfoS("error cleaning up jobs", "cronjob", klog.KRef(cronJob.GetNamespace(), cronJob.GetName()), "resourceVersion", cronJob.GetResourceVersion(), "err", err)
		return err, nil
	}

	if requeueAfter != nil {
		klog.V(4).InfoS("re-queuing cronjob", "cronjob", klog.KRef(cronJob.GetNamespace(), cronJob.GetName()), "requeueAfter", requeueAfter)
		return nil, requeueAfter
	}
	// this marks the key done, currently only happens when the cronjob is suspended or spec has invalid schedule format
	return nil, nil
}

// resolveControllerRef returns the controller referenced by a ControllerRef,
// or nil if the ControllerRef could not be resolved to a matching controller
// of the correct Kind.
func (jm *ControllerV2) resolveControllerRef(namespace string, controllerRef *metav1.OwnerReference) *batchv1beta1.CronJob {
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

// When a job is created, enqueue the controller that manages it and update it's expectations.
func (jm *ControllerV2) addJob(obj interface{}) {
	job := obj.(*batchv1.Job)
	if job.DeletionTimestamp != nil {
		// on a restart of the controller controller, it's possible a new job shows up in a state that
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

func (jm *ControllerV2) enqueueControllerWithTime(obj interface{}, t time.Duration) {
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
	oldCJ, okOld := old.(*batchv1beta1.CronJob)
	newCJ, okNew := curr.(*batchv1beta1.CronJob)

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
			klog.V(2).InfoS("unparseable schedule for cronjob", "cronjob", klog.KRef(newCJ.GetNamespace(), newCJ.GetName()), "schedule", newCJ.Spec.Schedule, "err", err)
			return
		}
		now := jm.now()
		t := nextScheduledTimeDurationWithDelta(sched, now)

		jm.enqueueControllerWithTime(curr, *t)
		return
	}

	// other parameters changed, requeue this now and if this gets triggered
	// within deadline, sync loop will work on the CJ otherwise updates will be handled
	// during the next schedule
	// TODO: need to handle the change of spec.JobTemplate.metadata.labels explicitly
	//   to cleanup jobs with old labels
	jm.enqueueController(curr)
}

// TODO: @alpatel we need to return errors from this function, in order to allow
//  for errors to propagate up the reconcile function and kick in queues exponential
//  backed off retries more details on inline code comments in the func

// syncOne reconciles a CronJob with a list of any Jobs that it created.
// All known jobs created by "cj" should be included in "js".
// The current time is passed in to facilitate testing.
// It has no receiver, to facilitate testing.
func syncOne2(
	cj *batchv1beta1.CronJob,
	js []batchv1.Job,
	now time.Time,
	jc jobControlInterface,
	cjc cjControlInterface,
	recorder record.EventRecorder) (error, *time.Duration) {

	childrenJobs := make(map[types.UID]bool)
	for _, j := range js {
		childrenJobs[j.ObjectMeta.UID] = true
		found := inActiveList(*cj, j.ObjectMeta.UID)
		if !found && !IsJobFinished(&j) {
			recorder.Eventf(cj, v1.EventTypeWarning, "UnexpectedJob", "Saw a job that the controller did not create or forgot: %s", j.Name)
			// We found an unfinished job that has us as the parent, but it is not in our Active list.
			// This could happen if we crashed right after creating the Job and before updating the status,
			// or if our jobs list is newer than our cj status after a relist, or if someone intentionally created
			// a job that they wanted us to adopt.

			// TODO: maybe handle the adoption case?  Concurrency/suspend rules will not apply in that case, obviously, since we can't
			// stop users from creating jobs if they have permission.  It is assumed that if a
			// user has permission to create a job within a namespace, then they have permission to make any cronJob
			// in the same namespace "adopt" that job.  ReplicaSets and their Pods work the same way.
			// TBS: how to update cj.Status.LastScheduleTime if the adopted job is newer than any we knew about?
		} else if found && IsJobFinished(&j) {
			_, status := getFinishedStatus(&j)
			deleteFromActiveList(cj, j.ObjectMeta.UID)
			recorder.Eventf(cj, v1.EventTypeNormal, "SawCompletedJob", "Saw completed job: %s, status: %v", j.Name, status)
		}
	}

	// Remove any job reference from the active list if the corresponding job does not exist any more.
	// Otherwise, the cronjob may be stuck in active mode forever even though there is no matching
	// job running.
	for _, j := range cj.Status.Active {
		if found := childrenJobs[j.UID]; !found {
			recorder.Eventf(cj, v1.EventTypeNormal, "MissingJob", "Active job went missing: %v", j.Name)
			deleteFromActiveList(cj, j.UID)
		}
	}

	// TODO: @alpatel explore if cached client can be used as realCJControl
	updatedCJ, err := cjc.UpdateStatus(cj)
	if err != nil {
		klog.V(2).InfoS("Unable to update status for cronjon", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "resourceVersion", cj.ResourceVersion, "err", err)
		return err, nil
	}
	*cj = *updatedCJ

	if cj.DeletionTimestamp != nil {
		// The CronJob is being deleted.
		// Don't do anything other than updating status.
		return fmt.Errorf("cronjob %s/%s is being deleted", cj.Namespace, cj.Name), nil
	}

	if cj.Spec.Suspend != nil && *cj.Spec.Suspend {
		klog.V(4).InfoS("Not starting job because the cron is suspended", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		return nil, nil
	}

	// TODO: alpatel: this is here now just to pass the unit tests,
	//  move this to the start of the function to return on error early
	sched, err := cron.ParseStandard(cj.Spec.Schedule)
	if err != nil {
		// this is likely a user error in defining the spec value
		// we should log the error and not reconcile this cronjob until an update to spec
		klog.V(2).InfoS("unparseable schedule", "cronjob", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "schedule", cj.Spec.Schedule, "err", err)
		recorder.Eventf(cj, v1.EventTypeWarning, "UnparseableSchedule", "unparseable schedule: %s : %s", cj.Spec.Schedule, err)
		return nil, nil
	}
	times, err := getRecentUnmetScheduleTimes2(*cj, now, sched)
	switch {
	case err != nil && len(times) == 0:
		// too many missed jobs, schedule the next one on time and return
		// TODO: @alpatel, with revised workflow this probably needs be reworked
		// 		the thought process is we will always miss the schedule time and
		// 		controller will reconcile after scheduled time + delta time spent
		//		in reconciliation loop. With that if a job misses 100 schedule times
		// 		with this block, it will always return here.
		recorder.Eventf(cj, v1.EventTypeWarning, "TooManyMissedTimes", "Too many missed times for the cronjob, will schedule the next one", err)
		klog.ErrorS(err, "too many missed times", "cronjob", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), err)
		// schedule for next period
		t := nextScheduledTimeDurationWithDelta(sched, now)

		// in order to unwedge the cronjob from always returning from this block
		// TODO: @alpatel, in the future we should add a .status.nextScheduleTime
		//    and refactor getRecentUnmetScheduleTimes to give us 101th time after
		//    100 missed times
		cj.Status.LastScheduleTime = &metav1.Time{Time: now}
		if _, err := cjc.UpdateStatus(cj); err != nil {
			klog.InfoS("Unable to update status", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "resourceVersion", cj.ResourceVersion, "err", err)
			return fmt.Errorf("unable to update status for %s (rv = %s): %v", klog.KRef(cj.GetNamespace(), cj.GetName()), cj.ResourceVersion, err), nil
		}
		return nil, t
	case len(times) == 0:
		// no unmet start time, return.
		// The only time this should happen is if queue is filled after restart.
		// Otherwise, the queue is always suppose to trigger sync function at the time of
		// the scheduled time, that will give atleast 1 unmet time schedule
		klog.V(4).InfoS("No unmet start times", "cronjob", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		t := nextScheduledTimeDurationWithDelta(sched, now)
		return nil, t
	}

	scheduledTime := times[len(times)-1]
	tooLate := false
	if cj.Spec.StartingDeadlineSeconds != nil {
		tooLate = scheduledTime.Add(time.Second * time.Duration(*cj.Spec.StartingDeadlineSeconds)).Before(now)
	}
	if tooLate {
		klog.V(4).InfoS("Missed starting window", "cronjob", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		recorder.Eventf(cj, v1.EventTypeWarning, "MissSchedule", "Missed scheduled time to start a job: %s", scheduledTime.Format(time.RFC1123Z))
		// TODO: @alpatel: confirm with @soltysh the following TODO is not true anymore. We will now
		//      only requeue the for the next scheduled time, instead of hitting this error again
		//      and again. With the new workflow we might only hit this if controller got wedged
		//		and on restart we miss the schedule time by more than deadline. In that case
		// 		schedule for next time
		// TODO: Since we don't set LastScheduleTime when not scheduling, we are going to keep noticing
		// the miss every cycle.  In order to avoid sending multiple events, and to avoid processing
		// the cj again and again, we could set a Status.LastMissedTime when we notice a miss.
		// Then, when we call getRecentUnmetScheduleTimes, we can take max(creationTimestamp,
		// Status.LastScheduleTime, Status.LastMissedTime), and then so we won't generate
		// and event the next time we process it, and also so the user looking at the status
		// can see easily that there was a missed execution.
		t := nextScheduledTimeDurationWithDelta(sched, now)
		return nil, t
	}
	if cj.Spec.ConcurrencyPolicy == batchv1beta1.ForbidConcurrent && len(cj.Status.Active) > 0 {
		// Regardless which source of information we use for the set of active jobs,
		// there is some risk that we won't see an active job when there is one.
		// (because we haven't seen the status update to the SJ or the created pod).
		// So it is theoretically possible to have concurrency with Forbid.
		// As long the as the invocations are "far enough apart in time", this usually won't happen.
		//
		// TODO: @alpatel confirm we @soltysh we can probably set a deterministic job name per
		// 		schedule time. The formula: last_run=UTCInSeconds(now)-UTCInSeconds(creationTime)/intervalInSeconds
		//  	might give use last scheduled counter. May be save it in status and reconcile on that
		// TODO: for Forbid, we could use the same name for every execution, as a lock.
		// With replace, we could use a name that is deterministic per execution time.
		// But that would mean that you could not inspect prior successes or failures of Forbid jobs.
		klog.V(4).InfoS("Not starting job because prior execution is still running and concurrency policy is Forbid", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		t := nextScheduledTimeDurationWithDelta(sched, now)
		return nil, t
	}
	if cj.Spec.ConcurrencyPolicy == batchv1beta1.ReplaceConcurrent {
		for _, j := range cj.Status.Active {
			klog.V(4).InfoS("Deleting job that was still running at next scheduled start time", "job", klog.KRef(j.Namespace, j.Name))

			job, err := jc.GetJob(j.Namespace, j.Name)
			if err != nil {
				recorder.Eventf(cj, v1.EventTypeWarning, "FailedGet", "Get job: %v", err)
				return err, nil
			}
			if !deleteJob(cj, job, jc, recorder) {
				return fmt.Errorf("could not replace job %s/%s", job.Namespace, job.Name), nil
			}
		}
	}

	jobReq, err := getJobFromTemplate(cj, scheduledTime)
	if err != nil {
		klog.ErrorS(err, "Unable to make Job from template", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
		return err, nil
	}
	jobResp, err := jc.CreateJob(cj.Namespace, jobReq)
	switch {
	case errors.HasStatusCause(err, v1.NamespaceTerminatingCause):
		// TODO: @alpatel log, event?
	case errors.IsAlreadyExists(err):
		// TODO: @alpatel handle this, we tried to create a job that already exists. may be update/patch?
	case err != nil:
		// default error handling
		recorder.Eventf(cj, v1.EventTypeWarning, "FailedCreate", "Error creating job: %v", err)
		return err, nil
	}
	klog.V(4).InfoS("Created Job", "job", klog.KRef(jobResp.GetNamespace(), jobResp.GetName()), "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()))
	recorder.Eventf(cj, v1.EventTypeNormal, "SuccessfulCreate", "Created job %v", jobResp.Name)

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
		return fmt.Errorf("unable to make object reference for job for %s", klog.KRef(cj.GetNamespace(), cj.GetName())), nil
	} else {
		cj.Status.Active = append(cj.Status.Active, *jobRef)
	}
	cj.Status.LastScheduleTime = &metav1.Time{Time: scheduledTime}
	if _, err := cjc.UpdateStatus(cj); err != nil {
		klog.InfoS("Unable to update status", "cronjob", klog.KRef(cj.GetNamespace(), cj.GetName()), "resourceVersion", cj.ResourceVersion, "err", err)
		return fmt.Errorf("unable to update status for %s (rv = %s): %v", klog.KRef(cj.GetNamespace(), cj.GetName()), cj.ResourceVersion, err), nil
	}

	t := nextScheduledTimeDurationWithDelta(sched, now)
	return nil, t
}

// nextScheduledTimeDurationWithDelta returns the time duration to requeue based on
// the schedule and current time. It adds a 100ms padding to the next requeue to account
// for Network Time Protocol(NTP) time skews. If the time drifts are adjusted which in most
// realistic cases would be around 100s, scheduled cron will still be executed without missing
// the schedule.
func nextScheduledTimeDurationWithDelta(sched cron.Schedule, now time.Time) *time.Duration {
	t := sched.Next(now).Add(delta100ms).Sub(now)
	return &t
}

// cleanupFinishedJobs cleanups finished jobs created by a CronJob
func cleanupFinishedJobs2(cj *batchv1beta1.CronJob, js []batchv1.Job, jc jobControlInterface,
	cjc cjControlInterface, recorder record.EventRecorder) error {
	// If neither limits are active, there is no need to do anything.
	if cj.Spec.FailedJobsHistoryLimit == nil && cj.Spec.SuccessfulJobsHistoryLimit == nil {
		return nil
	}

	failedJobs := []batchv1.Job{}
	successfulJobs := []batchv1.Job{}

	for _, job := range js {
		isFinished, finishedStatus := getFinishedStatus(&job)
		if isFinished && finishedStatus == batchv1.JobComplete {
			successfulJobs = append(successfulJobs, job)
		} else if isFinished && finishedStatus == batchv1.JobFailed {
			failedJobs = append(failedJobs, job)
		}
	}

	if cj.Spec.SuccessfulJobsHistoryLimit != nil {
		removeOldestJobs(cj,
			successfulJobs,
			jc,
			*cj.Spec.SuccessfulJobsHistoryLimit,
			recorder)
	}

	if cj.Spec.FailedJobsHistoryLimit != nil {
		removeOldestJobs(cj,
			failedJobs,
			jc,
			*cj.Spec.FailedJobsHistoryLimit,
			recorder)
	}

	// Update the CronJob, in case jobs were removed from the list.
	_, err := cjc.UpdateStatus(cj)
	return err
}
