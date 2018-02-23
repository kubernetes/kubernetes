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
	"sort"
	"time"

	"github.com/golang/glog"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	batchv1informer "k8s.io/client-go/informers/batch/v1"
	batchv1beta1informer "k8s.io/client-go/informers/batch/v1beta1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	batchv1Lister "k8s.io/client-go/listers/batch/v1"
	batchv1beta1Lister "k8s.io/client-go/listers/batch/v1beta1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	ref "k8s.io/client-go/tools/reference"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
)

// controllerKind contains the schema.GroupVersionKind for this controller type.
var controllerKind = batchv1beta1.SchemeGroupVersion.WithKind("CronJob")

type CronJobController struct {
	kubeClient clientset.Interface
	jobControl jobControlInterface
	sjControl  sjControlInterface
	podControl podControlInterface
	recorder   record.EventRecorder
	// CronJob that need to be synced
	queue workqueue.RateLimitingInterface
	// cjobSynced returns true if the cronJob store has been synced at least once.
	cjobSynced cache.InformerSynced
	// jobSynced returns true if the job store has been synced at least once.
	jobSynced cache.InformerSynced
	// podSynced returns true if the pod store has been synced at least once.
	podSynced cache.InformerSynced
	// CronJob sync handler
	syncHandler func(key string) error
	// podLister can list/get pods from the shared informer's store
	podLister corelisters.PodLister
	//cjobLister can list/get CronJobs from shared informer's store
	cjobLister batchv1beta1Lister.CronJobLister
	//jobLister can list/get Job from shared informer's store
	jobLister batchv1Lister.JobLister
}

func NewCronJobControllerFromInformer(cjbInformer batchv1beta1informer.CronJobInformer, jobInformer batchv1informer.JobInformer, podInformer coreinformers.PodInformer, kubeClient clientset.Interface) (*CronJobController, error) {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.CoreV1().RESTClient()).Events("")})

	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := metrics.RegisterMetricAndTrackRateLimiterUsage("cronjob_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	jm := &CronJobController{
		kubeClient: kubeClient,
		jobControl: realJobControl{KubeClient: kubeClient},
		sjControl:  &realSJControl{KubeClient: kubeClient},
		podControl: &realPodControl{KubeClient: kubeClient},
		recorder:   eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cronjob-controller"}),
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "cronjob"),
	}
	cjbInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jm.addCronjob,
		UpdateFunc: jm.updateCronjob,
		DeleteFunc: jm.deleteCronjob,
	})

	jm.cjobLister = cjbInformer.Lister()
	jm.jobLister = jobInformer.Lister()
	jm.podLister = podInformer.Lister()
	jm.cjobSynced = cjbInformer.Informer().HasSynced
	jm.jobSynced = jobInformer.Informer().HasSynced
	jm.podSynced = podInformer.Informer().HasSynced
	jm.syncHandler = jm.syncCronJob
	return jm, nil
}

// Run the main goroutine responsible for watching and syncing jobs.
func (jm *CronJobController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer jm.queue.ShutDown()

	if !controller.WaitForCacheSync("cronjob", stopCh, jm.cjobSynced, jm.jobSynced, jm.podSynced) {
		return
	}

	glog.Infof("Starting CronJob Manager")

	// we need add all the CronJobs to the queue periodically to make sure we can start a job according to the schedule
	go wait.Until(func() {
		glog.Info("Start enqueue all CronJobs")
		cjobs, err := jm.cjobLister.List(labels.Everything())
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("couldn't list CronJobs %v", err))
			return
		}
		for _, job := range cjobs {
			glog.Infof("Start enqueue CronJob: %v", job)
			jm.enqueue(job)
		}
		glog.Infof("Finished enqueue %v CronJobs", len(cjobs))
	}, time.Second*10, stopCh)

	//TODO Add ConcurrentCronJobSyncs to KubeControllerManagerConfiguration
	for i := 0; i < 5; i++ {
		go wait.Until(jm.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down CronJob Manager")
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (jm *CronJobController) worker() {
	for jm.processNextWorkItem() {
	}
}

func (jm *CronJobController) processNextWorkItem() bool {
	key, quit := jm.queue.Get()
	if quit {
		return false
	}
	defer jm.queue.Done(key)

	err := jm.syncHandler(key.(string))
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("error syncing cronjob: %v", err))
		jm.queue.AddRateLimited(key)
	}
	return true
}

func (jm *CronJobController) enqueue(cronjob *batchv1beta1.CronJob) {
	key, err := controller.KeyFunc(cronjob)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", cronjob, err))
		return
	}

	jm.queue.Add(key)
}
func (jm *CronJobController) syncCronJob(key string) error {
	startTime := time.Now()
	glog.V(4).Infof("Started syncing CronJob %q (%v)", key, startTime)
	defer func() {
		glog.V(4).Infof("Finished syncing CronJob %q (%v)", key, time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	cjob, err := jm.cjobLister.CronJobs(namespace).Get(name)
	if errors.IsNotFound(err) {
		glog.V(2).Infof("CronJob %v has been deleted", key)
		return nil
	}
	if err != nil {
		return err
	}
	jobs, err := jm.jobLister.Jobs(namespace).List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("can't list Jobs: %v", err))
		return err
	}
	var jobsCopy []batchv1.Job
	for _, job := range jobs {
		jobsCopy = append(jobsCopy, *job.DeepCopy())
	}
	jobsBySj := groupJobsByParent(jobsCopy)

	glog.V(4).Infof("Found %d groups", len(jobsBySj))
	cjobCopy := cjob.DeepCopy()
	syncOne(cjobCopy, jobsBySj[cjob.UID], time.Now(), jm.jobControl, jm.sjControl, jm.podControl, jm.recorder)
	cleanupFinishedJobs(cjobCopy, jobsBySj[cjob.UID], jm.jobControl, jm.sjControl, jm.podControl, jm.recorder)
	return nil
}

func (jm *CronJobController) addCronjob(obj interface{}) {
	d := obj.(*batchv1beta1.CronJob)
	glog.V(4).Infof("Adding CronJob %s", d.Name)
	jm.enqueue(d)
}

func (jm *CronJobController) updateCronjob(old, cur interface{}) {
	oldC := old.(*batchv1beta1.CronJob)
	curC := cur.(*batchv1beta1.CronJob)
	glog.V(4).Infof("Updating CronJob %s", oldC.Name)
	jm.enqueue(curC)
}

func (jm *CronJobController) deleteCronjob(obj interface{}) {
	d, ok := obj.(*batchv1beta1.CronJob)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return
		}
		d, ok = tombstone.Obj.(*batchv1beta1.CronJob)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a CronJob %#v", obj))
			return
		}
	}
	glog.V(4).Infof("Deleting CronJob %s", d.Name)
	jm.enqueue(d)
}

// cleanupFinishedJobs cleanups finished jobs created by a CronJob
func cleanupFinishedJobs(sj *batchv1beta1.CronJob, js []batchv1.Job, jc jobControlInterface,
	sjc sjControlInterface, pc podControlInterface, recorder record.EventRecorder) {
	// If neither limits are active, there is no need to do anything.
	if sj.Spec.FailedJobsHistoryLimit == nil && sj.Spec.SuccessfulJobsHistoryLimit == nil {
		return
	}

	failedJobs := []batchv1.Job{}
	succesfulJobs := []batchv1.Job{}

	for _, job := range js {
		isFinished, finishedStatus := getFinishedStatus(&job)
		if isFinished && finishedStatus == batchv1.JobComplete {
			succesfulJobs = append(succesfulJobs, job)
		} else if isFinished && finishedStatus == batchv1.JobFailed {
			failedJobs = append(failedJobs, job)
		}
	}

	if sj.Spec.SuccessfulJobsHistoryLimit != nil {
		removeOldestJobs(sj,
			succesfulJobs,
			jc,
			pc,
			*sj.Spec.SuccessfulJobsHistoryLimit,
			recorder)
	}

	if sj.Spec.FailedJobsHistoryLimit != nil {
		removeOldestJobs(sj,
			failedJobs,
			jc,
			pc,
			*sj.Spec.FailedJobsHistoryLimit,
			recorder)
	}

	// Update the CronJob, in case jobs were removed from the list.
	if _, err := sjc.UpdateStatus(sj); err != nil {
		nameForLog := fmt.Sprintf("%s/%s", sj.Namespace, sj.Name)
		glog.Infof("Unable to update status for %s (rv = %s): %v", nameForLog, sj.ResourceVersion, err)
	}
}

// removeOldestJobs removes the oldest jobs from a list of jobs
func removeOldestJobs(sj *batchv1beta1.CronJob, js []batchv1.Job, jc jobControlInterface,
	pc podControlInterface, maxJobs int32, recorder record.EventRecorder) {
	numToDelete := len(js) - int(maxJobs)
	if numToDelete <= 0 {
		return
	}

	nameForLog := fmt.Sprintf("%s/%s", sj.Namespace, sj.Name)
	glog.V(4).Infof("Cleaning up %d/%d jobs from %s", numToDelete, len(js), nameForLog)

	sort.Sort(byJobStartTime(js))
	for i := 0; i < numToDelete; i++ {
		glog.V(4).Infof("Removing job %s from %s", js[i].Name, nameForLog)
		deleteJob(sj, &js[i], jc, pc, recorder, "history limit reached")
	}
}

// syncOne reconciles a CronJob with a list of any Jobs that it created.
// All known jobs created by "sj" should be included in "js".
// The current time is passed in to facilitate testing.
// It has no receiver, to facilitate testing.
func syncOne(sj *batchv1beta1.CronJob, js []batchv1.Job, now time.Time, jc jobControlInterface, sjc sjControlInterface, pc podControlInterface, recorder record.EventRecorder) {
	nameForLog := fmt.Sprintf("%s/%s", sj.Namespace, sj.Name)

	childrenJobs := make(map[types.UID]bool)
	for _, j := range js {
		childrenJobs[j.ObjectMeta.UID] = true
		found := inActiveList(*sj, j.ObjectMeta.UID)
		if !found && !IsJobFinished(&j) {
			recorder.Eventf(sj, v1.EventTypeWarning, "UnexpectedJob", "Saw a job that the controller did not create or forgot: %v", j.Name)
			// We found an unfinished job that has us as the parent, but it is not in our Active list.
			// This could happen if we crashed right after creating the Job and before updating the status,
			// or if our jobs list is newer than our sj status after a relist, or if someone intentionally created
			// a job that they wanted us to adopt.

			// TODO: maybe handle the adoption case?  Concurrency/suspend rules will not apply in that case, obviously, since we can't
			// stop users from creating jobs if they have permission.  It is assumed that if a
			// user has permission to create a job within a namespace, then they have permission to make any cronJob
			// in the same namespace "adopt" that job.  ReplicaSets and their Pods work the same way.
			// TBS: how to update sj.Status.LastScheduleTime if the adopted job is newer than any we knew about?
		} else if found && IsJobFinished(&j) {
			deleteFromActiveList(sj, j.ObjectMeta.UID)
			// TODO: event to call out failure vs success.
			recorder.Eventf(sj, v1.EventTypeNormal, "SawCompletedJob", "Saw completed job: %v", j.Name)
		}
	}

	// Remove any job reference from the active list if the corresponding job does not exist any more.
	// Otherwise, the cronjob may be stuck in active mode forever even though there is no matching
	// job running.
	for _, j := range sj.Status.Active {
		if found := childrenJobs[j.UID]; !found {
			recorder.Eventf(sj, v1.EventTypeNormal, "MissingJob", "Active job went missing: %v", j.Name)
			deleteFromActiveList(sj, j.UID)
		}
	}

	updatedSJ, err := sjc.UpdateStatus(sj)
	if err != nil {
		glog.Errorf("Unable to update status for %s (rv = %s): %v", nameForLog, sj.ResourceVersion, err)
		return
	}
	*sj = *updatedSJ

	if sj.DeletionTimestamp != nil {
		// The CronJob is being deleted.
		// Don't do anything other than updating status.
		return
	}

	if sj.Spec.Suspend != nil && *sj.Spec.Suspend {
		glog.V(4).Infof("Not starting job for %s because it is suspended", nameForLog)
		return
	}

	times, err := getRecentUnmetScheduleTimes(*sj, now)
	if err != nil {
		recorder.Eventf(sj, v1.EventTypeWarning, "FailedNeedsStart", "Cannot determine if job needs to be started: %v", err)
		glog.Errorf("Cannot determine if %s needs to be started: %v", nameForLog, err)
		return
	}
	// TODO: handle multiple unmet start times, from oldest to newest, updating status as needed.
	if len(times) == 0 {
		glog.V(4).Infof("No unmet start times for %s", nameForLog)
		return
	}
	if len(times) > 1 {
		glog.V(4).Infof("Multiple unmet start times for %s so only starting last one", nameForLog)
	}

	scheduledTime := times[len(times)-1]
	tooLate := false
	if sj.Spec.StartingDeadlineSeconds != nil {
		tooLate = scheduledTime.Add(time.Second * time.Duration(*sj.Spec.StartingDeadlineSeconds)).Before(now)
	}
	if tooLate {
		glog.V(4).Infof("Missed starting window for %s", nameForLog)
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
	if sj.Spec.ConcurrencyPolicy == batchv1beta1.ForbidConcurrent && len(sj.Status.Active) > 0 {
		// Regardless which source of information we use for the set of active jobs,
		// there is some risk that we won't see an active job when there is one.
		// (because we haven't seen the status update to the SJ or the created pod).
		// So it is theoretically possible to have concurrency with Forbid.
		// As long the as the invocations are "far enough apart in time", this usually won't happen.
		//
		// TODO: for Forbid, we could use the same name for every execution, as a lock.
		// With replace, we could use a name that is deterministic per execution time.
		// But that would mean that you could not inspect prior successes or failures of Forbid jobs.
		glog.V(4).Infof("Not starting job for %s because of prior execution still running and concurrency policy is Forbid", nameForLog)
		return
	}
	if sj.Spec.ConcurrencyPolicy == batchv1beta1.ReplaceConcurrent {
		for _, j := range sj.Status.Active {
			// TODO: this should be replaced with server side job deletion
			// currently this mimics JobReaper from pkg/kubectl/stop.go
			glog.V(4).Infof("Deleting job %s of %s that was still running at next scheduled start time", j.Name, nameForLog)

			job, err := jc.GetJob(j.Namespace, j.Name)
			if err != nil {
				recorder.Eventf(sj, v1.EventTypeWarning, "FailedGet", "Get job: %v", err)
				return
			}
			if !deleteJob(sj, job, jc, pc, recorder, "") {
				return
			}
		}
	}

	jobReq, err := getJobFromTemplate(sj, scheduledTime)
	if err != nil {
		glog.Errorf("Unable to make Job from template in %s: %v", nameForLog, err)
		return
	}
	jobResp, err := jc.CreateJob(sj.Namespace, jobReq)
	if err != nil {
		recorder.Eventf(sj, v1.EventTypeWarning, "FailedCreate", "Error creating job: %v", err)
		return
	}
	glog.V(4).Infof("Created Job %s for %s", jobResp.Name, nameForLog)
	recorder.Eventf(sj, v1.EventTypeNormal, "SuccessfulCreate", "Created job %v", jobResp.Name)

	// ------------------------------------------------------------------ //

	// If this process restarts at this point (after posting a job, but
	// before updating the status), then we might try to start the job on
	// the next time.  Actually, if we relist the SJs and Jobs on the next
	// iteration of syncAll, we might not see our own status update, and
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
	sj.Status.LastScheduleTime = &metav1.Time{Time: scheduledTime}
	if _, err := sjc.UpdateStatus(sj); err != nil {
		glog.Infof("Unable to update status for %s (rv = %s): %v", nameForLog, sj.ResourceVersion, err)
	}

	return
}

// deleteJob reaps a job, deleting the job, the pobs and the reference in the active list
func deleteJob(sj *batchv1beta1.CronJob, job *batchv1.Job, jc jobControlInterface,
	pc podControlInterface, recorder record.EventRecorder, reason string) bool {
	// TODO: this should be replaced with server side job deletion
	// currencontinuetly this mimics JobReaper from pkg/kubectl/stop.go
	nameForLog := fmt.Sprintf("%s/%s", sj.Namespace, sj.Name)

	// scale job down to 0
	if *job.Spec.Parallelism != 0 {
		zero := int32(0)
		var err error
		job.Spec.Parallelism = &zero
		job, err = jc.UpdateJob(job.Namespace, job)
		if err != nil {
			recorder.Eventf(sj, v1.EventTypeWarning, "FailedUpdate", "Update job: %v", err)
			return false
		}
	}
	// remove all pods...
	selector, _ := metav1.LabelSelectorAsSelector(job.Spec.Selector)
	options := metav1.ListOptions{LabelSelector: selector.String()}
	podList, err := pc.ListPods(job.Namespace, options)
	if err != nil {
		recorder.Eventf(sj, v1.EventTypeWarning, "FailedList", "List job-pods: %v", err)
		return false
	}
	errList := []error{}
	for _, pod := range podList.Items {
		glog.V(2).Infof("CronJob controller is deleting Pod %v/%v", pod.Namespace, pod.Name)
		if err := pc.DeletePod(pod.Namespace, pod.Name); err != nil {
			// ignores the error when the pod isn't found
			if !errors.IsNotFound(err) {
				errList = append(errList, err)
			}
		}
	}
	if len(errList) != 0 {
		recorder.Eventf(sj, v1.EventTypeWarning, "FailedDelete", "Deleted job-pods: %v", utilerrors.NewAggregate(errList))
		return false
	}
	// ... the job itself...
	if err := jc.DeleteJob(job.Namespace, job.Name); err != nil {
		recorder.Eventf(sj, v1.EventTypeWarning, "FailedDelete", "Deleted job: %v", err)
		glog.Errorf("Error deleting job %s from %s: %v", job.Name, nameForLog, err)
		return false
	}
	// ... and its reference from active list
	deleteFromActiveList(sj, job.ObjectMeta.UID)
	recorder.Eventf(sj, v1.EventTypeNormal, "SuccessfulDelete", "Deleted job %v", job.Name)

	return true
}

func getRef(object runtime.Object) (*v1.ObjectReference, error) {
	return ref.GetReference(scheme.Scheme, object)
}
