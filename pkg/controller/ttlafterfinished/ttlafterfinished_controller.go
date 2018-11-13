/*
Copyright 2018 The Kubernetes Authors.

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

package ttlafterfinished

import (
	"fmt"
	"time"

	"k8s.io/klog"

	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	batchinformers "k8s.io/client-go/informers/batch/v1"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	batchlisters "k8s.io/client-go/listers/batch/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	jobutil "k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/util/metrics"
)

// Controller watches for changes of Jobs API objects. Triggered by Job creation
// and updates, it enqueues Jobs that have non-nil `.spec.ttlSecondsAfterFinished`
// to the `queue`. The Controller has workers who consume `queue`, check whether
// the Job TTL has expired or not; if the Job TTL hasn't expired, it will add the
// Job to the queue after the TTL is expected to expire; if the TTL has expired, the
// worker will send requests to the API server to delete the Jobs accordingly.
// This is implemented outside of Job controller for separation of concerns, and
// because it will be extended to handle other finishable resource types.
type Controller struct {
	client   clientset.Interface
	recorder record.EventRecorder

	// jLister can list/get Jobs from the shared informer's store
	jLister batchlisters.JobLister

	// jStoreSynced returns true if the Job store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jListerSynced cache.InformerSynced

	// Jobs that the controller will check its TTL and attempt to delete when the TTL expires.
	queue workqueue.RateLimitingInterface

	// The clock for tracking time
	clock clock.Clock
}

// New creates an instance of Controller
func New(jobInformer batchinformers.JobInformer, client clientset.Interface) *Controller {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(klog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})

	if client != nil && client.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("ttl_after_finished_controller", client.CoreV1().RESTClient().GetRateLimiter())
	}

	tc := &Controller{
		client:   client,
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "ttl-after-finished-controller"}),
		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "ttl_jobs_to_delete"),
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    tc.addJob,
		UpdateFunc: tc.updateJob,
	})

	tc.jLister = jobInformer.Lister()
	tc.jListerSynced = jobInformer.Informer().HasSynced

	tc.clock = clock.RealClock{}

	return tc
}

// Run starts the workers to clean up Jobs.
func (tc *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer tc.queue.ShutDown()

	klog.Infof("Starting TTL after finished controller")
	defer klog.Infof("Shutting down TTL after finished controller")

	if !controller.WaitForCacheSync("TTL after finished", stopCh, tc.jListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(tc.worker, time.Second, stopCh)
	}

	<-stopCh
}

func (tc *Controller) addJob(obj interface{}) {
	job := obj.(*batch.Job)
	klog.V(4).Infof("Adding job %s/%s", job.Namespace, job.Name)

	if job.DeletionTimestamp == nil && needsCleanup(job) {
		tc.enqueue(job)
	}
}

func (tc *Controller) updateJob(old, cur interface{}) {
	job := cur.(*batch.Job)
	klog.V(4).Infof("Updating job %s/%s", job.Namespace, job.Name)

	if job.DeletionTimestamp == nil && needsCleanup(job) {
		tc.enqueue(job)
	}
}

func (tc *Controller) enqueue(job *batch.Job) {
	klog.V(4).Infof("Add job %s/%s to cleanup", job.Namespace, job.Name)
	key, err := controller.KeyFunc(job)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", job, err))
		return
	}

	tc.queue.Add(key)
}

func (tc *Controller) enqueueAfter(job *batch.Job, after time.Duration) {
	key, err := controller.KeyFunc(job)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", job, err))
		return
	}

	tc.queue.AddAfter(key, after)
}

func (tc *Controller) worker() {
	for tc.processNextWorkItem() {
	}
}

func (tc *Controller) processNextWorkItem() bool {
	key, quit := tc.queue.Get()
	if quit {
		return false
	}
	defer tc.queue.Done(key)

	err := tc.processJob(key.(string))
	tc.handleErr(err, key)

	return true
}

func (tc *Controller) handleErr(err error, key interface{}) {
	if err == nil {
		tc.queue.Forget(key)
		return
	}

	utilruntime.HandleError(fmt.Errorf("error cleaning up Job %v, will retry: %v", key, err))
	tc.queue.AddRateLimited(key)
}

// processJob will check the Job's state and TTL and delete the Job when it
// finishes and its TTL after finished has expired. If the Job hasn't finished or
// its TTL hasn't expired, it will be added to the queue after the TTL is expected
// to expire.
// This function is not meant to be invoked concurrently with the same key.
func (tc *Controller) processJob(key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	klog.V(4).Infof("Checking if Job %s/%s is ready for cleanup", namespace, name)
	// Ignore the Jobs that are already deleted or being deleted, or the ones that don't need clean up.
	job, err := tc.jLister.Jobs(namespace).Get(name)
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	if expired, err := tc.processTTL(job); err != nil {
		return err
	} else if !expired {
		return nil
	}

	// The Job's TTL is assumed to have expired, but the Job TTL might be stale.
	// Before deleting the Job, do a final sanity check.
	// If TTL is modified before we do this check, we cannot be sure if the TTL truly expires.
	// The latest Job may have a different UID, but it's fine because the checks will be run again.
	fresh, err := tc.client.BatchV1().Jobs(namespace).Get(name, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	// Use the latest Job TTL to see if the TTL truly expires.
	if expired, err := tc.processTTL(fresh); err != nil {
		return err
	} else if !expired {
		return nil
	}
	// Cascade deletes the Jobs if TTL truly expires.
	policy := metav1.DeletePropagationForeground
	options := &metav1.DeleteOptions{
		PropagationPolicy: &policy,
		Preconditions:     &metav1.Preconditions{UID: &fresh.UID},
	}
	klog.V(4).Infof("Cleaning up Job %s/%s", namespace, name)
	return tc.client.BatchV1().Jobs(fresh.Namespace).Delete(fresh.Name, options)
}

// processTTL checks whether a given Job's TTL has expired, and add it to the queue after the TTL is expected to expire
// if the TTL will expire later.
func (tc *Controller) processTTL(job *batch.Job) (expired bool, err error) {
	// We don't care about the Jobs that are going to be deleted, or the ones that don't need clean up.
	if job.DeletionTimestamp != nil || !needsCleanup(job) {
		return false, nil
	}

	now := tc.clock.Now()
	t, err := timeLeft(job, &now)
	if err != nil {
		return false, err
	}

	// TTL has expired
	if *t <= 0 {
		return true, nil
	}

	tc.enqueueAfter(job, *t)
	return false, nil
}

// needsCleanup checks whether a Job has finished and has a TTL set.
func needsCleanup(j *batch.Job) bool {
	return j.Spec.TTLSecondsAfterFinished != nil && jobutil.IsJobFinished(j)
}

func getFinishAndExpireTime(j *batch.Job) (*time.Time, *time.Time, error) {
	if !needsCleanup(j) {
		return nil, nil, fmt.Errorf("Job %s/%s should not be cleaned up", j.Namespace, j.Name)
	}
	finishAt, err := jobFinishTime(j)
	if err != nil {
		return nil, nil, err
	}
	finishAtUTC := finishAt.UTC()
	expireAtUTC := finishAtUTC.Add(time.Duration(*j.Spec.TTLSecondsAfterFinished) * time.Second)
	return &finishAtUTC, &expireAtUTC, nil
}

func timeLeft(j *batch.Job, since *time.Time) (*time.Duration, error) {
	finishAt, expireAt, err := getFinishAndExpireTime(j)
	if err != nil {
		return nil, err
	}
	if finishAt.UTC().After(since.UTC()) {
		klog.Warningf("Warning: Found Job %s/%s finished in the future. This is likely due to time skew in the cluster. Job cleanup will be deferred.", j.Namespace, j.Name)
	}
	remaining := expireAt.UTC().Sub(since.UTC())
	klog.V(4).Infof("Found Job %s/%s finished at %v, remaining TTL %v since %v, TTL will expire at %v", j.Namespace, j.Name, finishAt.UTC(), remaining, since.UTC(), expireAt.UTC())
	return &remaining, nil
}

// jobFinishTime takes an already finished Job and returns the time it finishes.
func jobFinishTime(finishedJob *batch.Job) (metav1.Time, error) {
	for _, c := range finishedJob.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
			finishAt := c.LastTransitionTime
			if finishAt.IsZero() {
				return metav1.Time{}, fmt.Errorf("unable to find the time when the Job %s/%s finished", finishedJob.Namespace, finishedJob.Name)
			}
			return c.LastTransitionTime, nil
		}
	}

	// This should never happen if the Jobs has finished
	return metav1.Time{}, fmt.Errorf("unable to find the status of the finished Job %s/%s", finishedJob.Namespace, finishedJob.Name)
}
