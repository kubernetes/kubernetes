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
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	batchlisters "k8s.io/client-go/listers/batch/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubernetes/pkg/controller"
	jobutil "k8s.io/kubernetes/pkg/controller/job"
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

	// jListerSynced returns true if the Job store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jListerSynced cache.InformerSynced

	// pLister can list/get Pods from the shared informer's store
	pLister corelisters.PodLister

	// pListerSynced returns true if the Pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	pListerSynced cache.InformerSynced

	// Jobs that the controller will check its TTL and attempt to delete when the TTL expires.
	queue workqueue.RateLimitingInterface

	// The clock for tracking time
	clock clock.Clock
}

// New creates an instance of Controller
func New(jobInformer batchinformers.JobInformer, podInformer coreinformers.PodInformer, client clientset.Interface) *Controller {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(klog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})

	if client != nil && client.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("ttl_after_finished_controller", client.CoreV1().RESTClient().GetRateLimiter())
	}

	tc := &Controller{
		client:   client,
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "ttl-after-finished-controller"}),
		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "ttl_objects_to_delete"),
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    tc.addResource,
		UpdateFunc: tc.updateResource,
	})

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    tc.addResource,
		UpdateFunc: tc.updateResource,
	})

	tc.jLister = jobInformer.Lister()
	tc.jListerSynced = jobInformer.Informer().HasSynced

	tc.pLister = podInformer.Lister()
	tc.pListerSynced = podInformer.Informer().HasSynced

	tc.clock = clock.RealClock{}

	return tc
}

// Run starts the workers to clean up Jobs.
func (tc *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer tc.queue.ShutDown()

	klog.Infof("Starting TTL after finished controller")
	defer klog.Infof("Shutting down TTL after finished controller")

	if !cache.WaitForNamedCacheSync("TTL after finished", stopCh, tc.jListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(tc.worker, time.Second, stopCh)
	}

	<-stopCh
}

func (tc *Controller) addResource(obj interface{}) {
	switch resource := obj.(type) {
	case *batch.Job:
		klog.V(4).Infof("Adding job %s/%s", resource.Namespace, resource.Name)
		if resource.DeletionTimestamp == nil && jobNeedsCleanup(resource) {
			tc.enqueue(resource, resource.Namespace, resource.Name)
		}
	case *v1.Pod:
		klog.V(4).Infof("Adding pod %s/%s", resource.Namespace, resource.Name)
		if resource.DeletionTimestamp == nil && podNeedsCleanup(resource) {
			tc.enqueue(resource, resource.Namespace, resource.Name)
		}
	}
}

func (tc *Controller) updateResource(old, cur interface{}) {
	switch obj := cur.(type) {
	case *batch.Job:
		klog.V(4).Infof("Updating job %s/%s", obj.Namespace, obj.Name)
		if obj.DeletionTimestamp == nil && jobNeedsCleanup(obj) {
			tc.enqueue(obj, obj.Namespace, obj.Name)
		}
	case *v1.Pod:
		klog.V(4).Infof("Updating pod %s/%s", obj.Namespace, obj.Name)
		if obj.DeletionTimestamp == nil && podNeedsCleanup(obj) {
			tc.enqueue(obj, obj.Namespace, obj.Name)
		}
	}
}

func (tc *Controller) enqueue(obj interface{}, namespace, name string) {
	klog.V(4).Infof("Add obj %s/%s to cleanup", namespace, name)
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
		return
	}

	tc.queue.Add(key)
}

func (tc *Controller) enqueueAfter(obj interface{}, after time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", obj, err))
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

	jErr, pErr := tc.process(key.(string))
	tc.handleErr(jErr, key)
	tc.handleErr(pErr, key)
	return true
}

func (tc *Controller) handleErr(err error, key interface{}) {
	if err == nil {
		tc.queue.Forget(key)
		return
	}

	utilruntime.HandleError(fmt.Errorf("error cleaning up Job/Pod %v, will retry: %v", key, err))
	tc.queue.AddRateLimited(key)
}

// process will check try to get the named job/pod, processes TTL for them.
// It collects and returns errors for jobs and pods separately
func (tc *Controller) process(key string) (error, error) {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err, nil
	}
	var jErr, pErr error
	var job *batch.Job
	var pod *v1.Pod

	klog.V(4).Infof("Checking if Job %s/%s and Pod %s/%s is ready for cleanup", namespace, name, namespace, name)
	// Ignore the Jobs/Pods that are already deleted or being deleted, or the ones that don't need clean up.
	job, jErr = tc.jLister.Jobs(namespace).Get(name)
	pod, pErr = tc.pLister.Pods(namespace).Get(name)
	if errors.IsNotFound(jErr) && errors.IsNotFound(pErr) {
		return nil, nil
	}

	if jErr == nil {
		jErr = tc.processJob(job)
	}
	if pErr == nil {
		pErr = tc.processPod(pod)
	}
	return jErr, pErr
}

// processJob will check the Job's state and TTL and delete the Job when it
// finishes and its TTL after finished has expired. If the Job hasn't finished or
// its TTL hasn't expired, it will be added to the queue after the TTL is expected
// to expire.
// This function is not meant to be invoked concurrently with the same key.
func (tc *Controller) processJob(job *batch.Job) error {
	if expired, err := tc.processJobTTL(job); err != nil {
		return err
	} else if !expired {
		return nil
	}

	// The Job's TTL is assumed to have expired, but the Job TTL might be stale.
	// Before deleting the Job, do a final sanity check.
	// If TTL is modified before we do this check, we cannot be sure if the TTL truly expires.
	// The latest Job may have a different UID, but it's fine because the checks will be run again.
	fresh, err := tc.client.BatchV1().Jobs(job.Namespace).Get(job.Name, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	// Use the latest Job TTL to see if the TTL truly expires.
	expired, err := tc.processJobTTL(fresh)
	if err != nil {
		return err
	}
	if !expired {
		return nil
	}
	// Cascade deletes the Jobs if TTL truly expires.
	policy := metav1.DeletePropagationForeground
	options := &metav1.DeleteOptions{
		PropagationPolicy: &policy,
		Preconditions:     &metav1.Preconditions{UID: &fresh.UID},
	}
	klog.V(4).Infof("Cleaning up Job %s/%s", job.Namespace, job.Name)
	return tc.client.BatchV1().Jobs(fresh.Namespace).Delete(fresh.Name, options)
}

// processPod will check the Pod's state and TTL and delete the Pod when it
// finishes and its TTL after finished has expired. If the Pod hasn't finished or
// its TTL hasn't expired, it will be added to the queue after the TTL is expected
// to expire.
// This function is not meant to be invoked concurrently with the same key.
func (tc *Controller) processPod(pod *v1.Pod) error {
	if expired, err := tc.processPodTTL(pod); err != nil {
		return err
	} else if !expired {
		return nil
	}

	// The Pod's TTL is assumed to have expired, but the Pod TTL might be stale.
	// Before deleting the Pod, do a final sanity check.
	// If TTL is modified before we do this check, we cannot be sure if the TTL truly expires.
	// The latest Pod may have a different UID, but it's fine because the checks will be run again.
	fresh, err := tc.client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	// Use the latest Job TTL to see if the TTL truly expires.
	expired, err := tc.processPodTTL(fresh)
	if err != nil {
		return err
	}
	if !expired {
		return nil
	}
	// Cascade deletes the Jobs if TTL truly expires.
	policy := metav1.DeletePropagationForeground
	options := &metav1.DeleteOptions{
		PropagationPolicy: &policy,
		Preconditions:     &metav1.Preconditions{UID: &fresh.UID},
	}
	klog.V(4).Infof("Cleaning up Pod %s/%s", pod.Namespace, pod.Name)
	return tc.client.CoreV1().Pods(fresh.Namespace).Delete(fresh.Name, options)
}

// processJobTTL checks whether a given Job's TTL has expired, and add it to the queue after the TTL is expected to expire
// if the TTL will expire later.
func (tc *Controller) processJobTTL(job *batch.Job) (expired bool, err error) {
	// We don't care about the Jobs that are going to be deleted, or the ones that don't need clean up.
	if job.DeletionTimestamp != nil || !jobNeedsCleanup(job) {
		return false, nil
	}

	now := tc.clock.Now()
	t, err := timeLeftForJob(job, &now)
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

// processPodTTL checks whether a given Job's TTL has expired, and add it to the queue after the TTL is expected to expire
// if the TTL will expire later.
func (tc *Controller) processPodTTL(pod *v1.Pod) (expired bool, err error) {
	// We don't care about the Jobs that are going to be deleted, or the ones that don't need clean up.
	if pod.DeletionTimestamp != nil || !podNeedsCleanup(pod) {
		return false, nil
	}

	now := tc.clock.Now()
	t, err := timeLeftForPod(pod, &now)
	if err != nil {
		return false, err
	}

	// TTL has expired
	if *t <= 0 {
		return true, nil
	}

	tc.enqueueAfter(pod, *t)
	return false, nil
}

// jobNeedsCleanup checks whether a Job has finished and has a TTL set.
func jobNeedsCleanup(j *batch.Job) bool {
	return j.Spec.TTLSecondsAfterFinished != nil && jobutil.IsJobFinished(j)
}

// podNeedsCleanup checks whether a Pod has finished and has a TTL set.
func podNeedsCleanup(p *v1.Pod) bool {
	return p.Spec.TTLSecondsAfterFinished != nil && isPodFinished(p)
}

func getFinishAndExpireTimeForJob(j *batch.Job) (*time.Time, *time.Time, error) {
	if !jobNeedsCleanup(j) {
		return nil, nil, fmt.Errorf("job %s/%s should not be cleaned up", j.Namespace, j.Name)
	}
	finishAt, err := jobFinishTime(j)
	if err != nil {
		return nil, nil, err
	}
	finishAtUTC := finishAt.UTC()
	expireAtUTC := finishAtUTC.Add(time.Duration(*j.Spec.TTLSecondsAfterFinished) * time.Second)
	return &finishAtUTC, &expireAtUTC, nil
}

func timeLeftForJob(j *batch.Job, since *time.Time) (*time.Duration, error) {
	finishAt, expireAt, err := getFinishAndExpireTimeForJob(j)
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

func timeLeftForPod(p *v1.Pod, since *time.Time) (*time.Duration, error) {
	finishAt, expireAt, err := getFinishAndExpireTimeForPod(p)
	if err != nil {
		return nil, err
	}
	if finishAt.UTC().After(since.UTC()) {
		klog.Warningf("Warning: Found Pod %s/%s finished in the future. This is likely due to time skew in the cluster.Pod cleanup will be deferred.", p.Namespace, p.Name)
	}
	remaining := expireAt.UTC().Sub(since.UTC())
	klog.V(4).Infof("Found Pod %s/%s finished at %v, remaining TTL %v since %v, TTL will expire at %v", p.Namespace, p.Name, finishAt.UTC(), remaining, since.UTC(), expireAt.UTC())
	return &remaining, nil

}

func getFinishAndExpireTimeForPod(p *v1.Pod) (*time.Time, *time.Time, error) {
	if !podNeedsCleanup(p) {
		return nil, nil, fmt.Errorf("pod %s/%s should not be cleaned up", p.Namespace, p.Name)
	}
	finishAt, err := podFinishTime(p)
	if err != nil {
		return nil, nil, err
	}
	finishAtUTC := finishAt.UTC()
	expireAtUTC := finishAtUTC.Add(time.Duration(*p.Spec.TTLSecondsAfterFinished) * time.Second)
	return &finishAtUTC, &expireAtUTC, nil
}

// podFinishTime takes an already finished Pod and returns the time it finishes.
func podFinishTime(finishedPod *v1.Pod) (metav1.Time, error) {
	t := metav1.Time{
		Time: time.Time{},
	}
	for _, c := range finishedPod.Status.ContainerStatuses {
		if c.State.Terminated.FinishedAt.After(t.Time) {
			t = c.State.Terminated.FinishedAt
		}
	}
	return t, nil
}

func isPodFinished(p *v1.Pod) bool {
	if p.Status.Phase == v1.PodSucceeded || p.Status.Phase == v1.PodFailed {
		return true
	}
	return false
}
