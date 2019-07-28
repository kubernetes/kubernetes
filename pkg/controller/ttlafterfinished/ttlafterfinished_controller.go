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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/clock"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	jobutil "k8s.io/kubernetes/pkg/controller/job"
)

// Controller watches for changes of Jobs/Pods API objects. Triggered by Job creation
// and updates, it enqueues Jobs/Pods that have non-nil `.spec.ttlSecondsAfterFinished`
// to the `queue`. The Controller has workers who consume `queue`, check whether
// the TTL has expired or not; if the TTL hasn't expired, it will add the
// Job/Pod to the queue after the TTL is expected to expire; if the TTL has expired, the
// worker will send requests to the API server to delete the Jobs/Pods accordingly.
// This is implemented outside of Job/Pod controller for separation of concerns, and
// because it will be extended to handle other finishable resource types.
type Controller struct {
	client dynamic.NamespaceableResourceInterface
	// recorder record.EventRecorder

	resource schema.GroupVersionResource
	// rLister can list/get resources from the shared informer's store
	rLister cache.GenericLister

	// rInformer can register the watch event handler for the resource
	rInformer informers.GenericInformer

	// rStoreSynced returns true if the resource store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	rListerSynced cache.InformerSynced

	// Resources that the controller will check its TTL and attempt to delete when the TTL expires.
	queue workqueue.RateLimitingInterface

	// The clock for tracking time
	clock clock.Clock
}

// New creates an instance of Controller
func New(informerFactory informers.SharedInformerFactory, resource schema.GroupVersionResource, client dynamic.Interface) *Controller {
	// eventBroadcaster := record.NewBroadcaster()
	// eventBroadcaster.StartLogging(klog.Infof)
	// eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})

	// if client != nil && client.CoreV1().RESTClient().GetRateLimiter() != nil {
	// metrics.RegisterMetricAndTrackRateLimiterUsage("ttl_after_finished_controller", client.CoreV1().RESTClient().GetRateLimiter())
	// }

	tc := &Controller{
		client: client.Resource(resource),
		// recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "ttl-after-finished-controller"}),
		resource: resource,
		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), fmt.Sprintf("ttl_%s_to_delete", resource.Resource)),
	}

	genericInformer, err := informerFactory.ForResource(resource)
	if err != nil {
		return nil
	}

	tc.rInformer = genericInformer

	tc.rInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    tc.addResource,
		UpdateFunc: tc.updateResource,
	})

	tc.rListerSynced = tc.rInformer.Informer().HasSynced

	tc.clock = clock.RealClock{}

	return tc
}

// Run starts the workers to clean up Jobs/Pods.
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
	resource := obj.(metav1.Object)
	klog.V(4).Infof("Adding %s %s%s", tc.resource.Resource, resource.GetNamespace(), resource.GetName())

	if resource.GetDeletionTimestamp() == nil && needsCleanup(obj) {
		tc.enqueue(resource)
	}
}

func (tc *Controller) updateResource(old, cur interface{}) {
	resource := old.(metav1.Object)
	klog.V(4).Infof("Updating %s %s%s", tc.resource.Resource, resource.GetNamespace(), resource.GetName())

	if resource.GetDeletionTimestamp() == nil && needsCleanup(old) {
		tc.enqueue(resource)
	}
}

func (tc *Controller) enqueue(obj metav1.Object) {
	klog.V(4).Infof("Add %s %s/%s to cleanup", tc.resource.Resource, obj.GetNamespace(), obj.GetName())
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for %s %#v: %v", tc.resource.Resource, obj, err))
		return
	}

	tc.queue.Add(key)
}

func (tc *Controller) enqueueAfter(obj metav1.Object, after time.Duration) {
	klog.V(4).Infof("Add %s %s/%s to cleanup after %#v", tc.resource.Resource, obj.GetNamespace(), obj.GetName(), after)
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for %s %#v: %v", tc.resource.Resource, obj, err))
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

	err := tc.process(key.(string))
	tc.handleErr(err, key)

	return true
}

func (tc *Controller) handleErr(err error, key interface{}) {
	if err == nil {
		tc.queue.Forget(key)
		return
	}

	utilruntime.HandleError(fmt.Errorf("error cleaning up %s %v, will retry: %v", tc.resource.Resource, key, err))
	tc.queue.AddRateLimited(key)
}

// process will check the resource's state and TTL and delete the resource when it
// finishes and its TTL after finished has expired. If the resource hasn't finished or
// its TTL hasn't expired, it will be added to the queue after the TTL is expected
// to expire.
// This function is not meant to be invoked concurrently with the same key.
func (tc *Controller) process(key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	klog.V(4).Infof("Checking if %s %s/%s is ready for cleanup", tc.resource.Resource, namespace, name)
	fresh, err := tc.client.Namespace(namespace).Get(name, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	// Use the latest resource TTL to see if the TTL truly expires.
	if expired, err := tc.processTTL(fresh); err != nil {
		return err
	} else if !expired {
		return nil
	}
	// Cascade deletes the Resource if TTL truly expires.
	policy := metav1.DeletePropagationForeground
	uid := fresh.GetUID()
	options := &metav1.DeleteOptions{
		PropagationPolicy: &policy,
		Preconditions:     &metav1.Preconditions{UID: &uid},
	}
	klog.V(4).Infof("Cleaning up %s %s/%s", tc.resource.Resource, namespace, name)
	return tc.client.Namespace(fresh.GetNamespace()).Delete(fresh.GetName(), options)
}

// processTTL checks whether a given resource's TTL has expired, and add it to the queue after the TTL is expected to expire
// if the TTL will expire later.
func (tc *Controller) processTTL(u *unstructured.Unstructured) (bool, error) {
	// We don't care about the resources that are going to be deleted, or the ones that don't need clean up.
	var j *batch.Job
	var p *v1.Pod
	var t *time.Duration
	var err error
	now := tc.clock.Now()
	if u.GetKind() == "job" {
		if err = runtime.DefaultUnstructuredConverter.FromUnstructured(u.Object, j); err != nil {
			return false, err
		}
		t, err = timeLeftForJob(j, &now)
		if err != nil {
			return false, err
		}
	} else if u.GetKind() == "pod" {
		if err = runtime.DefaultUnstructuredConverter.FromUnstructured(u.Object, p); err != nil {
			return false, err
		}
		t, err = timeLeftForPod(p, &now)
		if err != nil {
			return false, err
		}
	} else {
		return false, nil
	}
	if u.GetDeletionTimestamp() != nil || !needsCleanup(u) {
		return false, nil
	}

	// TTL has expired
	if *t <= 0 {
		return true, nil
	}

	tc.enqueueAfter(u, *t)
	return false, nil
}

// needsCleanup checks whether a resource has finished and has a TTL set.
func needsCleanup(obj interface{}) bool {
	if job, ok := obj.(*batch.Job); ok {
		return needsCleanupJob(job)
	}
	if pod, ok := obj.(*v1.Pod); ok {
		return needsCleanupPod(pod)
	}
	return false
}

// needsCleanupPod checks whether a Pod has finished and has a TTL set.
func needsCleanupPod(p *v1.Pod) bool {
	return p.Spec.TTLSecondsAfterFinished != nil && isPodFinished(p)
}

// needsCleanupJob checks whether a Job has finished and has a TTL set.
func needsCleanupJob(j *batch.Job) bool {
	return j.Spec.TTLSecondsAfterFinished != nil && jobutil.IsJobFinished(j)
}

func getFinishAndExpireTimeForJob(j *batch.Job) (*time.Time, *time.Time, error) {
	if !needsCleanup(j) {
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
	if !needsCleanup(p) {
		return nil, nil, fmt.Errorf("Pod %s/%s should not be cleaned up", p.Namespace, p.Name)
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
