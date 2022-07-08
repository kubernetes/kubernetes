/*
Copyright 2015 The Kubernetes Authors.

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

package job

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/feature"
	batchinformers "k8s.io/client-go/informers/batch/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	batchv1listers "k8s.io/client-go/listers/batch/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/job/metrics"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/integer"
	"k8s.io/utils/pointer"
)

// podUpdateBatchPeriod is the batch period to hold pod updates before syncing
// a Job. It is used if the feature gate JobReadyPods is enabled.
const podUpdateBatchPeriod = time.Second

// controllerKind contains the schema.GroupVersionKind for this controller type.
var controllerKind = batch.SchemeGroupVersion.WithKind("Job")

var (
	// DefaultJobBackOff is the default backoff period. Exported for tests.
	DefaultJobBackOff = 10 * time.Second
	// MaxJobBackOff is the max backoff period. Exported for tests.
	MaxJobBackOff = 360 * time.Second
	// MaxUncountedPods is the maximum size the slices in
	// .status.uncountedTerminatedPods should have to keep their representation
	// roughly below 20 KB. Exported for tests
	MaxUncountedPods = 500
	// MaxPodCreateDeletePerSync is the maximum number of pods that can be
	// created or deleted in a single sync call. Exported for tests.
	MaxPodCreateDeletePerSync = 500
)

// Controller ensures that all Job objects have corresponding pods to
// run their configured workload.
type Controller struct {
	kubeClient clientset.Interface
	podControl controller.PodControlInterface

	// To allow injection of the following for testing.
	updateStatusHandler func(ctx context.Context, job *batch.Job) (*batch.Job, error)
	patchJobHandler     func(ctx context.Context, job *batch.Job, patch []byte) error
	syncHandler         func(ctx context.Context, jobKey string) (bool, error)
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// jobStoreSynced returns true if the job store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jobStoreSynced cache.InformerSynced

	// A TTLCache of pod creates/deletes each rc expects to see
	expectations controller.ControllerExpectationsInterface

	// finalizerExpectations tracks the Pod UIDs for which the controller
	// expects to observe the tracking finalizer removed.
	finalizerExpectations *uidTrackingExpectations

	// A store of jobs
	jobLister batchv1listers.JobLister

	// A store of pods, populated by the podController
	podStore corelisters.PodLister

	// Jobs that need to be updated
	queue workqueue.RateLimitingInterface

	// Orphan deleted pods that still have a Job tracking finalizer to be removed
	orphanQueue workqueue.RateLimitingInterface

	broadcaster record.EventBroadcaster
	recorder    record.EventRecorder

	podUpdateBatchPeriod time.Duration
}

// NewController creates a new Job controller that keeps the relevant pods
// in sync with their corresponding Job objects.
func NewController(podInformer coreinformers.PodInformer, jobInformer batchinformers.JobInformer, kubeClient clientset.Interface) *Controller {
	eventBroadcaster := record.NewBroadcaster()

	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("job_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter())
	}

	jm := &Controller{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "job-controller"}),
		},
		expectations:          controller.NewControllerExpectations(),
		finalizerExpectations: newUIDTrackingExpectations(),
		queue:                 workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultJobBackOff, MaxJobBackOff), "job"),
		orphanQueue:           workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultJobBackOff, MaxJobBackOff), "job_orphan_pod"),
		broadcaster:           eventBroadcaster,
		recorder:              eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "job-controller"}),
	}
	if feature.DefaultFeatureGate.Enabled(features.JobReadyPods) {
		jm.podUpdateBatchPeriod = podUpdateBatchPeriod
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			jm.enqueueController(obj, true)
		},
		UpdateFunc: jm.updateJob,
		DeleteFunc: jm.deleteJob,
	})
	jm.jobLister = jobInformer.Lister()
	jm.jobStoreSynced = jobInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jm.addPod,
		UpdateFunc: jm.updatePod,
		DeleteFunc: func(obj interface{}) {
			jm.deletePod(obj, true)
		},
	})
	jm.podStore = podInformer.Lister()
	jm.podStoreSynced = podInformer.Informer().HasSynced

	jm.updateStatusHandler = jm.updateJobStatus
	jm.patchJobHandler = jm.patchJob
	jm.syncHandler = jm.syncJob

	metrics.Register()

	return jm
}

// Run the main goroutine responsible for watching and syncing jobs.
func (jm *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	// Start events processing pipeline.
	jm.broadcaster.StartStructuredLogging(0)
	jm.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: jm.kubeClient.CoreV1().Events("")})
	defer jm.broadcaster.Shutdown()

	defer jm.queue.ShutDown()
	defer jm.orphanQueue.ShutDown()

	klog.Infof("Starting job controller")
	defer klog.Infof("Shutting down job controller")

	if !cache.WaitForNamedCacheSync("job", ctx.Done(), jm.podStoreSynced, jm.jobStoreSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, jm.worker, time.Second)
	}

	go wait.UntilWithContext(ctx, jm.orphanWorker, time.Second)

	<-ctx.Done()
}

// getPodJobs returns a list of Jobs that potentially match a Pod.
func (jm *Controller) getPodJobs(pod *v1.Pod) []*batch.Job {
	jobs, err := jm.jobLister.GetPodJobs(pod)
	if err != nil {
		return nil
	}
	if len(jobs) > 1 {
		// ControllerRef will ensure we don't do anything crazy, but more than one
		// item in this list nevertheless constitutes user error.
		utilruntime.HandleError(fmt.Errorf("user error! more than one job is selecting pods with labels: %+v", pod.Labels))
	}
	ret := make([]*batch.Job, 0, len(jobs))
	for i := range jobs {
		ret = append(ret, &jobs[i])
	}
	return ret
}

// resolveControllerRef returns the controller referenced by a ControllerRef,
// or nil if the ControllerRef could not be resolved to a matching controller
// of the correct Kind.
func (jm *Controller) resolveControllerRef(namespace string, controllerRef *metav1.OwnerReference) *batch.Job {
	// We can't look up by UID, so look up by Name and then verify UID.
	// Don't even try to look up by Name if it's the wrong Kind.
	if controllerRef.Kind != controllerKind.Kind {
		return nil
	}
	job, err := jm.jobLister.Jobs(namespace).Get(controllerRef.Name)
	if err != nil {
		return nil
	}
	if job.UID != controllerRef.UID {
		// The controller we found with this Name is not the same one that the
		// ControllerRef points to.
		return nil
	}
	return job
}

// When a pod is created, enqueue the controller that manages it and update its expectations.
func (jm *Controller) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		jm.deletePod(pod, false)
		return
	}

	// If it has a ControllerRef, that's all that matters.
	if controllerRef := metav1.GetControllerOf(pod); controllerRef != nil {
		job := jm.resolveControllerRef(pod.Namespace, controllerRef)
		if job == nil {
			return
		}
		jobKey, err := controller.KeyFunc(job)
		if err != nil {
			return
		}
		jm.expectations.CreationObserved(jobKey)
		jm.enqueueControllerPodUpdate(job, true)
		return
	}

	// Otherwise, it's an orphan.
	// Clean the finalizer.
	if hasJobTrackingFinalizer(pod) {
		jm.enqueueOrphanPod(pod)
	}
	// Get a list of all matching controllers and sync
	// them to see if anyone wants to adopt it.
	// DO NOT observe creation because no controller should be waiting for an
	// orphan.
	for _, job := range jm.getPodJobs(pod) {
		jm.enqueueControllerPodUpdate(job, true)
	}
}

// When a pod is updated, figure out what job/s manage it and wake them up.
// If the labels of the pod have changed we need to awaken both the old
// and new job. old and cur must be *v1.Pod types.
func (jm *Controller) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs.
		return
	}
	if curPod.DeletionTimestamp != nil {
		// when a pod is deleted gracefully it's deletion timestamp is first modified to reflect a grace period,
		// and after such time has passed, the kubelet actually deletes it from the store. We receive an update
		// for modification of the deletion timestamp and expect an job to create more pods asap, not wait
		// until the kubelet actually deletes the pod.
		jm.deletePod(curPod, false)
		return
	}

	// the only time we want the backoff to kick-in, is when the pod failed
	immediate := curPod.Status.Phase != v1.PodFailed

	// Don't check if oldPod has the finalizer, as during ownership transfer
	// finalizers might be re-added and removed again in behalf of the new owner.
	// If all those Pod updates collapse into a single event, the finalizer
	// might be removed in oldPod and curPod. We want to record the latest
	// state.
	finalizerRemoved := !hasJobTrackingFinalizer(curPod)
	curControllerRef := metav1.GetControllerOf(curPod)
	oldControllerRef := metav1.GetControllerOf(oldPod)
	controllerRefChanged := !reflect.DeepEqual(curControllerRef, oldControllerRef)
	if controllerRefChanged && oldControllerRef != nil {
		// The ControllerRef was changed. Sync the old controller, if any.
		if job := jm.resolveControllerRef(oldPod.Namespace, oldControllerRef); job != nil {
			if finalizerRemoved {
				key, err := controller.KeyFunc(job)
				if err == nil {
					jm.finalizerExpectations.finalizerRemovalObserved(key, string(curPod.UID))
				}
			}
			jm.enqueueControllerPodUpdate(job, immediate)
		}
	}

	// If it has a ControllerRef, that's all that matters.
	if curControllerRef != nil {
		job := jm.resolveControllerRef(curPod.Namespace, curControllerRef)
		if job == nil {
			return
		}
		if finalizerRemoved {
			key, err := controller.KeyFunc(job)
			if err == nil {
				jm.finalizerExpectations.finalizerRemovalObserved(key, string(curPod.UID))
			}
		}
		jm.enqueueControllerPodUpdate(job, immediate)
		return
	}

	// Otherwise, it's an orphan.
	// Clean the finalizer.
	if hasJobTrackingFinalizer(curPod) {
		jm.enqueueOrphanPod(curPod)
	}
	// If anything changed, sync matching controllers
	// to see if anyone wants to adopt it now.
	labelChanged := !reflect.DeepEqual(curPod.Labels, oldPod.Labels)
	if labelChanged || controllerRefChanged {
		for _, job := range jm.getPodJobs(curPod) {
			jm.enqueueControllerPodUpdate(job, immediate)
		}
	}
}

// When a pod is deleted, enqueue the job that manages the pod and update its expectations.
// obj could be an *v1.Pod, or a DeleteFinalStateUnknown marker item.
func (jm *Controller) deletePod(obj interface{}, final bool) {
	pod, ok := obj.(*v1.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new job will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %+v", obj))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a pod %+v", obj))
			return
		}
	}

	controllerRef := metav1.GetControllerOf(pod)
	hasFinalizer := hasJobTrackingFinalizer(pod)
	if controllerRef == nil {
		// No controller should care about orphans being deleted.
		// But this pod might have belonged to a Job and the GC removed the reference.
		if hasFinalizer {
			jm.enqueueOrphanPod(pod)
		}
		return
	}
	job := jm.resolveControllerRef(pod.Namespace, controllerRef)
	if job == nil || IsJobFinished(job) {
		// syncJob will not remove this finalizer.
		if hasFinalizer {
			jm.enqueueOrphanPod(pod)
		}
		return
	}
	jobKey, err := controller.KeyFunc(job)
	if err != nil {
		return
	}
	jm.expectations.DeletionObserved(jobKey)

	// Consider the finalizer removed if this is the final delete. Otherwise,
	// it's an update for the deletion timestamp, then check finalizer.
	if final || !hasFinalizer {
		jm.finalizerExpectations.finalizerRemovalObserved(jobKey, string(pod.UID))
	}

	jm.enqueueControllerPodUpdate(job, true)
}

func (jm *Controller) updateJob(old, cur interface{}) {
	oldJob := old.(*batch.Job)
	curJob := cur.(*batch.Job)

	// never return error
	key, err := controller.KeyFunc(curJob)
	if err != nil {
		return
	}
	jm.enqueueController(curJob, true)
	// check if need to add a new rsync for ActiveDeadlineSeconds
	if curJob.Status.StartTime != nil {
		curADS := curJob.Spec.ActiveDeadlineSeconds
		if curADS == nil {
			return
		}
		oldADS := oldJob.Spec.ActiveDeadlineSeconds
		if oldADS == nil || *oldADS != *curADS {
			now := metav1.Now()
			start := curJob.Status.StartTime.Time
			passed := now.Time.Sub(start)
			total := time.Duration(*curADS) * time.Second
			// AddAfter will handle total < passed
			jm.queue.AddAfter(key, total-passed)
			klog.V(4).Infof("job %q ActiveDeadlineSeconds updated, will rsync after %d seconds", key, total-passed)
		}
	}
}

// deleteJob enqueues the job and all the pods associated with it that still
// have a finalizer.
func (jm *Controller) deleteJob(obj interface{}) {
	jm.enqueueController(obj, true)
	jobObj, ok := obj.(*batch.Job)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %+v", obj))
			return
		}
		jobObj, ok = tombstone.Obj.(*batch.Job)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a job %+v", obj))
			return
		}
	}
	// Listing pods shouldn't really fail, as we are just querying the informer cache.
	selector, err := metav1.LabelSelectorAsSelector(jobObj.Spec.Selector)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("parsing deleted job selector: %v", err))
		return
	}
	pods, _ := jm.podStore.Pods(jobObj.Namespace).List(selector)
	for _, pod := range pods {
		if metav1.IsControlledBy(pod, jobObj) && hasJobTrackingFinalizer(pod) {
			jm.enqueueOrphanPod(pod)
		}
	}
}

// obj could be an *batch.Job, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (jm *Controller) enqueueController(obj interface{}, immediate bool) {
	jm.enqueueControllerDelayed(obj, immediate, 0)
}

func (jm *Controller) enqueueControllerPodUpdate(obj interface{}, immediate bool) {
	jm.enqueueControllerDelayed(obj, immediate, jm.podUpdateBatchPeriod)
}

func (jm *Controller) enqueueControllerDelayed(obj interface{}, immediate bool, delay time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	backoff := delay
	if !immediate {
		backoff = getBackoff(jm.queue, key)
	}

	// TODO: Handle overlapping controllers better. Either disallow them at admission time or
	// deterministically avoid syncing controllers that fight over pods. Currently, we only
	// ensure that the same controller is synced for a given pod. When we periodically relist
	// all controllers there will still be some replica instability. One way to handle this is
	// by querying the store for all controllers that this rc overlaps, as well as all
	// controllers that overlap this rc, and sorting them.
	klog.Infof("enqueueing job %s", key)
	jm.queue.AddAfter(key, backoff)
}

func (jm *Controller) enqueueOrphanPod(obj *v1.Pod) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}
	jm.orphanQueue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (jm *Controller) worker(ctx context.Context) {
	for jm.processNextWorkItem(ctx) {
	}
}

func (jm *Controller) processNextWorkItem(ctx context.Context) bool {
	key, quit := jm.queue.Get()
	if quit {
		return false
	}
	defer jm.queue.Done(key)

	forget, err := jm.syncHandler(ctx, key.(string))
	if err == nil {
		if forget {
			jm.queue.Forget(key)
		}
		return true
	}

	utilruntime.HandleError(fmt.Errorf("syncing job: %w", err))
	jm.queue.AddRateLimited(key)

	return true
}

func (jm *Controller) orphanWorker(ctx context.Context) {
	for jm.processNextOrphanPod(ctx) {
	}
}

func (jm Controller) processNextOrphanPod(ctx context.Context) bool {
	key, quit := jm.orphanQueue.Get()
	if quit {
		return false
	}
	defer jm.orphanQueue.Done(key)
	err := jm.syncOrphanPod(ctx, key.(string))
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Error syncing orphan pod: %v", err))
		jm.orphanQueue.AddRateLimited(key)
	} else {
		jm.orphanQueue.Forget(key)
	}

	return true
}

// syncOrphanPod removes the tracking finalizer from an orphan pod if found.
func (jm Controller) syncOrphanPod(ctx context.Context, key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing orphan pod %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	sharedPod, err := jm.podStore.Pods(ns).Get(name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			klog.V(4).Infof("Orphan pod has been deleted: %v", key)
			return nil
		}
		return err
	}
	// Make sure the pod is still orphaned.
	if controllerRef := metav1.GetControllerOf(sharedPod); controllerRef != nil {
		job := jm.resolveControllerRef(sharedPod.Namespace, controllerRef)
		if job != nil && !IsJobFinished(job) {
			// The pod was adopted. Do not remove finalizer.
			return nil
		}
	}
	if patch := removeTrackingFinalizerPatch(sharedPod); patch != nil {
		if err := jm.podControl.PatchPod(ctx, ns, name, patch); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

// getPodsForJob returns the set of pods that this Job should manage.
// It also reconciles ControllerRef by adopting/orphaning, adding tracking
// finalizers, if enabled.
// Note that the returned Pods are pointers into the cache.
func (jm *Controller) getPodsForJob(ctx context.Context, j *batch.Job, withFinalizers bool) ([]*v1.Pod, error) {
	selector, err := metav1.LabelSelectorAsSelector(j.Spec.Selector)
	if err != nil {
		return nil, fmt.Errorf("couldn't convert Job selector: %v", err)
	}
	// List all pods to include those that don't match the selector anymore
	// but have a ControllerRef pointing to this controller.
	pods, err := jm.podStore.Pods(j.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	// If any adoptions are attempted, we should first recheck for deletion
	// with an uncached quorum read sometime after listing Pods (see #42639).
	canAdoptFunc := controller.RecheckDeletionTimestamp(func(ctx context.Context) (metav1.Object, error) {
		fresh, err := jm.kubeClient.BatchV1().Jobs(j.Namespace).Get(ctx, j.Name, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if fresh.UID != j.UID {
			return nil, fmt.Errorf("original Job %v/%v is gone: got uid %v, wanted %v", j.Namespace, j.Name, fresh.UID, j.UID)
		}
		return fresh, nil
	})
	var finalizers []string
	if withFinalizers {
		finalizers = append(finalizers, batch.JobTrackingFinalizer)
	}
	cm := controller.NewPodControllerRefManager(jm.podControl, j, selector, controllerKind, canAdoptFunc, finalizers...)
	// When adopting Pods, this operation adds an ownerRef and finalizers.
	pods, err = cm.ClaimPods(ctx, pods)
	if err != nil || !withFinalizers {
		return pods, err
	}
	// Set finalizer on adopted pods for the remaining calculations.
	for i, p := range pods {
		adopted := true
		for _, r := range p.OwnerReferences {
			if r.UID == j.UID {
				adopted = false
				break
			}
		}
		if adopted && !hasJobTrackingFinalizer(p) {
			pods[i] = p.DeepCopy()
			pods[i].Finalizers = append(p.Finalizers, batch.JobTrackingFinalizer)
		}
	}
	return pods, err
}

// syncJob will sync the job with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (jm *Controller) syncJob(ctx context.Context, key string) (forget bool, rErr error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing job %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid job key %q: either namespace or name is missing", key)
	}
	sharedJob, err := jm.jobLister.Jobs(ns).Get(name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			klog.V(4).Infof("Job has been deleted: %v", key)
			jm.expectations.DeleteExpectations(key)
			jm.finalizerExpectations.deleteExpectations(key)
			return true, nil
		}
		return false, err
	}
	// make a copy so we don't mutate the shared cache
	job := *sharedJob.DeepCopy()

	// if job was finished previously, we don't want to redo the termination
	if IsJobFinished(&job) {
		return true, nil
	}

	if job.Spec.CompletionMode != nil && *job.Spec.CompletionMode != batch.NonIndexedCompletion && *job.Spec.CompletionMode != batch.IndexedCompletion {
		jm.recorder.Event(&job, v1.EventTypeWarning, "UnknownCompletionMode", "Skipped Job sync because completion mode is unknown")
		return false, nil
	}

	completionMode := string(batch.NonIndexedCompletion)
	if isIndexedJob(&job) {
		completionMode = string(batch.IndexedCompletion)
	}
	action := metrics.JobSyncActionReconciling

	defer func() {
		result := "success"
		if rErr != nil {
			result = "error"
		}

		metrics.JobSyncDurationSeconds.WithLabelValues(completionMode, result, action).Observe(time.Since(startTime).Seconds())
		metrics.JobSyncNum.WithLabelValues(completionMode, result, action).Inc()
	}()

	var expectedRmFinalizers sets.String
	var uncounted *uncountedTerminatedPods
	if trackingUncountedPods(&job) {
		klog.V(4).InfoS("Tracking uncounted Pods with pod finalizers", "job", klog.KObj(&job))
		if job.Status.UncountedTerminatedPods == nil {
			job.Status.UncountedTerminatedPods = &batch.UncountedTerminatedPods{}
		}
		uncounted = newUncountedTerminatedPods(*job.Status.UncountedTerminatedPods)
		expectedRmFinalizers = jm.finalizerExpectations.getExpectedUIDs(key)
	} else if patch := removeTrackingAnnotationPatch(&job); patch != nil {
		if err := jm.patchJobHandler(ctx, &job, patch); err != nil {
			return false, fmt.Errorf("removing tracking finalizer from job %s: %w", key, err)
		}
	}

	// Check the expectations of the job before counting active pods, otherwise a new pod can sneak in
	// and update the expectations after we've retrieved active pods from the store. If a new pod enters
	// the store after we've checked the expectation, the job sync is just deferred till the next relist.
	satisfiedExpectations := jm.expectations.SatisfiedExpectations(key)

	pods, err := jm.getPodsForJob(ctx, &job, uncounted != nil)
	if err != nil {
		return false, err
	}

	activePods := controller.FilterActivePods(pods)
	active := int32(len(activePods))
	succeeded, failed := getStatus(&job, pods, uncounted, expectedRmFinalizers)
	var ready *int32
	if feature.DefaultFeatureGate.Enabled(features.JobReadyPods) {
		ready = pointer.Int32(countReadyPods(activePods))
	}

	// Job first start. Set StartTime only if the job is not in the suspended state.
	if job.Status.StartTime == nil && !jobSuspended(&job) {
		now := metav1.Now()
		job.Status.StartTime = &now
	}

	var manageJobErr error
	var finishedCondition *batch.JobCondition

	jobHasNewFailure := failed > job.Status.Failed
	// new failures happen when status does not reflect the failures and active
	// is different than parallelism, otherwise the previous controller loop
	// failed updating status so even if we pick up failure it is not a new one
	exceedsBackoffLimit := jobHasNewFailure && (active != *job.Spec.Parallelism) &&
		(failed > *job.Spec.BackoffLimit)

	if exceedsBackoffLimit || pastBackoffLimitOnFailure(&job, pods) {
		// check if the number of pod restart exceeds backoff (for restart OnFailure only)
		// OR if the number of failed jobs increased since the last syncJob
		finishedCondition = newCondition(batch.JobFailed, v1.ConditionTrue, "BackoffLimitExceeded", "Job has reached the specified backoff limit")
	} else if pastActiveDeadline(&job) {
		finishedCondition = newCondition(batch.JobFailed, v1.ConditionTrue, "DeadlineExceeded", "Job was active longer than specified deadline")
	} else if job.Spec.ActiveDeadlineSeconds != nil && !jobSuspended(&job) {
		syncDuration := time.Duration(*job.Spec.ActiveDeadlineSeconds)*time.Second - time.Since(job.Status.StartTime.Time)
		klog.V(2).InfoS("Job has activeDeadlineSeconds configuration. Will sync this job again", "job", key, "nextSyncIn", syncDuration)
		jm.queue.AddAfter(key, syncDuration)
	}

	var prevSucceededIndexes, succeededIndexes orderedIntervals
	if isIndexedJob(&job) {
		prevSucceededIndexes, succeededIndexes = calculateSucceededIndexes(&job, pods)
		succeeded = int32(succeededIndexes.total())
	}
	suspendCondChanged := false
	// Remove active pods if Job failed.
	if finishedCondition != nil {
		deleted, err := jm.deleteActivePods(ctx, &job, activePods)
		if uncounted == nil {
			// Legacy behavior: pretend all active pods were successfully removed.
			deleted = active
		} else if deleted != active || !satisfiedExpectations {
			// Can't declare the Job as finished yet, as there might be remaining
			// pod finalizers or pods that are not in the informer's cache yet.
			finishedCondition = nil
		}
		active -= deleted
		failed += deleted
		manageJobErr = err
	} else {
		manageJobCalled := false
		if satisfiedExpectations && job.DeletionTimestamp == nil {
			active, action, manageJobErr = jm.manageJob(ctx, &job, activePods, succeeded, succeededIndexes)
			manageJobCalled = true
		}
		complete := false
		if job.Spec.Completions == nil {
			// This type of job is complete when any pod exits with success.
			// Each pod is capable of
			// determining whether or not the entire Job is done.  Subsequent pods are
			// not expected to fail, but if they do, the failure is ignored.  Once any
			// pod succeeds, the controller waits for remaining pods to finish, and
			// then the job is complete.
			complete = succeeded > 0 && active == 0
		} else {
			// Job specifies a number of completions.  This type of job signals
			// success by having that number of successes.  Since we do not
			// start more pods than there are remaining completions, there should
			// not be any remaining active pods once this count is reached.
			complete = succeeded >= *job.Spec.Completions && active == 0
		}
		if complete {
			finishedCondition = newCondition(batch.JobComplete, v1.ConditionTrue, "", "")
		} else if manageJobCalled {
			// Update the conditions / emit events only if manageJob was called in
			// this syncJob. Otherwise wait for the right syncJob call to make
			// updates.
			if job.Spec.Suspend != nil && *job.Spec.Suspend {
				// Job can be in the suspended state only if it is NOT completed.
				var isUpdated bool
				job.Status.Conditions, isUpdated = ensureJobConditionStatus(job.Status.Conditions, batch.JobSuspended, v1.ConditionTrue, "JobSuspended", "Job suspended")
				if isUpdated {
					suspendCondChanged = true
					jm.recorder.Event(&job, v1.EventTypeNormal, "Suspended", "Job suspended")
				}
			} else {
				// Job not suspended.
				var isUpdated bool
				job.Status.Conditions, isUpdated = ensureJobConditionStatus(job.Status.Conditions, batch.JobSuspended, v1.ConditionFalse, "JobResumed", "Job resumed")
				if isUpdated {
					suspendCondChanged = true
					jm.recorder.Event(&job, v1.EventTypeNormal, "Resumed", "Job resumed")
					// Resumed jobs will always reset StartTime to current time. This is
					// done because the ActiveDeadlineSeconds timer shouldn't go off
					// whilst the Job is still suspended and resetting StartTime is
					// consistent with resuming a Job created in the suspended state.
					// (ActiveDeadlineSeconds is interpreted as the number of seconds a
					// Job is continuously active.)
					now := metav1.Now()
					job.Status.StartTime = &now
				}
			}
		}
	}

	// Check if the number of jobs succeeded increased since the last check. If yes "forget" should be true
	// This logic is linked to the issue: https://github.com/kubernetes/kubernetes/issues/56853 that aims to
	// improve the Job backoff policy when parallelism > 1 and few Jobs failed but others succeed.
	// In this case, we should clear the backoff delay.
	forget = job.Status.Succeeded < succeeded

	if uncounted != nil {
		needsStatusUpdate := suspendCondChanged || active != job.Status.Active || !equalReady(ready, job.Status.Ready)
		job.Status.Active = active
		job.Status.Ready = ready
		err = jm.trackJobStatusAndRemoveFinalizers(ctx, &job, pods, prevSucceededIndexes, *uncounted, expectedRmFinalizers, finishedCondition, needsStatusUpdate)
		if err != nil {
			return false, fmt.Errorf("tracking status: %w", err)
		}
		jobFinished := IsJobFinished(&job)
		if jobHasNewFailure && !jobFinished {
			// returning an error will re-enqueue Job after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for job key %q", key)
		}
		forget = true
		return forget, manageJobErr
	}
	// Legacy path: tracking without finalizers.

	// Ensure that there are no leftover tracking finalizers.
	if err := jm.removeTrackingFinalizersFromAllPods(ctx, pods); err != nil {
		return false, fmt.Errorf("removing disabled finalizers from job pods %s: %w", key, err)
	}

	// no need to update the job if the status hasn't changed since last time
	if job.Status.Active != active || job.Status.Succeeded != succeeded || job.Status.Failed != failed || !equalReady(job.Status.Ready, ready) || suspendCondChanged || finishedCondition != nil {
		job.Status.Active = active
		job.Status.Succeeded = succeeded
		job.Status.Failed = failed
		job.Status.Ready = ready
		if isIndexedJob(&job) {
			job.Status.CompletedIndexes = succeededIndexes.String()
		}
		job.Status.UncountedTerminatedPods = nil
		jm.enactJobFinished(&job, finishedCondition)

		if _, err := jm.updateStatusHandler(ctx, &job); err != nil {
			return forget, err
		}

		if jobHasNewFailure && !IsJobFinished(&job) {
			// returning an error will re-enqueue Job after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for job key %q", key)
		}

		forget = true
	}

	return forget, manageJobErr
}

// deleteActivePods issues deletion for active Pods, preserving finalizers.
// This is done through DELETE calls that set deletion timestamps.
// The method trackJobStatusAndRemoveFinalizers removes the finalizers, after
// which the objects can actually be deleted.
// Returns number of successfully deletions issued.
func (jm *Controller) deleteActivePods(ctx context.Context, job *batch.Job, pods []*v1.Pod) (int32, error) {
	errCh := make(chan error, len(pods))
	successfulDeletes := int32(len(pods))
	wg := sync.WaitGroup{}
	wg.Add(len(pods))
	for i := range pods {
		go func(pod *v1.Pod) {
			defer wg.Done()
			if err := jm.podControl.DeletePod(ctx, job.Namespace, pod.Name, job); err != nil && !apierrors.IsNotFound(err) {
				atomic.AddInt32(&successfulDeletes, -1)
				errCh <- err
				utilruntime.HandleError(err)
			}
		}(pods[i])
	}
	wg.Wait()
	return successfulDeletes, errorFromChannel(errCh)
}

// deleteJobPods deletes the pods, returns the number of successful removals
// and any error.
func (jm *Controller) deleteJobPods(ctx context.Context, job *batch.Job, jobKey string, pods []*v1.Pod) (int32, error) {
	errCh := make(chan error, len(pods))
	successfulDeletes := int32(len(pods))

	failDelete := func(pod *v1.Pod, err error) {
		// Decrement the expected number of deletes because the informer won't observe this deletion
		jm.expectations.DeletionObserved(jobKey)
		if !apierrors.IsNotFound(err) {
			klog.V(2).Infof("Failed to delete Pod", "job", klog.KObj(job), "pod", klog.KObj(pod), "err", err)
			atomic.AddInt32(&successfulDeletes, -1)
			errCh <- err
			utilruntime.HandleError(err)
		}
	}

	wg := sync.WaitGroup{}
	wg.Add(len(pods))
	for i := range pods {
		go func(pod *v1.Pod) {
			defer wg.Done()
			if patch := removeTrackingFinalizerPatch(pod); patch != nil {
				if err := jm.podControl.PatchPod(ctx, pod.Namespace, pod.Name, patch); err != nil {
					failDelete(pod, fmt.Errorf("removing completion finalizer: %w", err))
					return
				}
			}
			if err := jm.podControl.DeletePod(ctx, job.Namespace, pod.Name, job); err != nil {
				failDelete(pod, err)
			}
		}(pods[i])
	}
	wg.Wait()
	return successfulDeletes, errorFromChannel(errCh)
}

// removeTrackingFinalizersFromAllPods removes finalizers from any Job Pod. This is called
// when Job tracking with finalizers is disabled.
func (jm *Controller) removeTrackingFinalizersFromAllPods(ctx context.Context, pods []*v1.Pod) error {
	var podsWithFinalizer []*v1.Pod
	for _, pod := range pods {
		if hasJobTrackingFinalizer(pod) {
			podsWithFinalizer = append(podsWithFinalizer, pod)
		}
	}
	if len(podsWithFinalizer) == 0 {
		return nil
	}
	// Tracking with finalizers is disabled, no need to set expectations.
	_, err := jm.removeTrackingFinalizerFromPods(ctx, "", podsWithFinalizer)
	return err
}

// trackJobStatusAndRemoveFinalizers does:
// 1. Add finished Pods to .status.uncountedTerminatedPods
// 2. Remove the finalizers from the Pods if they completed or were removed
//    or the job was removed.
// 3. Increment job counters for pods that no longer have a finalizer.
// 4. Add Complete condition if satisfied with current counters.
// It does this up to a limited number of Pods so that the size of .status
// doesn't grow too much and this sync doesn't starve other Jobs.
func (jm *Controller) trackJobStatusAndRemoveFinalizers(ctx context.Context, job *batch.Job, pods []*v1.Pod, succeededIndexes orderedIntervals, uncounted uncountedTerminatedPods, expectedRmFinalizers sets.String, finishedCond *batch.JobCondition, needsFlush bool) error {
	isIndexed := isIndexedJob(job)
	var podsToRemoveFinalizer []*v1.Pod
	uncountedStatus := job.Status.UncountedTerminatedPods
	var newSucceededIndexes []int
	if isIndexed {
		// Sort to introduce completed Indexes in order.
		sort.Sort(byCompletionIndex(pods))
	}
	uidsWithFinalizer := make(sets.String, len(pods))
	for _, p := range pods {
		uid := string(p.UID)
		if hasJobTrackingFinalizer(p) && !expectedRmFinalizers.Has(uid) {
			uidsWithFinalizer.Insert(uid)
		}
	}
	// Shallow copy, as it will only be used to detect changes in the counters.
	oldCounters := job.Status
	if cleanUncountedPodsWithoutFinalizers(&job.Status, uidsWithFinalizer) {
		needsFlush = true
	}
	for _, pod := range pods {
		if !hasJobTrackingFinalizer(pod) || expectedRmFinalizers.Has(string(pod.UID)) {
			continue
		}
		podFinished := pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed
		// Terminating pods are counted as failed. This guarantees that orphan Pods
		// count as failures.
		// Active pods are terminated when the job has completed, thus they count as
		// failures as well.
		podTerminating := pod.DeletionTimestamp != nil || finishedCond != nil
		if podFinished || podTerminating || job.DeletionTimestamp != nil {
			podsToRemoveFinalizer = append(podsToRemoveFinalizer, pod)
		}
		if pod.Status.Phase == v1.PodSucceeded {
			if isIndexed {
				// The completion index is enough to avoid recounting succeeded pods.
				// No need to track UIDs.
				ix := getCompletionIndex(pod.Annotations)
				if ix != unknownCompletionIndex && ix < int(*job.Spec.Completions) && !succeededIndexes.has(ix) {
					newSucceededIndexes = append(newSucceededIndexes, ix)
					needsFlush = true
				}
			} else if !uncounted.succeeded.Has(string(pod.UID)) {
				needsFlush = true
				uncountedStatus.Succeeded = append(uncountedStatus.Succeeded, pod.UID)
			}
		} else if pod.Status.Phase == v1.PodFailed || podTerminating {
			ix := getCompletionIndex(pod.Annotations)
			if !uncounted.failed.Has(string(pod.UID)) && (!isIndexed || (ix != unknownCompletionIndex && ix < int(*job.Spec.Completions))) {
				needsFlush = true
				uncountedStatus.Failed = append(uncountedStatus.Failed, pod.UID)
			}
		}
		if len(newSucceededIndexes)+len(uncountedStatus.Succeeded)+len(uncountedStatus.Failed) >= MaxUncountedPods {
			// The controller added enough Pods already to .status.uncountedTerminatedPods
			// We stop counting pods and removing finalizers here to:
			// 1. Ensure that the UIDs representation are under 20 KB.
			// 2. Cap the number of finalizer removals so that syncing of big Jobs
			//    doesn't starve smaller ones.
			//
			// The job will be synced again because the Job status and Pod updates
			// will put the Job back to the work queue.
			break
		}
	}
	if len(newSucceededIndexes) > 0 {
		succeededIndexes = succeededIndexes.withOrderedIndexes(newSucceededIndexes)
		job.Status.Succeeded = int32(succeededIndexes.total())
		job.Status.CompletedIndexes = succeededIndexes.String()
	}
	var err error
	if job, needsFlush, err = jm.flushUncountedAndRemoveFinalizers(ctx, job, podsToRemoveFinalizer, uidsWithFinalizer, &oldCounters, needsFlush); err != nil {
		return err
	}
	if jm.enactJobFinished(job, finishedCond) {
		needsFlush = true
	}
	if needsFlush {
		if _, err := jm.updateStatusHandler(ctx, job); err != nil {
			return fmt.Errorf("removing uncounted pods from status: %w", err)
		}
		recordJobPodFinished(job, oldCounters)
	}
	return nil
}

// flushUncountedAndRemoveFinalizers does:
// 1. flush the Job status that might include new uncounted Pod UIDs.
// 2. perform the removal of finalizers from Pods which are in the uncounted
//    lists.
// 3. update the counters based on the Pods for which it successfully removed
//    the finalizers.
// 4. (if not all removals succeeded) flush Job status again.
// Returns whether there are pending changes in the Job status that need to be
// flushed in subsequent calls.
func (jm *Controller) flushUncountedAndRemoveFinalizers(ctx context.Context, job *batch.Job, podsToRemoveFinalizer []*v1.Pod, uidsWithFinalizer sets.String, oldCounters *batch.JobStatus, needsFlush bool) (*batch.Job, bool, error) {
	var err error
	if needsFlush {
		if job, err = jm.updateStatusHandler(ctx, job); err != nil {
			return job, needsFlush, fmt.Errorf("adding uncounted pods to status: %w", err)
		}
		recordJobPodFinished(job, *oldCounters)
		// Shallow copy, as it will only be used to detect changes in the counters.
		*oldCounters = job.Status
		needsFlush = false
	}
	jobKey, err := controller.KeyFunc(job)
	if err != nil {
		return job, needsFlush, fmt.Errorf("getting job key: %w", err)
	}
	var rmErr error
	if len(podsToRemoveFinalizer) > 0 {
		var rmSucceded []bool
		rmSucceded, rmErr = jm.removeTrackingFinalizerFromPods(ctx, jobKey, podsToRemoveFinalizer)
		for i, p := range podsToRemoveFinalizer {
			if rmSucceded[i] {
				uidsWithFinalizer.Delete(string(p.UID))
			}
		}
	}
	// Failed to remove some finalizers. Attempt to update the status with the
	// partial progress.
	if cleanUncountedPodsWithoutFinalizers(&job.Status, uidsWithFinalizer) {
		needsFlush = true
	}
	if rmErr != nil && needsFlush {
		if job, err := jm.updateStatusHandler(ctx, job); err != nil {
			return job, needsFlush, fmt.Errorf("removing uncounted pods from status: %w", err)
		}
	}
	return job, needsFlush, rmErr
}

// cleanUncountedPodsWithoutFinalizers removes the Pod UIDs from
// .status.uncountedTerminatedPods for which the finalizer was successfully
// removed and increments the corresponding status counters.
// Returns whether there was any status change.
func cleanUncountedPodsWithoutFinalizers(status *batch.JobStatus, uidsWithFinalizer sets.String) bool {
	updated := false
	uncountedStatus := status.UncountedTerminatedPods
	newUncounted := filterInUncountedUIDs(uncountedStatus.Succeeded, uidsWithFinalizer)
	if len(newUncounted) != len(uncountedStatus.Succeeded) {
		updated = true
		status.Succeeded += int32(len(uncountedStatus.Succeeded) - len(newUncounted))
		uncountedStatus.Succeeded = newUncounted
	}
	newUncounted = filterInUncountedUIDs(uncountedStatus.Failed, uidsWithFinalizer)
	if len(newUncounted) != len(uncountedStatus.Failed) {
		updated = true
		status.Failed += int32(len(uncountedStatus.Failed) - len(newUncounted))
		uncountedStatus.Failed = newUncounted
	}
	return updated
}

// removeTrackingFinalizerFromPods removes tracking finalizers from Pods and
// returns an array of booleans where the i-th value is true if the finalizer
// of the i-th Pod was successfully removed (if the pod was deleted when this
// function was called, it's considered as the finalizer was removed successfully).
func (jm *Controller) removeTrackingFinalizerFromPods(ctx context.Context, jobKey string, pods []*v1.Pod) ([]bool, error) {
	errCh := make(chan error, len(pods))
	succeeded := make([]bool, len(pods))
	uids := make([]string, len(pods))
	for i, p := range pods {
		uids[i] = string(p.UID)
	}
	if jobKey != "" {
		err := jm.finalizerExpectations.expectFinalizersRemoved(jobKey, uids)
		if err != nil {
			return succeeded, fmt.Errorf("setting expected removed finalizers: %w", err)
		}
	}
	wg := sync.WaitGroup{}
	wg.Add(len(pods))
	for i := range pods {
		go func(i int) {
			pod := pods[i]
			defer wg.Done()
			if patch := removeTrackingFinalizerPatch(pod); patch != nil {
				if err := jm.podControl.PatchPod(ctx, pod.Namespace, pod.Name, patch); err != nil {
					// In case of any failure, we don't expect a Pod update for the
					// finalizer removed. Clear expectation now.
					if jobKey != "" {
						jm.finalizerExpectations.finalizerRemovalObserved(jobKey, string(pod.UID))
					}
					if !apierrors.IsNotFound(err) {
						errCh <- err
						utilruntime.HandleError(err)
						return
					}
				}
				succeeded[i] = true
			}
		}(i)
	}
	wg.Wait()

	return succeeded, errorFromChannel(errCh)
}

// enactJobFinished adds the Complete or Failed condition and records events.
// Returns whether the Job was considered finished.
func (jm *Controller) enactJobFinished(job *batch.Job, finishedCond *batch.JobCondition) bool {
	if finishedCond == nil {
		return false
	}
	if uncounted := job.Status.UncountedTerminatedPods; uncounted != nil {
		if len(uncounted.Succeeded) > 0 || len(uncounted.Failed) > 0 {
			return false
		}
	}
	completionMode := string(batch.NonIndexedCompletion)
	if isIndexedJob(job) {
		completionMode = string(*job.Spec.CompletionMode)
	}
	job.Status.Conditions, _ = ensureJobConditionStatus(job.Status.Conditions, finishedCond.Type, finishedCond.Status, finishedCond.Reason, finishedCond.Message)
	if finishedCond.Type == batch.JobComplete {
		if job.Spec.Completions != nil && job.Status.Succeeded > *job.Spec.Completions {
			jm.recorder.Event(job, v1.EventTypeWarning, "TooManySucceededPods", "Too many succeeded pods running after completion count reached")
		}
		job.Status.CompletionTime = &finishedCond.LastTransitionTime
		jm.recorder.Event(job, v1.EventTypeNormal, "Completed", "Job completed")
		metrics.JobFinishedNum.WithLabelValues(completionMode, "succeeded").Inc()
	} else {
		jm.recorder.Event(job, v1.EventTypeWarning, finishedCond.Reason, finishedCond.Message)
		metrics.JobFinishedNum.WithLabelValues(completionMode, "failed").Inc()
	}
	return true
}

func filterInUncountedUIDs(uncounted []types.UID, include sets.String) []types.UID {
	var newUncounted []types.UID
	for _, uid := range uncounted {
		if include.Has(string(uid)) {
			newUncounted = append(newUncounted, uid)
		}
	}
	return newUncounted
}

// pastBackoffLimitOnFailure checks if container restartCounts sum exceeds BackoffLimit
// this method applies only to pods with restartPolicy == OnFailure
func pastBackoffLimitOnFailure(job *batch.Job, pods []*v1.Pod) bool {
	if job.Spec.Template.Spec.RestartPolicy != v1.RestartPolicyOnFailure {
		return false
	}
	result := int32(0)
	for i := range pods {
		po := pods[i]
		if po.Status.Phase == v1.PodRunning || po.Status.Phase == v1.PodPending {
			for j := range po.Status.InitContainerStatuses {
				stat := po.Status.InitContainerStatuses[j]
				result += stat.RestartCount
			}
			for j := range po.Status.ContainerStatuses {
				stat := po.Status.ContainerStatuses[j]
				result += stat.RestartCount
			}
		}
	}
	if *job.Spec.BackoffLimit == 0 {
		return result > 0
	}
	return result >= *job.Spec.BackoffLimit
}

// pastActiveDeadline checks if job has ActiveDeadlineSeconds field set and if
// it is exceeded. If the job is currently suspended, the function will always
// return false.
func pastActiveDeadline(job *batch.Job) bool {
	if job.Spec.ActiveDeadlineSeconds == nil || job.Status.StartTime == nil || jobSuspended(job) {
		return false
	}
	now := metav1.Now()
	start := job.Status.StartTime.Time
	duration := now.Time.Sub(start)
	allowedDuration := time.Duration(*job.Spec.ActiveDeadlineSeconds) * time.Second
	return duration >= allowedDuration
}

func newCondition(conditionType batch.JobConditionType, status v1.ConditionStatus, reason, message string) *batch.JobCondition {
	return &batch.JobCondition{
		Type:               conditionType,
		Status:             status,
		LastProbeTime:      metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// getStatus returns number of succeeded and failed pods running a job
func getStatus(job *batch.Job, pods []*v1.Pod, uncounted *uncountedTerminatedPods, expectedRmFinalizers sets.String) (succeeded, failed int32) {
	if uncounted != nil {
		succeeded = job.Status.Succeeded
		failed = job.Status.Failed
	}
	succeeded += int32(countValidPodsWithFilter(job, pods, uncounted.Succeeded(), expectedRmFinalizers, func(p *v1.Pod) bool {
		return p.Status.Phase == v1.PodSucceeded
	}))
	failed += int32(countValidPodsWithFilter(job, pods, uncounted.Failed(), expectedRmFinalizers, func(p *v1.Pod) bool {
		if p.Status.Phase == v1.PodFailed {
			return true
		}
		// When tracking with finalizers: counting deleted Pods as failures to
		// account for orphan Pods that never have a chance to reach the Failed
		// phase.
		return uncounted != nil && p.DeletionTimestamp != nil && p.Status.Phase != v1.PodSucceeded
	}))
	return succeeded, failed
}

// jobSuspended returns whether a Job is suspended while taking the feature
// gate into account.
func jobSuspended(job *batch.Job) bool {
	return job.Spec.Suspend != nil && *job.Spec.Suspend
}

// manageJob is the core method responsible for managing the number of running
// pods according to what is specified in the job.Spec.
// Does NOT modify <activePods>.
func (jm *Controller) manageJob(ctx context.Context, job *batch.Job, activePods []*v1.Pod, succeeded int32, succeededIndexes []interval) (int32, string, error) {
	active := int32(len(activePods))
	parallelism := *job.Spec.Parallelism
	jobKey, err := controller.KeyFunc(job)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for job %#v: %v", job, err))
		return 0, metrics.JobSyncActionTracking, nil
	}

	if jobSuspended(job) {
		klog.V(4).InfoS("Deleting all active pods in suspended job", "job", klog.KObj(job), "active", active)
		podsToDelete := activePodsForRemoval(job, activePods, int(active))
		jm.expectations.ExpectDeletions(jobKey, len(podsToDelete))
		removed, err := jm.deleteJobPods(ctx, job, jobKey, podsToDelete)
		active -= removed
		return active, metrics.JobSyncActionPodsDeleted, err
	}

	wantActive := int32(0)
	if job.Spec.Completions == nil {
		// Job does not specify a number of completions.  Therefore, number active
		// should be equal to parallelism, unless the job has seen at least
		// once success, in which leave whatever is running, running.
		if succeeded > 0 {
			wantActive = active
		} else {
			wantActive = parallelism
		}
	} else {
		// Job specifies a specific number of completions.  Therefore, number
		// active should not ever exceed number of remaining completions.
		wantActive = *job.Spec.Completions - succeeded
		if wantActive > parallelism {
			wantActive = parallelism
		}
		if wantActive < 0 {
			wantActive = 0
		}
	}

	rmAtLeast := active - wantActive
	if rmAtLeast < 0 {
		rmAtLeast = 0
	}
	podsToDelete := activePodsForRemoval(job, activePods, int(rmAtLeast))
	if len(podsToDelete) > MaxPodCreateDeletePerSync {
		podsToDelete = podsToDelete[:MaxPodCreateDeletePerSync]
	}
	if len(podsToDelete) > 0 {
		jm.expectations.ExpectDeletions(jobKey, len(podsToDelete))
		klog.V(4).InfoS("Too many pods running for job", "job", klog.KObj(job), "deleted", len(podsToDelete), "target", wantActive)
		removed, err := jm.deleteJobPods(ctx, job, jobKey, podsToDelete)
		active -= removed
		// While it is possible for a Job to require both pod creations and
		// deletions at the same time (e.g. indexed Jobs with repeated indexes), we
		// restrict ourselves to either just pod deletion or pod creation in any
		// given sync cycle. Of these two, pod deletion takes precedence.
		return active, metrics.JobSyncActionPodsDeleted, err
	}

	if active < wantActive {
		diff := wantActive - active
		if diff > int32(MaxPodCreateDeletePerSync) {
			diff = int32(MaxPodCreateDeletePerSync)
		}

		jm.expectations.ExpectCreations(jobKey, int(diff))
		errCh := make(chan error, diff)
		klog.V(4).Infof("Too few pods running job %q, need %d, creating %d", jobKey, wantActive, diff)

		wait := sync.WaitGroup{}

		var indexesToAdd []int
		if isIndexedJob(job) {
			indexesToAdd = firstPendingIndexes(activePods, succeededIndexes, int(diff), int(*job.Spec.Completions))
			diff = int32(len(indexesToAdd))
		}
		active += diff

		podTemplate := job.Spec.Template.DeepCopy()
		if isIndexedJob(job) {
			addCompletionIndexEnvVariables(podTemplate)
		}
		if trackingUncountedPods(job) {
			podTemplate.Finalizers = appendJobCompletionFinalizerIfNotFound(podTemplate.Finalizers)
		}

		// Batch the pod creates. Batch sizes start at SlowStartInitialBatchSize
		// and double with each successful iteration in a kind of "slow start".
		// This handles attempts to start large numbers of pods that would
		// likely all fail with the same error. For example a project with a
		// low quota that attempts to create a large number of pods will be
		// prevented from spamming the API service with the pod create requests
		// after one of its pods fails.  Conveniently, this also prevents the
		// event spam that those failures would generate.
		for batchSize := int32(integer.IntMin(int(diff), controller.SlowStartInitialBatchSize)); diff > 0; batchSize = integer.Int32Min(2*batchSize, diff) {
			errorCount := len(errCh)
			wait.Add(int(batchSize))
			for i := int32(0); i < batchSize; i++ {
				completionIndex := unknownCompletionIndex
				if len(indexesToAdd) > 0 {
					completionIndex = indexesToAdd[0]
					indexesToAdd = indexesToAdd[1:]
				}
				go func() {
					template := podTemplate
					generateName := ""
					if completionIndex != unknownCompletionIndex {
						template = podTemplate.DeepCopy()
						addCompletionIndexAnnotation(template, completionIndex)
						template.Spec.Hostname = fmt.Sprintf("%s-%d", job.Name, completionIndex)
						generateName = podGenerateNameWithIndex(job.Name, completionIndex)
					}
					defer wait.Done()
					err := jm.podControl.CreatePodsWithGenerateName(ctx, job.Namespace, template, job, metav1.NewControllerRef(job, controllerKind), generateName)
					if err != nil {
						if apierrors.HasStatusCause(err, v1.NamespaceTerminatingCause) {
							// If the namespace is being torn down, we can safely ignore
							// this error since all subsequent creations will fail.
							return
						}
					}
					if err != nil {
						defer utilruntime.HandleError(err)
						// Decrement the expected number of creates because the informer won't observe this pod
						klog.V(2).Infof("Failed creation, decrementing expectations for job %q/%q", job.Namespace, job.Name)
						jm.expectations.CreationObserved(jobKey)
						atomic.AddInt32(&active, -1)
						errCh <- err
					}
				}()
			}
			wait.Wait()
			// any skipped pods that we never attempted to start shouldn't be expected.
			skippedPods := diff - batchSize
			if errorCount < len(errCh) && skippedPods > 0 {
				klog.V(2).Infof("Slow-start failure. Skipping creation of %d pods, decrementing expectations for job %q/%q", skippedPods, job.Namespace, job.Name)
				active -= skippedPods
				for i := int32(0); i < skippedPods; i++ {
					// Decrement the expected number of creates because the informer won't observe this pod
					jm.expectations.CreationObserved(jobKey)
				}
				// The skipped pods will be retried later. The next controller resync will
				// retry the slow start process.
				break
			}
			diff -= batchSize
		}
		return active, metrics.JobSyncActionPodsCreated, errorFromChannel(errCh)
	}

	return active, metrics.JobSyncActionTracking, nil
}

// activePodsForRemoval returns Pods that should be removed because there
// are too many pods running or, if this is an indexed job, there are repeated
// indexes or invalid indexes or some pods don't have indexes.
// Sorts candidate pods in the order such that not-ready < ready, unscheduled
// < scheduled, and pending < running. This ensures that we delete pods
// in the earlier stages whenever possible.
func activePodsForRemoval(job *batch.Job, pods []*v1.Pod, rmAtLeast int) []*v1.Pod {
	var rm, left []*v1.Pod

	if isIndexedJob(job) {
		rm = make([]*v1.Pod, 0, rmAtLeast)
		left = make([]*v1.Pod, 0, len(pods)-rmAtLeast)
		rm, left = appendDuplicatedIndexPodsForRemoval(rm, left, pods, int(*job.Spec.Completions))
	} else {
		left = pods
	}

	if len(rm) < rmAtLeast {
		sort.Sort(controller.ActivePods(left))
		rm = append(rm, left[:rmAtLeast-len(rm)]...)
	}
	return rm
}

// updateJobStatus calls the API to update the job status.
func (jm *Controller) updateJobStatus(ctx context.Context, job *batch.Job) (*batch.Job, error) {
	return jm.kubeClient.BatchV1().Jobs(job.Namespace).UpdateStatus(ctx, job, metav1.UpdateOptions{})
}

func (jm *Controller) patchJob(ctx context.Context, job *batch.Job, data []byte) error {
	_, err := jm.kubeClient.BatchV1().Jobs(job.Namespace).Patch(
		ctx, job.Name, types.StrategicMergePatchType, data, metav1.PatchOptions{})
	return err
}

func getBackoff(queue workqueue.RateLimitingInterface, key interface{}) time.Duration {
	exp := queue.NumRequeues(key)

	if exp <= 0 {
		return time.Duration(0)
	}

	// The backoff is capped such that 'calculated' value never overflows.
	backoff := float64(DefaultJobBackOff.Nanoseconds()) * math.Pow(2, float64(exp-1))
	if backoff > math.MaxInt64 {
		return MaxJobBackOff
	}

	calculated := time.Duration(backoff)
	if calculated > MaxJobBackOff {
		return MaxJobBackOff
	}
	return calculated
}

// countValidPodsWithFilter returns number of valid pods that pass the filter.
// Pods are valid if they have a finalizer and, for Indexed Jobs, a valid
// completion index.
func countValidPodsWithFilter(job *batch.Job, pods []*v1.Pod, uncounted sets.String, expectedRmFinalizers sets.String, filter func(*v1.Pod) bool) int {
	result := len(uncounted)
	for _, p := range pods {
		uid := string(p.UID)
		// Pods that don't have a completion finalizer are in the uncounted set or
		// have already been accounted for in the Job status.
		if uncounted != nil && (!hasJobTrackingFinalizer(p) || uncounted.Has(uid) || expectedRmFinalizers.Has(uid)) {
			continue
		}
		if isIndexedJob(job) {
			idx := getCompletionIndex(p.Annotations)
			if idx == unknownCompletionIndex || idx >= int(*job.Spec.Completions) {
				continue
			}
		}
		if filter(p) {
			result++
		}
	}
	return result
}

func trackingUncountedPods(job *batch.Job) bool {
	return feature.DefaultFeatureGate.Enabled(features.JobTrackingWithFinalizers) && hasJobTrackingAnnotation(job)
}

func hasJobTrackingAnnotation(job *batch.Job) bool {
	if job.Annotations == nil {
		return false
	}
	_, ok := job.Annotations[batch.JobTrackingFinalizer]
	return ok
}

func appendJobCompletionFinalizerIfNotFound(finalizers []string) []string {
	for _, fin := range finalizers {
		if fin == batch.JobTrackingFinalizer {
			return finalizers
		}
	}
	return append(finalizers, batch.JobTrackingFinalizer)
}

func removeTrackingFinalizerPatch(pod *v1.Pod) []byte {
	if !hasJobTrackingFinalizer(pod) {
		return nil
	}
	patch := map[string]interface{}{
		"metadata": map[string]interface{}{
			"$deleteFromPrimitiveList/finalizers": []string{batch.JobTrackingFinalizer},
		},
	}
	patchBytes, _ := json.Marshal(patch)
	return patchBytes
}

func removeTrackingAnnotationPatch(job *batch.Job) []byte {
	if !hasJobTrackingAnnotation(job) {
		return nil
	}
	patch := map[string]interface{}{
		"metadata": map[string]interface{}{
			"annotations": map[string]interface{}{
				batch.JobTrackingFinalizer: nil,
			},
		},
	}
	patchBytes, _ := json.Marshal(patch)
	return patchBytes
}

func hasJobTrackingFinalizer(pod *v1.Pod) bool {
	for _, fin := range pod.Finalizers {
		if fin == batch.JobTrackingFinalizer {
			return true
		}
	}
	return false
}

type uncountedTerminatedPods struct {
	succeeded sets.String
	failed    sets.String
}

func newUncountedTerminatedPods(in batch.UncountedTerminatedPods) *uncountedTerminatedPods {
	obj := uncountedTerminatedPods{
		succeeded: make(sets.String, len(in.Succeeded)),
		failed:    make(sets.String, len(in.Failed)),
	}
	for _, v := range in.Succeeded {
		obj.succeeded.Insert(string(v))
	}
	for _, v := range in.Failed {
		obj.failed.Insert(string(v))
	}
	return &obj
}

func (u *uncountedTerminatedPods) Succeeded() sets.String {
	if u == nil {
		return nil
	}
	return u.succeeded
}

func (u *uncountedTerminatedPods) Failed() sets.String {
	if u == nil {
		return nil
	}
	return u.failed
}

func errorFromChannel(errCh <-chan error) error {
	select {
	case err := <-errCh:
		return err
	default:
	}
	return nil
}

// ensureJobConditionStatus appends or updates an existing job condition of the
// given type with the given status value. Note that this function will not
// append to the conditions list if the new condition's status is false
// (because going from nothing to false is meaningless); it can, however,
// update the status condition to false. The function returns a bool to let the
// caller know if the list was changed (either appended or updated).
func ensureJobConditionStatus(list []batch.JobCondition, cType batch.JobConditionType, status v1.ConditionStatus, reason, message string) ([]batch.JobCondition, bool) {
	if condition := findConditionByType(list, cType); condition != nil {
		if condition.Status != status || condition.Reason != reason || condition.Message != message {
			*condition = *newCondition(cType, status, reason, message)
			return list, true
		}
		return list, false
	}
	// A condition with that type doesn't exist in the list.
	if status != v1.ConditionFalse {
		return append(list, *newCondition(cType, status, reason, message)), true
	}
	return list, false
}

func findConditionByType(list []batch.JobCondition, cType batch.JobConditionType) *batch.JobCondition {
	for i := range list {
		if list[i].Type == cType {
			return &list[i]
		}
	}
	return nil
}

func recordJobPodFinished(job *batch.Job, oldCounters batch.JobStatus) {
	completionMode := completionModeStr(job)
	diff := job.Status.Succeeded - oldCounters.Succeeded
	metrics.JobPodsFinished.WithLabelValues(completionMode, metrics.Succeeded).Add(float64(diff))
	diff = job.Status.Failed - oldCounters.Failed
	metrics.JobPodsFinished.WithLabelValues(completionMode, metrics.Failed).Add(float64(diff))
}

func countReadyPods(pods []*v1.Pod) int32 {
	cnt := int32(0)
	for _, p := range pods {
		if podutil.IsPodReady(p) {
			cnt++
		}
	}
	return cnt
}

func equalReady(a, b *int32) bool {
	if a != nil && b != nil {
		return *a == *b
	}
	return a == b
}
