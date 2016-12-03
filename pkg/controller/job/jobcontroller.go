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
	"fmt"
	"reflect"
	"sort"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	batch "k8s.io/kubernetes/pkg/apis/batch/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	batchv1listers "k8s.io/kubernetes/pkg/client/listers/batch/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"

	"github.com/golang/glog"
)

type JobController struct {
	kubeClient clientset.Interface
	podControl controller.PodControlInterface

	// To allow injection of updateJobStatus for testing.
	updateHandler func(job *batch.Job) error
	syncHandler   func(jobKey string) error
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// jobStoreSynced returns true if the job store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jobStoreSynced cache.InformerSynced

	// A TTLCache of pod creates/deletes each rc expects to see
	expectations controller.ControllerExpectationsInterface

	// A store of jobs
	jobLister batchv1listers.JobLister

	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister

	// Jobs that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder
}

func NewJobController(podInformer cache.SharedIndexInformer, jobInformer informers.JobInformer, kubeClient clientset.Interface) *JobController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.Core().Events("")})

	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("job_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}

	jm := &JobController{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(v1.EventSource{Component: "job-controller"}),
		},
		expectations: controller.NewControllerExpectations(),
		queue:        workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "job"),
		recorder:     eventBroadcaster.NewRecorder(v1.EventSource{Component: "job-controller"}),
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: jm.enqueueController,
		UpdateFunc: func(old, cur interface{}) {
			if job := cur.(*batch.Job); !IsJobFinished(job) {
				jm.enqueueController(job)
			}
		},
		DeleteFunc: jm.enqueueController,
	})
	jm.jobLister = jobInformer.Lister()
	jm.jobStoreSynced = jobInformer.Informer().HasSynced

	podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jm.addPod,
		UpdateFunc: jm.updatePod,
		DeleteFunc: jm.deletePod,
	})
	jm.podStore.Indexer = podInformer.GetIndexer()
	jm.podStoreSynced = podInformer.HasSynced

	jm.updateHandler = jm.updateJobStatus
	jm.syncHandler = jm.syncJob
	return jm
}

// Run the main goroutine responsible for watching and syncing jobs.
func (jm *JobController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer jm.queue.ShutDown()

	if !cache.WaitForCacheSync(stopCh, jm.podStoreSynced, jm.jobStoreSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(jm.worker, time.Second, stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down Job Manager")
}

// getPodJob returns the job managing the given pod.
func (jm *JobController) getPodJob(pod *v1.Pod) *batch.Job {
	jobs, err := jm.jobLister.GetPodJobs(pod)
	if err != nil {
		glog.V(4).Infof("No jobs found for pod %v, job controller will avoid syncing", pod.Name)
		return nil
	}
	if len(jobs) > 1 {
		utilruntime.HandleError(fmt.Errorf("user error! more than one job is selecting pods with labels: %+v", pod.Labels))
		sort.Sort(byCreationTimestamp(jobs))
	}
	return &jobs[0]
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (jm *JobController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		jm.deletePod(pod)
		return
	}
	if job := jm.getPodJob(pod); job != nil {
		jobKey, err := controller.KeyFunc(job)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Couldn't get key for job %#v: %v", job, err))
			return
		}
		jm.expectations.CreationObserved(jobKey)
		jm.enqueueController(job)
	}
}

// When a pod is updated, figure out what job/s manage it and wake them up.
// If the labels of the pod have changed we need to awaken both the old
// and new job. old and cur must be *v1.Pod types.
func (jm *JobController) updatePod(old, cur interface{}) {
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
		jm.deletePod(curPod)
		return
	}
	if job := jm.getPodJob(curPod); job != nil {
		jm.enqueueController(job)
	}
	// Only need to get the old job if the labels changed.
	if !reflect.DeepEqual(curPod.Labels, oldPod.Labels) {
		// If the old and new job are the same, the first one that syncs
		// will set expectations preventing any damage from the second.
		if oldJob := jm.getPodJob(oldPod); oldJob != nil {
			jm.enqueueController(oldJob)
		}
	}
}

// When a pod is deleted, enqueue the job that manages the pod and update its expectations.
// obj could be an *v1.Pod, or a DeletionFinalStateUnknown marker item.
func (jm *JobController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new job will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %+v", obj))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a pod %+v", obj))
			return
		}
	}
	if job := jm.getPodJob(pod); job != nil {
		jobKey, err := controller.KeyFunc(job)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Couldn't get key for job %#v: %v", job, err))
			return
		}
		jm.expectations.DeletionObserved(jobKey)
		jm.enqueueController(job)
	}
}

// obj could be an *batch.Job, or a DeletionFinalStateUnknown marker item.
func (jm *JobController) enqueueController(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	// TODO: Handle overlapping controllers better. Either disallow them at admission time or
	// deterministically avoid syncing controllers that fight over pods. Currently, we only
	// ensure that the same controller is synced for a given pod. When we periodically relist
	// all controllers there will still be some replica instability. One way to handle this is
	// by querying the store for all controllers that this rc overlaps, as well as all
	// controllers that overlap this rc, and sorting them.
	jm.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (jm *JobController) worker() {
	for jm.processNextWorkItem() {
	}
}

func (jm *JobController) processNextWorkItem() bool {
	key, quit := jm.queue.Get()
	if quit {
		return false
	}
	defer jm.queue.Done(key)

	err := jm.syncHandler(key.(string))
	if err == nil {
		jm.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("Error syncing job: %v", err))
	jm.queue.AddRateLimited(key)

	return true
}

// syncJob will sync the job with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (jm *JobController) syncJob(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing job %q (%v)", key, time.Now().Sub(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	if len(ns) == 0 || len(name) == 0 {
		return fmt.Errorf("invalid job key %q: either namespace or name is missing", key)
	}
	sharedJob, err := jm.jobLister.Jobs(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			glog.V(4).Infof("Job has been deleted: %v", key)
			jm.expectations.DeleteExpectations(key)
			return nil
		}
		return err
	}
	job := *sharedJob

	// Check the expectations of the job before counting active pods, otherwise a new pod can sneak in
	// and update the expectations after we've retrieved active pods from the store. If a new pod enters
	// the store after we've checked the expectation, the job sync is just deferred till the next relist.
	jobNeedsSync := jm.expectations.SatisfiedExpectations(key)
	selector, _ := metav1.LabelSelectorAsSelector(job.Spec.Selector)
	pods, err := jm.podStore.Pods(job.Namespace).List(selector)
	if err != nil {
		return err
	}

	activePods := controller.FilterActivePods(pods)
	active := int32(len(activePods))
	succeeded, failed := getStatus(pods)
	conditions := len(job.Status.Conditions)
	if job.Status.StartTime == nil {
		now := metav1.Now()
		job.Status.StartTime = &now
	}
	// if job was finished previously, we don't want to redo the termination
	if IsJobFinished(&job) {
		return nil
	}
	if pastActiveDeadline(&job) {
		// TODO: below code should be replaced with pod termination resulting in
		// pod failures, rather than killing pods. Unfortunately none such solution
		// exists ATM. There's an open discussion in the topic in
		// https://github.com/kubernetes/kubernetes/issues/14602 which might give
		// some sort of solution to above problem.
		// kill remaining active pods
		wait := sync.WaitGroup{}
		wait.Add(int(active))
		for i := int32(0); i < active; i++ {
			go func(ix int32) {
				defer wait.Done()
				if err := jm.podControl.DeletePod(job.Namespace, activePods[ix].Name, &job); err != nil {
					defer utilruntime.HandleError(err)
				}
			}(i)
		}
		wait.Wait()
		// update status values accordingly
		failed += active
		active = 0
		job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobFailed, "DeadlineExceeded", "Job was active longer than specified deadline"))
		jm.recorder.Event(&job, v1.EventTypeNormal, "DeadlineExceeded", "Job was active longer than specified deadline")
	} else {
		if jobNeedsSync && job.DeletionTimestamp == nil {
			active = jm.manageJob(activePods, succeeded, &job)
		}
		completions := succeeded
		complete := false
		if job.Spec.Completions == nil {
			// This type of job is complete when any pod exits with success.
			// Each pod is capable of
			// determining whether or not the entire Job is done.  Subsequent pods are
			// not expected to fail, but if they do, the failure is ignored.  Once any
			// pod succeeds, the controller waits for remaining pods to finish, and
			// then the job is complete.
			if succeeded > 0 && active == 0 {
				complete = true
			}
		} else {
			// Job specifies a number of completions.  This type of job signals
			// success by having that number of successes.  Since we do not
			// start more pods than there are remaining completions, there should
			// not be any remaining active pods once this count is reached.
			if completions >= *job.Spec.Completions {
				complete = true
				if active > 0 {
					jm.recorder.Event(&job, v1.EventTypeWarning, "TooManyActivePods", "Too many active pods running after completion count reached")
				}
				if completions > *job.Spec.Completions {
					jm.recorder.Event(&job, v1.EventTypeWarning, "TooManySucceededPods", "Too many succeeded pods running after completion count reached")
				}
			}
		}
		if complete {
			job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobComplete, "", ""))
			now := metav1.Now()
			job.Status.CompletionTime = &now
		}
	}

	// no need to update the job if the status hasn't changed since last time
	if job.Status.Active != active || job.Status.Succeeded != succeeded || job.Status.Failed != failed || len(job.Status.Conditions) != conditions {
		job.Status.Active = active
		job.Status.Succeeded = succeeded
		job.Status.Failed = failed

		if err := jm.updateHandler(&job); err != nil {
			return err
		}
	}
	return nil
}

// pastActiveDeadline checks if job has ActiveDeadlineSeconds field set and if it is exceeded.
func pastActiveDeadline(job *batch.Job) bool {
	if job.Spec.ActiveDeadlineSeconds == nil || job.Status.StartTime == nil {
		return false
	}
	now := metav1.Now()
	start := job.Status.StartTime.Time
	duration := now.Time.Sub(start)
	allowedDuration := time.Duration(*job.Spec.ActiveDeadlineSeconds) * time.Second
	return duration >= allowedDuration
}

func newCondition(conditionType batch.JobConditionType, reason, message string) batch.JobCondition {
	return batch.JobCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastProbeTime:      metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// getStatus returns no of succeeded and failed pods running a job
func getStatus(pods []*v1.Pod) (succeeded, failed int32) {
	succeeded = int32(filterPods(pods, v1.PodSucceeded))
	failed = int32(filterPods(pods, v1.PodFailed))
	return
}

// manageJob is the core method responsible for managing the number of running
// pods according to what is specified in the job.Spec.
// Does NOT modify <activePods>.
func (jm *JobController) manageJob(activePods []*v1.Pod, succeeded int32, job *batch.Job) int32 {
	var activeLock sync.Mutex
	active := int32(len(activePods))
	parallelism := *job.Spec.Parallelism
	jobKey, err := controller.KeyFunc(job)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for job %#v: %v", job, err))
		return 0
	}

	if active > parallelism {
		diff := active - parallelism
		jm.expectations.ExpectDeletions(jobKey, int(diff))
		glog.V(4).Infof("Too many pods running job %q, need %d, deleting %d", jobKey, parallelism, diff)
		// Sort the pods in the order such that not-ready < ready, unscheduled
		// < scheduled, and pending < running. This ensures that we delete pods
		// in the earlier stages whenever possible.
		sort.Sort(controller.ActivePods(activePods))

		active -= diff
		wait := sync.WaitGroup{}
		wait.Add(int(diff))
		for i := int32(0); i < diff; i++ {
			go func(ix int32) {
				defer wait.Done()
				if err := jm.podControl.DeletePod(job.Namespace, activePods[ix].Name, job); err != nil {
					defer utilruntime.HandleError(err)
					// Decrement the expected number of deletes because the informer won't observe this deletion
					jm.expectations.DeletionObserved(jobKey)
					activeLock.Lock()
					active++
					activeLock.Unlock()
				}
			}(i)
		}
		wait.Wait()

	} else if active < parallelism {
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
		}
		diff := wantActive - active
		if diff < 0 {
			utilruntime.HandleError(fmt.Errorf("More active than wanted: job %q, want %d, have %d", jobKey, wantActive, active))
			diff = 0
		}
		jm.expectations.ExpectCreations(jobKey, int(diff))
		glog.V(4).Infof("Too few pods running job %q, need %d, creating %d", jobKey, wantActive, diff)

		active += diff
		wait := sync.WaitGroup{}
		wait.Add(int(diff))
		for i := int32(0); i < diff; i++ {
			go func() {
				defer wait.Done()
				if err := jm.podControl.CreatePods(job.Namespace, &job.Spec.Template, job); err != nil {
					defer utilruntime.HandleError(err)
					// Decrement the expected number of creates because the informer won't observe this pod
					jm.expectations.CreationObserved(jobKey)
					activeLock.Lock()
					active--
					activeLock.Unlock()
				}
			}()
		}
		wait.Wait()
	}

	return active
}

func (jm *JobController) updateJobStatus(job *batch.Job) error {
	_, err := jm.kubeClient.Batch().Jobs(job.Namespace).UpdateStatus(job)
	return err
}

// filterPods returns pods based on their phase.
func filterPods(pods []*v1.Pod, phase v1.PodPhase) int {
	result := 0
	for i := range pods {
		if phase == pods[i].Status.Phase {
			result++
		}
	}
	return result
}

// byCreationTimestamp sorts a list by creation timestamp, using their names as a tie breaker.
type byCreationTimestamp []batch.Job

func (o byCreationTimestamp) Len() int      { return len(o) }
func (o byCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}
