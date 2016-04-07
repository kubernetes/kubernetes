/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

type JobController struct {
	kubeClient clientset.Interface
	podControl controller.PodControlInterface

	// internalPodInformer is used to hold a personal informer.  If we're using
	// a normal shared informer, then the informer will be started for us.  If
	// we have a personal informer, we must start it ourselves.   If you start
	// the controller using NewJobController(passing SharedInformer), this
	// will be null
	internalPodInformer framework.SharedInformer

	// To allow injection of updateJobStatus for testing.
	updateHandler func(job *batch.Job) error
	syncHandler   func(jobKey string) error
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced func() bool

	// A TTLCache of pod creates/deletes each rc expects to see
	expectations controller.ControllerExpectationsInterface

	// A store of job, populated by the jobController
	jobStore cache.StoreToJobLister
	// Watches changes to all jobs
	jobController *framework.Controller

	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister

	// Jobs that need to be updated
	queue *workqueue.Type

	recorder record.EventRecorder
}

func NewJobController(podInformer framework.SharedIndexInformer, kubeClient clientset.Interface) *JobController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})

	if kubeClient != nil && kubeClient.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("job_controller", kubeClient.Core().GetRESTClient().GetRateLimiter())
	}

	jm := &JobController{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "job-controller"}),
		},
		expectations: controller.NewControllerExpectations(),
		queue:        workqueue.New(),
		recorder:     eventBroadcaster.NewRecorder(api.EventSource{Component: "job-controller"}),
	}

	jm.jobStore.Store, jm.jobController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return jm.kubeClient.Batch().Jobs(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return jm.kubeClient.Batch().Jobs(api.NamespaceAll).Watch(options)
			},
		},
		&batch.Job{},
		// TODO: Can we have much longer period here?
		replicationcontroller.FullControllerResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: jm.enqueueController,
			UpdateFunc: func(old, cur interface{}) {
				if job := cur.(*batch.Job); !isJobFinished(job) {
					jm.enqueueController(job)
				}
			},
			DeleteFunc: jm.enqueueController,
		},
	)

	podInformer.AddEventHandler(framework.ResourceEventHandlerFuncs{
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

func NewJobControllerFromClient(kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc) *JobController {
	podInformer := informers.CreateSharedIndexPodInformer(kubeClient, resyncPeriod(), cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	jm := NewJobController(podInformer, kubeClient)
	jm.internalPodInformer = podInformer

	return jm
}

// Run the main goroutine responsible for watching and syncing jobs.
func (jm *JobController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	go jm.jobController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(jm.worker, time.Second, stopCh)
	}

	if jm.internalPodInformer != nil {
		go jm.internalPodInformer.Run(stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down Job Manager")
	jm.queue.ShutDown()
}

// getPodJob returns the job managing the given pod.
func (jm *JobController) getPodJob(pod *api.Pod) *batch.Job {
	jobs, err := jm.jobStore.GetPodJobs(pod)
	if err != nil {
		glog.V(4).Infof("No jobs found for pod %v, job controller will avoid syncing", pod.Name)
		return nil
	}
	if len(jobs) > 1 {
		glog.Errorf("user error! more than one job is selecting pods with labels: %+v", pod.Labels)
		sort.Sort(byCreationTimestamp(jobs))
	}
	return &jobs[0]
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (jm *JobController) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		jm.deletePod(pod)
		return
	}
	if job := jm.getPodJob(pod); job != nil {
		jobKey, err := controller.KeyFunc(job)
		if err != nil {
			glog.Errorf("Couldn't get key for job %#v: %v", job, err)
			return
		}
		jm.expectations.CreationObserved(jobKey)
		jm.enqueueController(job)
	}
}

// When a pod is updated, figure out what job/s manage it and wake them up.
// If the labels of the pod have changed we need to awaken both the old
// and new job. old and cur must be *api.Pod types.
func (jm *JobController) updatePod(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		// A periodic relist will send update events for all known pods.
		return
	}
	curPod := cur.(*api.Pod)
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
	oldPod := old.(*api.Pod)
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
// obj could be an *api.Pod, or a DeletionFinalStateUnknown marker item.
func (jm *JobController) deletePod(obj interface{}) {
	pod, ok := obj.(*api.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new job will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*api.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %+v", obj)
			return
		}
	}
	if job := jm.getPodJob(pod); job != nil {
		jobKey, err := controller.KeyFunc(job)
		if err != nil {
			glog.Errorf("Couldn't get key for job %#v: %v", job, err)
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
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
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
	for {
		func() {
			key, quit := jm.queue.Get()
			if quit {
				return
			}
			defer jm.queue.Done(key)
			err := jm.syncHandler(key.(string))
			if err != nil {
				glog.Errorf("Error syncing job: %v", err)
			}
		}()
	}
}

// syncJob will sync the job with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (jm *JobController) syncJob(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing job %q (%v)", key, time.Now().Sub(startTime))
	}()

	if !jm.podStoreSynced() {
		// Sleep so we give the pod reflector goroutine a chance to run.
		time.Sleep(replicationcontroller.PodStoreSyncedPollPeriod)
		glog.V(4).Infof("Waiting for pods controller to sync, requeuing job %v", key)
		jm.queue.Add(key)
		return nil
	}

	obj, exists, err := jm.jobStore.Store.GetByKey(key)
	if !exists {
		glog.V(4).Infof("Job has been deleted: %v", key)
		jm.expectations.DeleteExpectations(key)
		return nil
	}
	if err != nil {
		glog.Errorf("Unable to retrieve job %v from store: %v", key, err)
		jm.queue.Add(key)
		return err
	}
	job := *obj.(*batch.Job)

	// Check the expectations of the job before counting active pods, otherwise a new pod can sneak in
	// and update the expectations after we've retrieved active pods from the store. If a new pod enters
	// the store after we've checked the expectation, the job sync is just deferred till the next relist.
	jobKey, err := controller.KeyFunc(&job)
	if err != nil {
		glog.Errorf("Couldn't get key for job %#v: %v", job, err)
		return err
	}
	jobNeedsSync := jm.expectations.SatisfiedExpectations(jobKey)
	selector, _ := unversioned.LabelSelectorAsSelector(job.Spec.Selector)
	podList, err := jm.podStore.Pods(job.Namespace).List(selector)
	if err != nil {
		glog.Errorf("Error getting pods for job %q: %v", key, err)
		jm.queue.Add(key)
		return err
	}

	activePods := controller.FilterActivePods(podList.Items)
	active := int32(len(activePods))
	succeeded, failed := getStatus(podList.Items)
	conditions := len(job.Status.Conditions)
	if job.Status.StartTime == nil {
		now := unversioned.Now()
		job.Status.StartTime = &now
	}
	// if job was finished previously, we don't want to redo the termination
	if isJobFinished(&job) {
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
		jm.recorder.Event(&job, api.EventTypeNormal, "DeadlineExceeded", "Job was active longer than specified deadline")
	} else {
		if jobNeedsSync {
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
					jm.recorder.Event(&job, api.EventTypeWarning, "TooManyActivePods", "Too many active pods running after completion count reached")
				}
				if completions > *job.Spec.Completions {
					jm.recorder.Event(&job, api.EventTypeWarning, "TooManySucceededPods", "Too many succeeded pods running after completion count reached")
				}
			}
		}
		if complete {
			job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobComplete, "", ""))
			now := unversioned.Now()
			job.Status.CompletionTime = &now
		}
	}

	// no need to update the job if the status hasn't changed since last time
	if job.Status.Active != active || job.Status.Succeeded != succeeded || job.Status.Failed != failed || len(job.Status.Conditions) != conditions {
		job.Status.Active = active
		job.Status.Succeeded = succeeded
		job.Status.Failed = failed

		if err := jm.updateHandler(&job); err != nil {
			glog.Errorf("Failed to update job %v, requeuing.  Error: %v", job.Name, err)
			jm.enqueueController(&job)
		}
	}
	return nil
}

// pastActiveDeadline checks if job has ActiveDeadlineSeconds field set and if it is exceeded.
func pastActiveDeadline(job *batch.Job) bool {
	if job.Spec.ActiveDeadlineSeconds == nil || job.Status.StartTime == nil {
		return false
	}
	now := unversioned.Now()
	start := job.Status.StartTime.Time
	duration := now.Time.Sub(start)
	allowedDuration := time.Duration(*job.Spec.ActiveDeadlineSeconds) * time.Second
	return duration >= allowedDuration
}

func newCondition(conditionType batch.JobConditionType, reason, message string) batch.JobCondition {
	return batch.JobCondition{
		Type:               conditionType,
		Status:             api.ConditionTrue,
		LastProbeTime:      unversioned.Now(),
		LastTransitionTime: unversioned.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// getStatus returns no of succeeded and failed pods running a job
func getStatus(pods []api.Pod) (succeeded, failed int32) {
	succeeded = int32(filterPods(pods, api.PodSucceeded))
	failed = int32(filterPods(pods, api.PodFailed))
	return
}

// manageJob is the core method responsible for managing the number of running
// pods according to what is specified in the job.Spec.
func (jm *JobController) manageJob(activePods []*api.Pod, succeeded int32, job *batch.Job) int32 {
	var activeLock sync.Mutex
	active := int32(len(activePods))
	parallelism := *job.Spec.Parallelism
	jobKey, err := controller.KeyFunc(job)
	if err != nil {
		glog.Errorf("Couldn't get key for job %#v: %v", job, err)
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
			glog.Errorf("More active than wanted: job %q, want %d, have %d", jobKey, wantActive, active)
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
func filterPods(pods []api.Pod, phase api.PodPhase) int {
	result := 0
	for i := range pods {
		if phase == pods[i].Status.Phase {
			result++
		}
	}
	return result
}

func isJobFinished(j *batch.Job) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == api.ConditionTrue {
			return true
		}
	}
	return false
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
