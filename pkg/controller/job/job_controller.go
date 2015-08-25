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
	// "sort"
	// "sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/cache"
	"k8s.io/kubernetes/pkg/client/unversioned/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	// "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	// "k8s.io/kubernetes/pkg/watch"
)

type JobManager struct {
	kubeClient client.Interface
	podControl controller.PodControlInterface

	// To allow injection of syncJob for testing.
	syncHandler func(jobKey string) error
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
	// Watches changes to all pods
	podController *framework.Controller

	// Jobs that need to be updated
	queue *workqueue.Type
}

func NewJobManager(kubeClient client.Interface) *JobManager {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	jm := &JobManager{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "job"}),
		},
		expectations: controller.NewControllerExpectations(),
		queue:        workqueue.New(),
	}

	client := kubeClient.(*client.Client)
	jm.jobStore.Store, jm.jobController = framework.NewInformer(
		cache.NewListWatchFromClient(client, "jobs", api.NamespaceAll, fields.Everything()),
		&expapi.Job{},
		replicationcontroller.FullControllerResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: jm.enqueueController,
			// TODO: rethink if this is needed
			// UpdateFunc: jm.enqueueController,
			DeleteFunc: jm.enqueueController,
		},
	)

	jm.podStore.Store, jm.podController = framework.NewInformer(
		cache.NewListWatchFromClient(client, "pods", api.NamespaceAll, fields.Everything()),
		&api.Pod{},
		replicationcontroller.PodRelistPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    jm.addPod,
			UpdateFunc: jm.updatePod,
			DeleteFunc: jm.deletePod,
		},
	)

	jm.syncHandler = jm.syncJob
	return jm
}

// Run the main goroutine responsible for watching and syncing jobs.
func (jm *JobManager) Run(workers int, stopCh <-chan struct{}) {
	defer util.HandleCrash()
	go jm.jobController.Run(stopCh)
	go jm.podController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(jm.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down Job Manager")
	jm.queue.ShutDown()
}

// getPodJob returns the job managing the given pod.
func (jm *JobManager) getPodJob(pod *api.Pod) *expapi.Job {
	jobs, err := jm.jobStore.GetPodJobs(pod)
	if err != nil {
		glog.V(4).Infof("No jobs found for pod %v, job manager will avoid syncing", pod.Name)
		return nil
	}
	// TODO: add sorting and rethink the overlaping controllers, internally and with RCs
	return &jobs[0]
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (jm *JobManager) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller manager, it's possible a new pod shows up in a state that
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
func (jm *JobManager) updatePod(old, cur interface{}) {
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
	// TODO: rethink if it's needed
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
func (jm *JobManager) deletePod(obj interface{}) {
	pod, ok := obj.(*api.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new job will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v, could take up to %v before a job recreates a pod", obj, controller.ExpectationsTimeout)
			return
		}
		pod, ok = tombstone.Obj.(*api.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %+v, could take up to %v before job recreates a pod", obj, controller.ExpectationsTimeout)
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

// obj could be an *expapi.Job, or a DeletionFinalStateUnknown marker item.
func (jm *JobManager) enqueueController(obj interface{}) {
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
func (jm *JobManager) worker() {
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
func (jm *JobManager) syncJob(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing job %q (%v)", key, time.Now().Sub(startTime))
	}()

	obj, exists, err := jm.jobStore.Store.GetByKey(key)
	if !exists {
		glog.Infof("Job has been deleted %v", key)
		jm.expectations.DeleteExpectations(key)
		return nil
	}
	if err != nil {
		glog.Infof("Unable to retrieve job %v from store: %v", key, err)
		jm.queue.Add(key)
		return err
	}
	job := *obj.(*expapi.Job)
	if !jm.podStoreSynced() {
		// Sleep so we give the pod reflector goroutine a chance to run.
		time.Sleep(replicationcontroller.PodStoreSyncedPollPeriod)
		glog.Infof("Waiting for pods controller to sync, requeuing job %v", job.Name)
		jm.enqueueController(&job)
		return nil
	}

	// Check the expectations of the job before counting active pods, otherwise a new pod can sneak in
	// and update the expectations after we've retrieved active pods from the store. If a new pod enters
	// the store after we've checked the expectation, the job sync is just deferred till the next relist.
	jobKey, err := controller.KeyFunc(&job)
	if err != nil {
		glog.Errorf("Couldn't get key for job %#v: %v", job, err)
		return err
	}
	jobNeedsSync := jm.expectations.SatisfiedExpectations(jobKey)
	podList, err := jm.podStore.Pods(job.Namespace).List(labels.Set(job.Spec.Selector).AsSelector())
	if err != nil {
		glog.Errorf("Error getting pods for job %q: %v", key, err)
		jm.queue.Add(key)
		return err
	}

	filteredPods := controller.FilterActivePods(podList.Items)
	if jobNeedsSync {
		jm.manageJob(filteredPods, &job)
	}

	if err := jm.updateJob(&job, len(filteredPods)); err != nil {
		glog.V(2).Infof("Failed to update job %v, requeuing", job.Name)
		jm.enqueueController(&job)
	}
	return nil
}

func (jm *JobManager) manageJob(activePods []*api.Pod, job *expapi.Job) {
	// TODO
}

func (jm *JobManager) updateJob(job *expapi.Job, numPods int) error {
	// TODO
	return nil
}
