/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package replication

import (
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/workqueue"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

const (
	// We'll attempt to recompute the required replicas of all replication controllers
	// the have fulfilled their expectations at least this often. This recomputation
	// happens based on contents in local pod storage.
	FullControllerResyncPeriod = 30 * time.Second

	// If a watch misdelivers info about a pod, it'll take at least this long
	// to rectify the number of replicas. Note that dropped deletes are only
	// rectified after the expectation times out because we don't know the
	// final resting state of the pod.
	PodRelistPeriod = 5 * time.Minute

	// Realistic value of the burstReplica field for the replication manager based off
	// performance requirements for kubernetes 1.0.
	BurstReplicas = 500

	// We must avoid counting pods until the pod store has synced. If it hasn't synced, to
	// avoid a hot loop, we'll wait this long between checks.
	PodStoreSyncedPollPeriod = 100 * time.Millisecond

	// The number of times we retry updating a replication controller's status.
	statusUpdateRetries = 1
)

// ReplicationManager is responsible for synchronizing ReplicationController objects stored
// in the system with actual running pods.
type ReplicationManager struct {
	kubeClient client.Interface
	podControl controller.PodControlInterface

	// An rc is temporarily suspended after creating/deleting these many replicas.
	// It resumes normal action after observing the watch events for them.
	burstReplicas int
	// To allow injection of syncReplicationController for testing.
	syncHandler func(rcKey string) error

	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced func() bool

	// A TTLCache of pod creates/deletes each rc expects to see
	expectations controller.ControllerExpectationsInterface

	// A store of replication controllers, populated by the rcController
	rcStore cache.StoreToReplicationControllerLister

	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister
	// Watches changes to all replication controllers
	rcController *framework.Controller
	// Watches changes to all pods
	podController *framework.Controller
	// Controllers that need to be updated
	queue *workqueue.Type
}

// NewReplicationManager creates a new ReplicationManager.
func NewReplicationManager(kubeClient client.Interface, burstReplicas int) *ReplicationManager {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	rm := &ReplicationManager{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "replication-controller"}),
		},
		burstReplicas: burstReplicas,
		expectations:  controller.NewControllerExpectations(),
		queue:         workqueue.New(),
	}

	rm.rcStore.Store, rm.rcController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return rm.kubeClient.ReplicationControllers(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return rm.kubeClient.ReplicationControllers(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.ReplicationController{},
		FullControllerResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: rm.enqueueController,
			UpdateFunc: func(old, cur interface{}) {
				// We only really need to do this when spec changes, but for correctness it is safer to
				// periodically double check. It is overkill for 2 reasons:
				// 1. Status.Replica updates will cause a sync
				// 2. Every 30s we will get a full resync (this will happen anyway every 5 minutes when pods relist)
				// However, it shouldn't be that bad as rcs that haven't met expectations won't sync, and all
				// the listing is done using local stores.
				oldRC := old.(*api.ReplicationController)
				curRC := cur.(*api.ReplicationController)
				if oldRC.Status.Replicas != curRC.Status.Replicas {
					glog.V(4).Infof("Observed updated replica count for rc: %v, %d->%d", curRC.Name, oldRC.Status.Replicas, curRC.Status.Replicas)
				}
				rm.enqueueController(cur)
			},
			// This will enter the sync loop and no-op, becuase the controller has been deleted from the store.
			// Note that deleting a controller immediately after scaling it to 0 will not work. The recommended
			// way of achieving this is by performing a `stop` operation on the controller.
			DeleteFunc: rm.enqueueController,
		},
	)

	rm.podStore.Store, rm.podController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return rm.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return rm.kubeClient.Pods(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Pod{},
		PodRelistPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: rm.addPod,
			// This invokes the rc for every pod change, eg: host assignment. Though this might seem like overkill
			// the most frequent pod update is status, and the associated rc will only list from local storage, so
			// it should be ok.
			UpdateFunc: rm.updatePod,
			DeleteFunc: rm.deletePod,
		},
	)

	rm.syncHandler = rm.syncReplicationController
	rm.podStoreSynced = rm.podController.HasSynced
	return rm
}

// SetEventRecorder replaces the event recorder used by the replication manager
// with the given recorder. Only used for testing.
func (rm *ReplicationManager) SetEventRecorder(recorder record.EventRecorder) {
	// TODO: Hack. We can't cleanly shutdown the event recorder, so benchmarks
	// need to pass in a fake.
	rm.podControl = controller.RealPodControl{rm.kubeClient, recorder}
}

// Run begins watching and syncing.
func (rm *ReplicationManager) Run(workers int, stopCh <-chan struct{}) {
	defer util.HandleCrash()
	go rm.rcController.Run(stopCh)
	go rm.podController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(rm.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down RC Manager")
	rm.queue.ShutDown()
}

// getPodController returns the controller managing the given pod.
// TODO: Surface that we are ignoring multiple controllers for a single pod.
func (rm *ReplicationManager) getPodController(pod *api.Pod) *api.ReplicationController {
	controllers, err := rm.rcStore.GetPodControllers(pod)
	if err != nil {
		glog.V(4).Infof("No controllers found for pod %v, replication manager will avoid syncing", pod.Name)
		return nil
	}
	// In theory, overlapping controllers is user error. This sorting will not prevent
	// osciallation of replicas in all cases, eg:
	// rc1 (older rc): [(k1:v1)], replicas=1 rc2: [(k2:v2), (k1:v1)], replicas=2
	// pod: [(k1:v1)] will wake both rc1 and rc2, and we will sync rc1.
	// pod: [(k2:v2), (k1:v1)] will wake rc2 which creates a new replica.
	sort.Sort(overlappingControllers(controllers))
	return &controllers[0]
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (rm *ReplicationManager) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	if rc := rm.getPodController(pod); rc != nil {
		rcKey, err := controller.KeyFunc(rc)
		if err != nil {
			glog.Errorf("Couldn't get key for replication controller %#v: %v", rc, err)
			return
		}
		rm.expectations.CreationObserved(rcKey)
		rm.enqueueController(rc)
	}
}

// When a pod is updated, figure out what controller/s manage it and wake them
// up. If the labels of the pod have changed we need to awaken both the old
// and new controller. old and cur must be *api.Pod types.
func (rm *ReplicationManager) updatePod(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		// A periodic relist will send update events for all known pods.
		return
	}
	// TODO: Write a unittest for this case
	curPod := cur.(*api.Pod)
	if rc := rm.getPodController(curPod); rc != nil {
		rm.enqueueController(rc)
	}
	oldPod := old.(*api.Pod)
	// Only need to get the old controller if the labels changed.
	if !reflect.DeepEqual(curPod.Labels, oldPod.Labels) {
		// If the old and new rc are the same, the first one that syncs
		// will set expectations preventing any damage from the second.
		if oldRC := rm.getPodController(oldPod); oldRC != nil {
			rm.enqueueController(oldRC)
		}
	}
}

// When a pod is deleted, enqueue the controller that manages the pod and update its expectations.
// obj could be an *api.Pod, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) deletePod(obj interface{}) {
	pod, ok := obj.(*api.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new rc will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v, could take up to %v before a controller recreates a replica", obj, controller.ExpectationsTimeout)
			return
		}
		pod, ok = tombstone.Obj.(*api.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %+v, could take up to %v before controller recreates a replica", obj, controller.ExpectationsTimeout)
			return
		}
	}
	if rc := rm.getPodController(pod); rc != nil {
		rcKey, err := controller.KeyFunc(rc)
		if err != nil {
			glog.Errorf("Couldn't get key for replication controller %#v: %v", rc, err)
			return
		}
		rm.expectations.DeletionObserved(rcKey)
		rm.enqueueController(rc)
	}
}

// obj could be an *api.ReplicationController, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) enqueueController(obj interface{}) {
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
	rm.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (rm *ReplicationManager) worker() {
	for {
		func() {
			key, quit := rm.queue.Get()
			if quit {
				return
			}
			defer rm.queue.Done(key)
			err := rm.syncHandler(key.(string))
			if err != nil {
				glog.Errorf("Error syncing replication controller: %v", err)
			}
		}()
	}
}

// manageReplicas checks and updates replicas for the given replication controller.
func (rm *ReplicationManager) manageReplicas(filteredPods []*api.Pod, rc *api.ReplicationController) {
	diff := len(filteredPods) - rc.Spec.Replicas
	rcKey, err := controller.KeyFunc(rc)
	if err != nil {
		glog.Errorf("Couldn't get key for replication controller %#v: %v", rc, err)
		return
	}
	if diff < 0 {
		diff *= -1
		if diff > rm.burstReplicas {
			diff = rm.burstReplicas
		}
		rm.expectations.ExpectCreations(rcKey, diff)
		wait := sync.WaitGroup{}
		wait.Add(diff)
		glog.V(2).Infof("Too few %q/%q replicas, need %d, creating %d", rc.Namespace, rc.Name, rc.Spec.Replicas, diff)
		for i := 0; i < diff; i++ {
			go func() {
				defer wait.Done()
				if err := rm.podControl.CreateReplica(rc.Namespace, rc); err != nil {
					// Decrement the expected number of creates because the informer won't observe this pod
					glog.V(2).Infof("Failed creation, decrementing expectations for controller %q/%q", rc.Namespace, rc.Name)
					rm.expectations.CreationObserved(rcKey)
					util.HandleError(err)
				}
			}()
		}
		wait.Wait()
	} else if diff > 0 {
		if diff > rm.burstReplicas {
			diff = rm.burstReplicas
		}
		rm.expectations.ExpectDeletions(rcKey, diff)
		glog.V(2).Infof("Too many %q/%q replicas, need %d, deleting %d", rc.Namespace, rc.Name, rc.Spec.Replicas, diff)
		// No need to sort pods if we are about to delete all of them
		if rc.Spec.Replicas != 0 {
			// Sort the pods in the order such that not-ready < ready, unscheduled
			// < scheduled, and pending < running. This ensures that we delete pods
			// in the earlier stages whenever possible.
			sort.Sort(controller.ActivePods(filteredPods))
		}

		wait := sync.WaitGroup{}
		wait.Add(diff)
		for i := 0; i < diff; i++ {
			go func(ix int) {
				defer wait.Done()
				if err := rm.podControl.DeletePod(rc.Namespace, filteredPods[ix].Name); err != nil {
					// Decrement the expected number of deletes because the informer won't observe this deletion
					glog.V(2).Infof("Failed deletion, decrementing expectations for controller %q/%q", rc.Namespace, rc.Name)
					rm.expectations.DeletionObserved(rcKey)
				}
			}(i)
		}
		wait.Wait()
	}
}

// syncReplicationController will sync the rc with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (rm *ReplicationManager) syncReplicationController(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing controller %q (%v)", key, time.Now().Sub(startTime))
	}()

	obj, exists, err := rm.rcStore.Store.GetByKey(key)
	if !exists {
		glog.Infof("Replication Controller has been deleted %v", key)
		rm.expectations.DeleteExpectations(key)
		return nil
	}
	if err != nil {
		glog.Infof("Unable to retrieve rc %v from store: %v", key, err)
		rm.queue.Add(key)
		return err
	}
	rc := *obj.(*api.ReplicationController)
	if !rm.podStoreSynced() {
		// Sleep so we give the pod reflector goroutine a chance to run.
		time.Sleep(PodStoreSyncedPollPeriod)
		glog.Infof("Waiting for pods controller to sync, requeuing rc %v", rc.Name)
		rm.enqueueController(&rc)
		return nil
	}

	// Check the expectations of the rc before counting active pods, otherwise a new pod can sneak in
	// and update the expectations after we've retrieved active pods from the store. If a new pod enters
	// the store after we've checked the expectation, the rc sync is just deferred till the next relist.
	rcKey, err := controller.KeyFunc(&rc)
	if err != nil {
		glog.Errorf("Couldn't get key for replication controller %#v: %v", rc, err)
		return err
	}
	rcNeedsSync := rm.expectations.SatisfiedExpectations(rcKey)
	podList, err := rm.podStore.Pods(rc.Namespace).List(labels.Set(rc.Spec.Selector).AsSelector())
	if err != nil {
		glog.Errorf("Error getting pods for rc %q: %v", key, err)
		rm.queue.Add(key)
		return err
	}

	// TODO: Do this in a single pass, or use an index.
	filteredPods := controller.FilterActivePods(podList.Items)
	if rcNeedsSync {
		rm.manageReplicas(filteredPods, &rc)
	}

	// Always updates status as pods come up or die.
	if err := updateReplicaCount(rm.kubeClient.ReplicationControllers(rc.Namespace), rc, len(filteredPods)); err != nil {
		// Multiple things could lead to this update failing. Requeuing the controller ensures
		// we retry with some fairness.
		glog.V(2).Infof("Failed to update replica count for controller %v, requeuing", rc.Name)
		rm.enqueueController(&rc)
	}
	return nil
}
