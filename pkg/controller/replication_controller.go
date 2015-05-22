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

package controller

import (
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/workqueue"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

var (
	rcKeyFunc = framework.DeletionHandlingMetaNamespaceKeyFunc
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

	// If a watch drops a delete event for a pod, it'll take this long
	// before a dormant rc waiting for those packets is woken up anyway. It is
	// specifically targeted at the case where some problem prevents an update
	// of expectations, without it the RC could stay asleep forever. This should
	// be set based on the expected latency of watch events.
	//
	// Currently an rc can service (create *and* observe the watch events for said
	// creation) about 10-20 pods a second, so it takes about 1 min to service
	// 500 pods. Just creation is limited to 20qps, and watching happens with ~10-30s
	// latency/pod at the scale of 3000 pods over 100 nodes.
	ExpectationsTimeout = 3 * time.Minute

	// Realistic value of the burstReplica field for the replication manager based off
	// performance requirements for kubernetes 1.0.
	BurstReplicas = 500
)

// ReplicationManager is responsible for synchronizing ReplicationController objects stored
// in the system with actual running pods.
type ReplicationManager struct {
	kubeClient client.Interface
	podControl PodControlInterface

	// An rc is temporarily suspended after creating/deleting these many replicas.
	// It resumes normal action after observing the watch events for them.
	burstReplicas int
	// To allow injection of syncReplicationController for testing.
	syncHandler func(rcKey string) error
	// A TTLCache of pod creates/deletes each rc expects to see
	expectations RCExpectationsManager
	// A store of controllers, populated by the rcController
	controllerStore cache.StoreToControllerLister
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
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	rm := &ReplicationManager{
		kubeClient: kubeClient,
		podControl: RealPodControl{
			kubeClient: kubeClient,
			recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "replication-controller"}),
		},
		burstReplicas: burstReplicas,
		expectations:  NewRCExpectations(),
		queue:         workqueue.New(),
	}

	rm.controllerStore.Store, rm.rcController = framework.NewInformer(
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
			// Note that deleting a controller immediately after resizing it to 0 will not work. The recommended
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
	return rm
}

// SetEventRecorder replaces the event recorder used by the replication manager
// with the given recorder. Only used for testing.
func (rm *ReplicationManager) SetEventRecorder(recorder record.EventRecorder) {
	// TODO: Hack. We can't cleanly shutdown the event recorder, so benchmarks
	// need to pass in a fake.
	rm.podControl = RealPodControl{rm.kubeClient, recorder}
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

// getPodControllers returns the controller managing the given pod.
// TODO: Surface that we are ignoring multiple controllers for a single pod.
func (rm *ReplicationManager) getPodControllers(pod *api.Pod) *api.ReplicationController {
	controllers, err := rm.controllerStore.GetPodControllers(pod)
	if err != nil {
		glog.V(4).Infof("No controllers found for pod %v, replication manager will avoid syncing", pod.Name)
		return nil
	}
	return &controllers[0]
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (rm *ReplicationManager) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	if rc := rm.getPodControllers(pod); rc != nil {
		rm.expectations.CreationObserved(rc)
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
	if rc := rm.getPodControllers(curPod); rc != nil {
		rm.enqueueController(rc)
	}
	oldPod := old.(*api.Pod)
	// Only need to get the old controller if the labels changed.
	if !reflect.DeepEqual(curPod.Labels, oldPod.Labels) {
		// If the old and new rc are the same, the first one that syncs
		// will set expectations preventing any damage from the second.
		if oldRC := rm.getPodControllers(oldPod); oldRC != nil {
			rm.enqueueController(oldRC)
		}
	}
}

// When a pod is deleted, enqueue the controller that manages the pod and update its expectations.
// obj could be an *api.Pod, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) deletePod(obj interface{}) {
	if pod, ok := obj.(*api.Pod); ok {
		if rc := rm.getPodControllers(pod); rc != nil {
			rm.expectations.DeletionObserved(rc)
			rm.enqueueController(rc)
		}
		return
	}
	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone key. Since we don't
	// know which rc to wake up/update expectations, we rely on the ttl on the
	// expectation expiring. The rc syncs via the 30s periodic resync and notices
	// fewer pods than its replica count.
	podKey, err := framework.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	// A periodic relist might not have a pod that the store has, in such cases we are sent a tombstone key.
	// We don't know which controllers to sync, so just let the controller relist handle this.
	glog.Infof("Pod %q was deleted but we don't have a record of its final state so it could take up to %v before a controller recreates a replica.", podKey, ExpectationsTimeout)
}

// obj could be an *api.ReplicationController, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) enqueueController(obj interface{}) {
	key, err := rcKeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

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
func (rm *ReplicationManager) manageReplicas(filteredPods []*api.Pod, controller *api.ReplicationController) {
	diff := len(filteredPods) - controller.Spec.Replicas
	if diff < 0 {
		diff *= -1
		if diff > rm.burstReplicas {
			diff = rm.burstReplicas
		}
		rm.expectations.ExpectCreations(controller, diff)
		wait := sync.WaitGroup{}
		wait.Add(diff)
		glog.V(2).Infof("Too few %q/%q replicas, need %d, creating %d", controller.Namespace, controller.Name, controller.Spec.Replicas, diff)
		for i := 0; i < diff; i++ {
			go func() {
				defer wait.Done()
				if err := rm.podControl.createReplica(controller.Namespace, controller); err != nil {
					// Decrement the expected number of creates because the informer won't observe this pod
					glog.V(2).Infof("Failed creation, decrementing expectations for controller %q/%q", controller.Namespace, controller.Name)
					rm.expectations.CreationObserved(controller)
					util.HandleError(err)
				}
			}()
		}
		wait.Wait()
	} else if diff > 0 {
		if diff > rm.burstReplicas {
			diff = rm.burstReplicas
		}
		rm.expectations.ExpectDeletions(controller, diff)
		glog.V(2).Infof("Too many %q/%q replicas, need %d, deleting %d", controller.Namespace, controller.Name, controller.Spec.Replicas, diff)
		// No need to sort pods if we are about to delete all of them
		if controller.Spec.Replicas != 0 {
			// Sort the pods in the order such that not-ready < ready, unscheduled
			// < scheduled, and pending < running. This ensures that we delete pods
			// in the earlier stages whenever possible.
			sort.Sort(activePods(filteredPods))
		}

		wait := sync.WaitGroup{}
		wait.Add(diff)
		for i := 0; i < diff; i++ {
			go func(ix int) {
				defer wait.Done()
				if err := rm.podControl.deletePod(controller.Namespace, filteredPods[ix].Name); err != nil {
					// Decrement the expected number of deletes because the informer won't observe this deletion
					glog.V(2).Infof("Failed deletion, decrementing expectations for controller %q/%q", controller.Namespace, controller.Name)
					rm.expectations.DeletionObserved(controller)
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

	obj, exists, err := rm.controllerStore.Store.GetByKey(key)
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
	controller := *obj.(*api.ReplicationController)

	// Check the expectations of the rc before counting active pods, otherwise a new pod can sneak in
	// and update the expectations after we've retrieved active pods from the store. If a new pod enters
	// the store after we've checked the expectation, the rc sync is just deferred till the next relist.
	rcNeedsSync := rm.expectations.SatisfiedExpectations(&controller)
	podList, err := rm.podStore.Pods(controller.Namespace).List(labels.Set(controller.Spec.Selector).AsSelector())
	if err != nil {
		glog.Errorf("Error getting pods for rc %q: %v", key, err)
		rm.queue.Add(key)
		return err
	}

	// TODO: Do this in a single pass, or use an index.
	filteredPods := filterActivePods(podList.Items)
	if rcNeedsSync {
		rm.manageReplicas(filteredPods, &controller)
	}

	// Always updates status as pods come up or die.
	if err := updateReplicaCount(rm.kubeClient.ReplicationControllers(controller.Namespace), controller, len(filteredPods)); err != nil {
		// Multiple things could lead to this update failing. Requeuing the controller ensures
		// we retry with some fairness.
		glog.V(2).Infof("Failed to update replica count for controller %v, requeuing", controller.Name)
		rm.enqueueController(&controller)
	}
	return nil
}
