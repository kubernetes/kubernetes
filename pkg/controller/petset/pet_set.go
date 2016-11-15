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

package petset

import (
	"fmt"
	"reflect"
	"sort"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/record"

	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/errors"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	// Time to sleep before polling to see if the pod cache has synced.
	PodStoreSyncedPollPeriod = 100 * time.Millisecond
	// number of retries for a status update.
	statusUpdateRetries = 2
	// period to relist statefulsets and verify pets
	statefulSetResyncPeriod = 30 * time.Second
)

// StatefulSetController controls statefulsets.
type StatefulSetController struct {
	kubeClient internalclientset.Interface

	// newSyncer returns an interface capable of syncing a single pet.
	// Abstracted out for testing.
	newSyncer func(*pcb) *petSyncer

	// podStore is a cache of watched pods.
	podStore cache.StoreToPodLister

	// podStoreSynced returns true if the pod store has synced at least once.
	podStoreSynced func() bool
	// Watches changes to all pods.
	podController cache.ControllerInterface

	// A store of StatefulSets, populated by the psController.
	psStore cache.StoreToStatefulSetLister
	// Watches changes to all StatefulSets.
	psController *cache.Controller

	// A store of the 1 unhealthy pet blocking progress for a given ps
	blockingPetStore *unhealthyPetTracker

	// Controllers that need to be synced.
	queue workqueue.RateLimitingInterface

	// syncHandler handles sync events for statefulsets.
	// Abstracted as a func to allow injection for testing.
	syncHandler func(psKey string) error
}

// NewStatefulSetController creates a new statefulset controller.
func NewStatefulSetController(podInformer cache.SharedIndexInformer, kubeClient internalclientset.Interface, resyncPeriod time.Duration) *StatefulSetController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "statefulset"})
	pc := &apiServerPetClient{kubeClient, recorder, &defaultPetHealthChecker{}}

	psc := &StatefulSetController{
		kubeClient:       kubeClient,
		blockingPetStore: newUnHealthyPetTracker(pc),
		newSyncer: func(blockingPet *pcb) *petSyncer {
			return &petSyncer{pc, blockingPet}
		},
		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "statefulset"),
	}

	podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		// lookup the statefulset and enqueue
		AddFunc: psc.addPod,
		// lookup current and old statefulset if labels changed
		UpdateFunc: psc.updatePod,
		// lookup statefulset accounting for deletion tombstones
		DeleteFunc: psc.deletePod,
	})
	psc.podStore.Indexer = podInformer.GetIndexer()
	psc.podController = podInformer.GetController()

	psc.psStore.Store, psc.psController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return psc.kubeClient.Apps().StatefulSets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return psc.kubeClient.Apps().StatefulSets(api.NamespaceAll).Watch(options)
			},
		},
		&apps.StatefulSet{},
		statefulSetResyncPeriod,
		cache.ResourceEventHandlerFuncs{
			AddFunc: psc.enqueueStatefulSet,
			UpdateFunc: func(old, cur interface{}) {
				oldPS := old.(*apps.StatefulSet)
				curPS := cur.(*apps.StatefulSet)
				if oldPS.Status.Replicas != curPS.Status.Replicas {
					glog.V(4).Infof("Observed updated replica count for StatefulSet: %v, %d->%d", curPS.Name, oldPS.Status.Replicas, curPS.Status.Replicas)
				}
				psc.enqueueStatefulSet(cur)
			},
			DeleteFunc: psc.enqueueStatefulSet,
		},
	)
	// TODO: Watch volumes
	psc.podStoreSynced = psc.podController.HasSynced
	psc.syncHandler = psc.Sync
	return psc
}

// Run runs the statefulset controller.
func (psc *StatefulSetController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	glog.Infof("Starting statefulset controller")
	go psc.podController.Run(stopCh)
	go psc.psController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(psc.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down statefulset controller")
	psc.queue.ShutDown()
}

// addPod adds the statefulset for the pod to the sync queue
func (psc *StatefulSetController) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	glog.V(4).Infof("Pod %s created, labels: %+v", pod.Name, pod.Labels)
	ps := psc.getStatefulSetForPod(pod)
	if ps == nil {
		return
	}
	psc.enqueueStatefulSet(ps)
}

// updatePod adds the statefulset for the current and old pods to the sync queue.
// If the labels of the pod didn't change, this method enqueues a single statefulset.
func (psc *StatefulSetController) updatePod(old, cur interface{}) {
	curPod := cur.(*api.Pod)
	oldPod := old.(*api.Pod)
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs.
		return
	}
	ps := psc.getStatefulSetForPod(curPod)
	if ps == nil {
		return
	}
	psc.enqueueStatefulSet(ps)
	if !reflect.DeepEqual(curPod.Labels, oldPod.Labels) {
		if oldPS := psc.getStatefulSetForPod(oldPod); oldPS != nil {
			psc.enqueueStatefulSet(oldPS)
		}
	}
}

// deletePod enqueues the statefulset for the pod accounting for deletion tombstones.
func (psc *StatefulSetController) deletePod(obj interface{}) {
	pod, ok := obj.(*api.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new StatefulSet will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("couldn't get object from tombstone %+v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*api.Pod)
		if !ok {
			glog.Errorf("tombstone contained object that is not a pod %+v", obj)
			return
		}
	}
	glog.V(4).Infof("Pod %s/%s deleted through %v.", pod.Namespace, pod.Name, utilruntime.GetCaller())
	if ps := psc.getStatefulSetForPod(pod); ps != nil {
		psc.enqueueStatefulSet(ps)
	}
}

// getPodsForStatefulSets returns the pods that match the selectors of the given statefulset.
func (psc *StatefulSetController) getPodsForStatefulSet(ps *apps.StatefulSet) ([]*api.Pod, error) {
	// TODO: Do we want the statefulset to fight with RCs? check parent statefulset annoation, or name prefix?
	sel, err := unversioned.LabelSelectorAsSelector(ps.Spec.Selector)
	if err != nil {
		return []*api.Pod{}, err
	}
	pods, err := psc.podStore.Pods(ps.Namespace).List(sel)
	if err != nil {
		return []*api.Pod{}, err
	}
	// TODO: Do we need to copy?
	result := make([]*api.Pod, 0, len(pods))
	for i := range pods {
		result = append(result, &(*pods[i]))
	}
	return result, nil
}

// getStatefulSetForPod returns the pet set managing the given pod.
func (psc *StatefulSetController) getStatefulSetForPod(pod *api.Pod) *apps.StatefulSet {
	ps, err := psc.psStore.GetPodStatefulSets(pod)
	if err != nil {
		glog.V(4).Infof("No StatefulSets found for pod %v, StatefulSet controller will avoid syncing", pod.Name)
		return nil
	}
	// Resolve a overlapping statefulset tie by creation timestamp.
	// Let's hope users don't create overlapping statefulsets.
	if len(ps) > 1 {
		glog.Errorf("user error! more than one StatefulSet is selecting pods with labels: %+v", pod.Labels)
		sort.Sort(overlappingStatefulSets(ps))
	}
	return &ps[0]
}

// enqueueStatefulSet enqueues the given statefulset in the work queue.
func (psc *StatefulSetController) enqueueStatefulSet(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Cound't get key for object %+v: %v", obj, err)
		return
	}
	psc.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (psc *StatefulSetController) worker() {
	for {
		func() {
			key, quit := psc.queue.Get()
			if quit {
				return
			}
			defer psc.queue.Done(key)
			if err := psc.syncHandler(key.(string)); err != nil {
				glog.Errorf("Error syncing StatefulSet %v, requeuing: %v", key.(string), err)
				psc.queue.AddRateLimited(key)
			} else {
				psc.queue.Forget(key)
			}
		}()
	}
}

// Sync syncs the given statefulset.
func (psc *StatefulSetController) Sync(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing statefulset %q (%v)", key, time.Now().Sub(startTime))
	}()

	if !psc.podStoreSynced() {
		// Sleep so we give the pod reflector goroutine a chance to run.
		time.Sleep(PodStoreSyncedPollPeriod)
		return fmt.Errorf("waiting for pods controller to sync")
	}

	obj, exists, err := psc.psStore.Store.GetByKey(key)
	if !exists {
		if err = psc.blockingPetStore.store.Delete(key); err != nil {
			return err
		}
		glog.Infof("StatefulSet has been deleted %v", key)
		return nil
	}
	if err != nil {
		glog.Errorf("Unable to retrieve StatefulSet %v from store: %v", key, err)
		return err
	}

	ps := *obj.(*apps.StatefulSet)
	petList, err := psc.getPodsForStatefulSet(&ps)
	if err != nil {
		return err
	}

	numPets, syncErr := psc.syncStatefulSet(&ps, petList)
	if updateErr := updatePetCount(psc.kubeClient.Apps(), ps, numPets); updateErr != nil {
		glog.Infof("Failed to update replica count for statefulset %v/%v; requeuing; error: %v", ps.Namespace, ps.Name, updateErr)
		return errors.NewAggregate([]error{syncErr, updateErr})
	}

	return syncErr
}

// syncStatefulSet syncs a tuple of (statefulset, pets).
func (psc *StatefulSetController) syncStatefulSet(ps *apps.StatefulSet, pets []*api.Pod) (int, error) {
	glog.V(2).Infof("Syncing StatefulSet %v/%v with %d pods", ps.Namespace, ps.Name, len(pets))

	it := NewStatefulSetIterator(ps, pets)
	blockingPet, err := psc.blockingPetStore.Get(ps, pets)
	if err != nil {
		return 0, err
	}
	if blockingPet != nil {
		glog.Infof("StatefulSet %v blocked from scaling on pod %v", ps.Name, blockingPet.pod.Name)
	}
	petManager := psc.newSyncer(blockingPet)
	numPets := 0

	for it.Next() {
		pet := it.Value()
		if pet == nil {
			continue
		}
		switch pet.event {
		case syncPet:
			err = petManager.Sync(pet)
			if err == nil {
				numPets++
			}
		case deletePet:
			err = petManager.Delete(pet)
		}
		switch err.(type) {
		case errUnhealthyPet:
			// We are not passing this error up, but we don't increment numPets if we encounter it,
			// since numPets directly translates to statefulset.status.replicas
			continue
		case nil:
			continue
		default:
			it.errs = append(it.errs, err)
		}
	}

	if err := psc.blockingPetStore.Add(petManager.blockingPet); err != nil {
		it.errs = append(it.errs, err)
	}
	// TODO: GC pvcs. We can't delete them per pet because of grace period, and
	// in fact we *don't want to* till statefulset is stable to guarantee that bugs
	// in the controller don't corrupt user data.
	return numPets, errors.NewAggregate(it.errs)
}
