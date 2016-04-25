/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
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
	// period to relist petsets and verify pets
	petSetResyncPeriod = 30 * time.Second
)

// PetSetController controls petsets.
type PetSetController struct {
	kubeClient *client.Client

	// newSyncer returns an interface capable of syncing a single pet.
	// Abstracted out for testing.
	newSyncer func(*pcb) *petSyncer

	// podStore is a cache of watched pods.
	podStore cache.StoreToPodLister

	// podStoreSynced returns true if the pod store has synced at least once.
	podStoreSynced func() bool
	// Watches changes to all pods.
	podController framework.ControllerInterface

	// A store of PetSets, populated by the psController.
	psStore cache.StoreToPetSetLister
	// Watches changes to all PetSets.
	psController *framework.Controller

	// A store of the 1 unhealthy pet blocking progress for a given ps
	blockingPetStore *unhealthyPetTracker

	// Controllers that need to be synced.
	queue *workqueue.Type

	// syncHandler handles sync events for petsets.
	// Abstracted as a func to allow injection for testing.
	syncHandler func(psKey string) []error
}

// NewPetSetController creates a new petset controller.
func NewPetSetController(podInformer framework.SharedInformer, kubeClient *client.Client, resyncPeriod time.Duration) *PetSetController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "petset"})
	pc := &apiServerPetClient{kubeClient, recorder, &defaultPetHealthChecker{}}

	psc := &PetSetController{
		kubeClient:       kubeClient,
		blockingPetStore: newUnHealthyPetTracker(pc),
		newSyncer: func(blockingPet *pcb) *petSyncer {
			return &petSyncer{pc, blockingPet}
		},
		queue: workqueue.New(),
	}

	podInformer.AddEventHandler(framework.ResourceEventHandlerFuncs{
		// lookup the petset and enqueue
		AddFunc: psc.addPod,
		// lookup current and old petset if labels changed
		UpdateFunc: psc.updatePod,
		// lookup petset accounting for deletion tombstones
		DeleteFunc: psc.deletePod,
	})
	psc.podStore.Store = podInformer.GetStore()
	psc.podController = podInformer.GetController()

	psc.psStore.Store, psc.psController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return psc.kubeClient.Apps().PetSets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return psc.kubeClient.Apps().PetSets(api.NamespaceAll).Watch(options)
			},
		},
		&apps.PetSet{},
		petSetResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: psc.enqueuePetSet,
			UpdateFunc: func(old, cur interface{}) {
				oldPS := old.(*apps.PetSet)
				curPS := cur.(*apps.PetSet)
				if oldPS.Status.Replicas != curPS.Status.Replicas {
					glog.V(4).Infof("Observed updated replica count for PetSet: %v, %d->%d", curPS.Name, oldPS.Status.Replicas, curPS.Status.Replicas)
				}
				psc.enqueuePetSet(cur)
			},
			DeleteFunc: psc.enqueuePetSet,
		},
	)
	// TODO: Watch volumes
	psc.podStoreSynced = psc.podController.HasSynced
	psc.syncHandler = psc.Sync
	return psc
}

// Run runs the petset controller.
func (psc *PetSetController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	glog.Infof("Starting petset controller")
	go psc.podController.Run(stopCh)
	go psc.psController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(psc.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down petset controller")
	psc.queue.ShutDown()
}

// addPod adds the petset for the pod to the sync queue
func (psc *PetSetController) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	glog.V(4).Infof("Pod %s created, labels: %+v", pod.Name, pod.Labels)
	ps := psc.getPetSetForPod(pod)
	if ps == nil {
		return
	}
	psc.enqueuePetSet(ps)
}

// updatePod adds the petset for the current and old pods to the sync queue.
// If the labels of the pod didn't change, this method enqueues a single petset.
func (psc *PetSetController) updatePod(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		return
	}
	curPod := cur.(*api.Pod)
	oldPod := old.(*api.Pod)
	ps := psc.getPetSetForPod(curPod)
	if ps == nil {
		return
	}
	psc.enqueuePetSet(ps)
	if !reflect.DeepEqual(curPod.Labels, oldPod.Labels) {
		if oldPS := psc.getPetSetForPod(oldPod); oldPS != nil {
			psc.enqueuePetSet(oldPS)
		}
	}
}

// deletePod enqueues the petset for the pod accounting for deletion tombstones.
func (psc *PetSetController) deletePod(obj interface{}) {
	pod, ok := obj.(*api.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new PetSet will not be woken up till the periodic resync.
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
	if ps := psc.getPetSetForPod(pod); ps != nil {
		psc.enqueuePetSet(ps)
	}
}

// getPodsForPetSets returns the pods that match the selectors of the given petset.
func (psc *PetSetController) getPodsForPetSet(ps *apps.PetSet) ([]*api.Pod, error) {
	// TODO: Do we want the petset to fight with RCs? check parent petset annoation, or name prefix?
	sel, err := unversioned.LabelSelectorAsSelector(ps.Spec.Selector)
	if err != nil {
		return []*api.Pod{}, err
	}
	petList, err := psc.podStore.Pods(ps.Namespace).List(sel)
	if err != nil {
		return []*api.Pod{}, err
	}
	pods := []*api.Pod{}
	for _, p := range petList.Items {
		pods = append(pods, &p)
	}
	return pods, nil
}

// getPetSetForPod returns the pet set managing the given pod.
func (psc *PetSetController) getPetSetForPod(pod *api.Pod) *apps.PetSet {
	ps, err := psc.psStore.GetPodPetSets(pod)
	if err != nil {
		glog.V(4).Infof("No PetSets found for pod %v, PetSet controller will avoid syncing", pod.Name)
		return nil
	}
	// Resolve a overlapping petset tie by creation timestamp.
	// Let's hope users don't create overlapping petsets.
	if len(ps) > 1 {
		glog.Errorf("user error! more than one PetSet is selecting pods with labels: %+v", pod.Labels)
		sort.Sort(overlappingPetSets(ps))
	}
	return &ps[0]
}

// enqueuePetSet enqueues the given petset in the work queue.
func (psc *PetSetController) enqueuePetSet(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Cound't get key for object %+v: %v", obj, err)
		return
	}
	psc.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (psc *PetSetController) worker() {
	for {
		func() {
			key, quit := psc.queue.Get()
			if quit {
				return
			}
			defer psc.queue.Done(key)
			if errs := psc.syncHandler(key.(string)); len(errs) != 0 {
				glog.Errorf("Error syncing PetSet %v, requeuing: %v", key.(string), errs)
				psc.queue.Add(key)
			}
		}()
	}
}

// Sync syncs the given petset.
func (psc *PetSetController) Sync(key string) []error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing pet set %q (%v)", key, time.Now().Sub(startTime))
	}()

	if !psc.podStoreSynced() {
		// Sleep so we give the pod reflector goroutine a chance to run.
		time.Sleep(PodStoreSyncedPollPeriod)
		return []error{fmt.Errorf("waiting for pods controller to sync")}
	}

	obj, exists, err := psc.psStore.Store.GetByKey(key)
	if !exists {
		if err = psc.blockingPetStore.store.Delete(key); err != nil {
			return []error{err}
		}
		glog.Infof("PetSet has been deleted %v", key)
		return []error{}
	}
	if err != nil {
		glog.Errorf("Unable to retrieve PetSet %v from store: %v", key, err)
		return []error{err}
	}

	ps := *obj.(*apps.PetSet)
	petList, err := psc.getPodsForPetSet(&ps)
	if err != nil {
		return []error{err}
	}

	numPets, errs := psc.syncPetSet(&ps, petList)
	if err := updatePetCount(psc.kubeClient, ps, numPets); err != nil {
		glog.Infof("Failed to update replica count for petset %v/%v; requeuing; error: %v", ps.Namespace, ps.Name, err)
		errs = append(errs, err)
	}

	return errs
}

// syncPetSet syncs a tuple of (petset, pets).
func (psc *PetSetController) syncPetSet(ps *apps.PetSet, pets []*api.Pod) (int, []error) {
	glog.Infof("Syncing PetSet %v/%v with %d pets", ps.Namespace, ps.Name, len(pets))

	it := NewPetSetIterator(ps, pets)
	blockingPet, err := psc.blockingPetStore.Get(ps, pets)
	if err != nil {
		return 0, []error{err}
	}
	if blockingPet != nil {
		glog.Infof("PetSet %v blocked from scaling on pet %v", ps.Name, blockingPet.pod.Name)
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
		if err != nil {
			it.errs = append(it.errs, err)
		}
	}

	if err := psc.blockingPetStore.Add(petManager.blockingPet); err != nil {
		it.errs = append(it.errs, err)
	}
	// TODO: GC pvcs. We can't delete them per pet because of grace period, and
	// in fact we *don't want to* till petset is stable to guarantee that bugs
	// in the controller don't corrupt user data.
	return numPets, it.errs
}
