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

package statefulset

import (
	//"fmt"
	//"math/rand"
	//"reflect"
	//"testing"
	//"github.com/coreos/etcd/store"
	//"k8s.io/apimachinery/pkg/util/errors"
	//"k8s.io/kubernetes/pkg/api/v1"
	//apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	//fakeinternal "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	//"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/apps/v1beta1"
	//"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/apps/v1beta1/fake"
	"k8s.io/kubernetes/pkg/controller"
	//"k8s.io/kubernetes/pkg/volume/fc"
	//"sort"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"testing"
)

func newFakeStatefulSetController() (*StatefulSetController, *fakeStatefulPodControl) {
	fpc := newFakeStatefulPodControl()
	ssc := &StatefulSetController{
		kubeClient:     nil,
		podStoreSynced: func() bool { return true },
		setStore:       cache.StoreToStatefulSetLister{Store: cache.NewStore(controller.KeyFunc)},
		podStore:       cache.StoreToPodLister{Indexer: cache.NewIndexer(controller.KeyFunc, cache.Indexers{})},
		control:        NewDefaultStatefulSetControl(fpc),
		queue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "statefulset"),
	}
	return ssc, fpc
}

func TestStatefulSetControllerAddPod(t *testing.T) {
	ssc, _ := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	ssc.addPod(pod)
	item, _ := ssc.queue.Get()
	if key, _ := controller.KeyFunc(set); key != item.(string) {
		t.Errorf("dequeued bad key %s", key)
	}
}

//
//func scaleStatefulSet(t *testing.T, set *apps.StatefulSet, ssc *StatefulSetController, pc *fakeStatefulPodControl, scale int) {
//	for i := 0; i < scale; i++ {
//		pods := pc.Pods(set.Namespace)
//		if len(pods) != i {
//			t.Errorf("Unexpected number of pods, expected %d found %d", i, len(pods))
//		}
//		setOneRunning(set,pc)
//		if err := ssc.syncStatefulSet(set, pc.Pods(set.Namespace)); err != nil {
//			t.Errorf("syncStatefulSet error : %s", err)
//
//		}
//		assertStatefulSetProperties(set, pc, t)
//	}
//}
//
//func saturateStatefulSet(t *testing.T, set *apps.StatefulSet, ssc *StatefulSetController, pc *fakeStatefulPodControl) {
//	scaleStatefulSet(t, set, ssc, pc, int(*(set.Spec.Replicas)))
//}
//
//func TestStatefulSetControllerCreates(t *testing.T) {
//	ssc, pc := newFakeStatefulSetController()
//	replicas := 3
//	set := newStatefulSet(replicas)
//	saturateStatefulSet(t, set, ssc, pc)
//	assertStatefulSetProperties(set, pc, t)
//
//	// Deleted pet gets recreated
//	pods := pc.Pods(set.Namespace)
//	sort.Sort(ascendingOrdinal(pods))
//	target := pods[len(pods)-1]
//	pc.DeleteStatefulPod(set,target)
//	if err := ssc.syncStatefulSet(set, pc.Pods(set.Namespace)); err != nil {
//		t.Errorf("Error syncing StatefulSet: %v", err)
//	}
//	assertStatefulSetProperties(set, pc, t)
//	setOneRunning(set,pc)
//	if err := ssc.syncStatefulSet(set, pc.Pods(set.Namespace)); err != nil {
//		t.Errorf("Error syncing StatefulSet: %v", err)
//	}
//	assertStatefulSetProperties(set, pc, t)
//
//}
//
//func TestStatefulSetControllerDeletes(t *testing.T) {
//	psc, fc := newFakeStatefulSetController()
//	replicas := 4
//	ps := newStatefulSet(replicas)
//
//	saturateStatefulSet(t, ps, psc, fc)
//
//	// Drain
//	errs := []error{}
//	*(ps.Spec.Replicas) = 0
//	knownPods := fc.getPodList()
//	for i := replicas - 1; i >= 0; i-- {
//		if len(fc.pets) != i+1 {
//			t.Errorf("Unexpected number of pods, expected %d found %d", i+1, len(fc.pets))
//		}
//		if _, syncErr := psc.syncStatefulSet(ps, knownPods); syncErr != nil {
//			errs = append(errs, syncErr)
//		}
//	}
//	if len(errs) != 0 {
//		t.Errorf("Error syncing StatefulSet: %v", errors.NewAggregate(errs))
//	}
//	checkPods(ps, replicas, replicas, fc, t)
//}
//
//func TestStatefulSetControllerRespectsTermination(t *testing.T) {
//	psc, fc := newFakeStatefulSetController()
//	replicas := 4
//	ps := newStatefulSet(replicas)
//
//	saturateStatefulSet(t, ps, psc, fc)
//
//	fc.setDeletionTimestamp(replicas - 1)
//	*(ps.Spec.Replicas) = 2
//	_, err := psc.syncStatefulSet(ps, fc.getPodList())
//	if err != nil {
//		t.Errorf("Error syncing StatefulSet: %v", err)
//	}
//	// Finding a pod with the deletion timestamp will pause all deletions.
//	knownPods := fc.getPodList()
//	if len(knownPods) != 4 {
//		t.Errorf("Pods deleted prematurely before deletion timestamp expired, len %d", len(knownPods))
//	}
//	fc.pets = fc.pets[:replicas-1]
//	_, err = psc.syncStatefulSet(ps, fc.getPodList())
//	if err != nil {
//		t.Errorf("Error syncing StatefulSet: %v", err)
//	}
//	checkPods(ps, replicas, 1, fc, t)
//}
//
//func TestStatefulSetControllerRespectsOrder(t *testing.T) {
//	psc, fc := newFakeStatefulSetController()
//	replicas := 4
//	ps := newStatefulSet(replicas)
//
//	saturateStatefulSet(t, ps, psc, fc)
//
//	errs := []error{}
//	*(ps.Spec.Replicas) = 0
//	// Shuffle known list and check that pets are deleted in reverse
//	knownPods := fc.getPodList()
//	for i := range knownPods {
//		j := rand.Intn(i + 1)
//		knownPods[i], knownPods[j] = knownPods[j], knownPods[i]
//	}
//
//	for i := 0; i < replicas; i++ {
//		if len(fc.pets) != replicas-i {
//			t.Errorf("Unexpected number of pods, expected %d found %d", i, len(fc.pets))
//		}
//		if _, syncErr := psc.syncStatefulSet(ps, knownPods); syncErr != nil {
//			errs = append(errs, syncErr)
//		}
//		checkPods(ps, replicas, i+1, fc, t)
//	}
//	if len(errs) != 0 {
//		t.Errorf("Error syncing StatefulSet: %v", errors.NewAggregate(errs))
//	}
//}
//
//func TestStatefulSetControllerBlocksScaling(t *testing.T) {
//	psc, fc := newFakeStatefulSetController()
//	replicas := 5
//	ps := newStatefulSet(replicas)
//	scaleStatefulSet(t, ps, psc, fc, 3)
//
//	// Create 4th pet, then before flipping it to healthy, kill the first pet.
//	// There should only be 1 not-healty pet at a time.
//	pl := fc.getPodList()
//	if _, err := psc.syncStatefulSet(ps, pl); err != nil {
//		t.Errorf("Error syncing StatefulSet: %v", err)
//	}
//
//	deletedPod := pl[0]
//	fc.deletePetAtIndex(0)
//	pl = fc.getPodList()
//	if _, err := psc.syncStatefulSet(ps, pl); err != nil {
//		t.Errorf("Error syncing StatefulSet: %v", err)
//	}
//	newPodList := fc.getPodList()
//	for _, p := range newPodList {
//		if p.Name == deletedPod.Name {
//			t.Errorf("Deleted pod was created while existing pod was unhealthy")
//		}
//	}
//
//	fc.setHealthy(len(newPodList) - 1)
//	if _, err := psc.syncStatefulSet(ps, pl); err != nil {
//		t.Errorf("Error syncing StatefulSet: %v", err)
//	}
//
//	found := false
//	for _, p := range fc.getPodList() {
//		if p.Name == deletedPod.Name {
//			found = true
//			break
//		}
//	}
//	if !found {
//		t.Errorf("Deleted pod was not created after existing pods became healthy")
//	}
//}
//
//func TestStatefulSetBlockingPetIsCleared(t *testing.T) {
//	psc, fc := newFakeStatefulSetController()
//	ps := newStatefulSet(3)
//	scaleStatefulSet(t, ps, psc, fc, 1)
//
//	if blocking, err := psc.blockingPetStore.Get(ps, fc.getPodList()); err != nil || blocking != nil {
//		t.Errorf("Unexpected blocking pod %v, err %v", blocking, err)
//	}
//
//	// 1 not yet healthy pet
//	psc.syncStatefulSet(ps, fc.getPodList())
//
//	if blocking, err := psc.blockingPetStore.Get(ps, fc.getPodList()); err != nil || blocking == nil {
//		t.Errorf("Expected blocking pod %v, err %v", blocking, err)
//	}
//
//	// Deleting the statefulset should clear the blocking pet
//	if err := psc.psStore.Store.Delete(ps); err != nil {
//		t.Fatalf("Unable to delete pod %v from statefulset controller store.", ps.Name)
//	}
//	if err := psc.Sync(fmt.Sprintf("%v/%v", ps.Namespace, ps.Name)); err != nil {
//		t.Errorf("Error during sync of deleted statefulset %v", err)
//	}
//	fc.pets = []*pcb{}
//	fc.petsCreated = 0
//	if blocking, err := psc.blockingPetStore.Get(ps, fc.getPodList()); err != nil || blocking != nil {
//		t.Errorf("Unexpected blocking pod %v, err %v", blocking, err)
//	}
//	saturateStatefulSet(t, ps, psc, fc)
//
//	// Make sure we don't leak the final blockin pet in the store
//	psc.syncStatefulSet(ps, fc.getPodList())
//	if p, exists, err := psc.blockingPetStore.store.GetByKey(fmt.Sprintf("%v/%v", ps.Namespace, ps.Name)); err != nil || exists {
//		t.Errorf("Unexpected blocking pod, err %v: %+v", err, p)
//	}
//}
//
//func TestSyncStatefulSetBlockedPet(t *testing.T) {
//	psc, fc := newFakeStatefulSetController()
//	ps := newStatefulSet(3)
//	i, _ := psc.syncStatefulSet(ps, fc.getPodList())
//	if i != len(fc.getPodList()) {
//		t.Errorf("syncStatefulSet should return actual amount of pods")
//	}
//}

//type fakeClient struct {
//	fakeinternal.Clientset
//	statefulsetClient *fakeStatefulSetClient
//}
//
//func (c *fakeClient) Apps() v1beta1.AppsV1beta1Interface {
//	return &fakeApps{c, &fake.FakeAppsV1beta1{}}
//}
//
//type fakeApps struct {
//	*fakeClient
//	*fake.FakeAppsV1beta1
//}
//
//func (c *fakeApps) StatefulSets(namespace string) v1beta1.StatefulSetInterface {
//	c.statefulsetClient.Namespace = namespace
//	return c.statefulsetClient
//}
//
//type fakeStatefulSetClient struct {
//	*fake.FakeStatefulSets
//	Namespace string
//	replicas  int32
//}
//
//func (f *fakeStatefulSetClient) UpdateStatus(statefulset *apps.StatefulSet) (*apps.StatefulSet, error) {
//	f.replicas = statefulset.Status.Replicas
//	return statefulset, nil
//}

//func TestStatefulSetReplicaCount(t *testing.T) {
//	fpsc := &fakeStatefulSetClient{}
//	psc, _ := newFakeStatefulSetController()
//	psc.kubeClient = &fakeClient{
//		statefulsetClient: fpsc,
//	}
//
//	ps := newStatefulSet(3)
//	psKey := fmt.Sprintf("%v/%v", ps.Namespace, ps.Name)
//	psc.psStore.Store.Add(ps)
//
//	if err := psc.Sync(psKey); err != nil {
//		t.Errorf("Error during sync of deleted statefulset %v", err)
//	}
//
//	if fpsc.replicas != 1 {
//		t.Errorf("Replicas count sent as status update for StatefulSet should be 1, is %d instead", fpsc.replicas)
//	}
//}
