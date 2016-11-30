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
	"math/rand"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	fakeinternal "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1beta1/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/errors"
)

func newFakeStatefulSetController() (*StatefulSetController, *fakePetClient) {
	fpc := newFakePetClient()
	return &StatefulSetController{
		kubeClient:       nil,
		blockingPetStore: newUnHealthyPetTracker(fpc),
		podStoreSynced:   func() bool { return true },
		psStore:          cache.StoreToStatefulSetLister{Store: cache.NewStore(controller.KeyFunc)},
		podStore:         cache.StoreToPodLister{Indexer: cache.NewIndexer(controller.KeyFunc, cache.Indexers{})},
		newSyncer: func(blockingPet *pcb) *petSyncer {
			return &petSyncer{fpc, blockingPet}
		},
	}, fpc
}

func checkPets(ps *apps.StatefulSet, creates, deletes int, fc *fakePetClient, t *testing.T) {
	if fc.petsCreated != creates || fc.petsDeleted != deletes {
		t.Errorf("Found (creates: %d, deletes: %d), expected (creates: %d, deletes: %d)", fc.petsCreated, fc.petsDeleted, creates, deletes)
	}
	gotClaims := map[string]v1.PersistentVolumeClaim{}
	for _, pvc := range fc.claims {
		gotClaims[pvc.Name] = pvc
	}
	for i := range fc.pets {
		expectedPet, _ := newPCB(fmt.Sprintf("%v", i), ps)
		if identityHash(ps, fc.pets[i].pod) != identityHash(ps, expectedPet.pod) {
			t.Errorf("Unexpected pod at index %d", i)
		}
		for _, pvc := range expectedPet.pvcs {
			gotPVC, ok := gotClaims[pvc.Name]
			if !ok {
				t.Errorf("PVC %v not created for pod %v", pvc.Name, expectedPet.pod.Name)
			}
			if !reflect.DeepEqual(gotPVC.Spec, pvc.Spec) {
				t.Errorf("got PVC %v differs from created pvc", pvc.Name)
			}
		}
	}
}

func scaleStatefulSet(t *testing.T, ps *apps.StatefulSet, psc *StatefulSetController, fc *fakePetClient, scale int) error {
	errs := []error{}
	for i := 0; i < scale; i++ {
		pl := fc.getPodList()
		if len(pl) != i {
			t.Errorf("Unexpected number of pods, expected %d found %d", i, len(pl))
		}
		if _, syncErr := psc.syncStatefulSet(ps, pl); syncErr != nil {
			errs = append(errs, syncErr)
		}
		fc.setHealthy(i)
		checkPets(ps, i+1, 0, fc, t)
	}
	return errors.NewAggregate(errs)
}

func saturateStatefulSet(t *testing.T, ps *apps.StatefulSet, psc *StatefulSetController, fc *fakePetClient) {
	err := scaleStatefulSet(t, ps, psc, fc, int(*(ps.Spec.Replicas)))
	if err != nil {
		t.Errorf("Error scaleStatefulSet: %v", err)
	}
}

func TestStatefulSetControllerCreates(t *testing.T) {
	psc, fc := newFakeStatefulSetController()
	replicas := 3
	ps := newStatefulSet(replicas)

	saturateStatefulSet(t, ps, psc, fc)

	podList := fc.getPodList()
	// Deleted pet gets recreated
	fc.pets = fc.pets[:replicas-1]
	if _, err := psc.syncStatefulSet(ps, podList); err != nil {
		t.Errorf("Error syncing StatefulSet: %v", err)
	}
	checkPets(ps, replicas+1, 0, fc, t)
}

func TestStatefulSetControllerDeletes(t *testing.T) {
	psc, fc := newFakeStatefulSetController()
	replicas := 4
	ps := newStatefulSet(replicas)

	saturateStatefulSet(t, ps, psc, fc)

	// Drain
	errs := []error{}
	*(ps.Spec.Replicas) = 0
	knownPods := fc.getPodList()
	for i := replicas - 1; i >= 0; i-- {
		if len(fc.pets) != i+1 {
			t.Errorf("Unexpected number of pods, expected %d found %d", i+1, len(fc.pets))
		}
		if _, syncErr := psc.syncStatefulSet(ps, knownPods); syncErr != nil {
			errs = append(errs, syncErr)
		}
	}
	if len(errs) != 0 {
		t.Errorf("Error syncing StatefulSet: %v", errors.NewAggregate(errs))
	}
	checkPets(ps, replicas, replicas, fc, t)
}

func TestStatefulSetControllerRespectsTermination(t *testing.T) {
	psc, fc := newFakeStatefulSetController()
	replicas := 4
	ps := newStatefulSet(replicas)

	saturateStatefulSet(t, ps, psc, fc)

	fc.setDeletionTimestamp(replicas - 1)
	*(ps.Spec.Replicas) = 2
	_, err := psc.syncStatefulSet(ps, fc.getPodList())
	if err != nil {
		t.Errorf("Error syncing StatefulSet: %v", err)
	}
	// Finding a pod with the deletion timestamp will pause all deletions.
	knownPods := fc.getPodList()
	if len(knownPods) != 4 {
		t.Errorf("Pods deleted prematurely before deletion timestamp expired, len %d", len(knownPods))
	}
	fc.pets = fc.pets[:replicas-1]
	_, err = psc.syncStatefulSet(ps, fc.getPodList())
	if err != nil {
		t.Errorf("Error syncing StatefulSet: %v", err)
	}
	checkPets(ps, replicas, 1, fc, t)
}

func TestStatefulSetControllerRespectsOrder(t *testing.T) {
	psc, fc := newFakeStatefulSetController()
	replicas := 4
	ps := newStatefulSet(replicas)

	saturateStatefulSet(t, ps, psc, fc)

	errs := []error{}
	*(ps.Spec.Replicas) = 0
	// Shuffle known list and check that pets are deleted in reverse
	knownPods := fc.getPodList()
	for i := range knownPods {
		j := rand.Intn(i + 1)
		knownPods[i], knownPods[j] = knownPods[j], knownPods[i]
	}

	for i := 0; i < replicas; i++ {
		if len(fc.pets) != replicas-i {
			t.Errorf("Unexpected number of pods, expected %d found %d", i, len(fc.pets))
		}
		if _, syncErr := psc.syncStatefulSet(ps, knownPods); syncErr != nil {
			errs = append(errs, syncErr)
		}
		checkPets(ps, replicas, i+1, fc, t)
	}
	if len(errs) != 0 {
		t.Errorf("Error syncing StatefulSet: %v", errors.NewAggregate(errs))
	}
}

func TestStatefulSetControllerBlocksScaling(t *testing.T) {
	psc, fc := newFakeStatefulSetController()
	replicas := 5
	ps := newStatefulSet(replicas)
	scaleStatefulSet(t, ps, psc, fc, 3)

	// Create 4th pet, then before flipping it to healthy, kill the first pet.
	// There should only be 1 not-healty pet at a time.
	pl := fc.getPodList()
	if _, err := psc.syncStatefulSet(ps, pl); err != nil {
		t.Errorf("Error syncing StatefulSet: %v", err)
	}

	deletedPod := pl[0]
	fc.deletePetAtIndex(0)
	pl = fc.getPodList()
	if _, err := psc.syncStatefulSet(ps, pl); err != nil {
		t.Errorf("Error syncing StatefulSet: %v", err)
	}
	newPodList := fc.getPodList()
	for _, p := range newPodList {
		if p.Name == deletedPod.Name {
			t.Errorf("Deleted pod was created while existing pod was unhealthy")
		}
	}

	fc.setHealthy(len(newPodList) - 1)
	if _, err := psc.syncStatefulSet(ps, pl); err != nil {
		t.Errorf("Error syncing StatefulSet: %v", err)
	}

	found := false
	for _, p := range fc.getPodList() {
		if p.Name == deletedPod.Name {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Deleted pod was not created after existing pods became healthy")
	}
}

func TestStatefulSetBlockingPetIsCleared(t *testing.T) {
	psc, fc := newFakeStatefulSetController()
	ps := newStatefulSet(3)
	scaleStatefulSet(t, ps, psc, fc, 1)

	if blocking, err := psc.blockingPetStore.Get(ps, fc.getPodList()); err != nil || blocking != nil {
		t.Errorf("Unexpected blocking pod %v, err %v", blocking, err)
	}

	// 1 not yet healthy pet
	psc.syncStatefulSet(ps, fc.getPodList())

	if blocking, err := psc.blockingPetStore.Get(ps, fc.getPodList()); err != nil || blocking == nil {
		t.Errorf("Expected blocking pod %v, err %v", blocking, err)
	}

	// Deleting the statefulset should clear the blocking pet
	if err := psc.psStore.Store.Delete(ps); err != nil {
		t.Fatalf("Unable to delete pod %v from statefulset controller store.", ps.Name)
	}
	if err := psc.Sync(fmt.Sprintf("%v/%v", ps.Namespace, ps.Name)); err != nil {
		t.Errorf("Error during sync of deleted statefulset %v", err)
	}
	fc.pets = []*pcb{}
	fc.petsCreated = 0
	if blocking, err := psc.blockingPetStore.Get(ps, fc.getPodList()); err != nil || blocking != nil {
		t.Errorf("Unexpected blocking pod %v, err %v", blocking, err)
	}
	saturateStatefulSet(t, ps, psc, fc)

	// Make sure we don't leak the final blockin pet in the store
	psc.syncStatefulSet(ps, fc.getPodList())
	if p, exists, err := psc.blockingPetStore.store.GetByKey(fmt.Sprintf("%v/%v", ps.Namespace, ps.Name)); err != nil || exists {
		t.Errorf("Unexpected blocking pod, err %v: %+v", err, p)
	}
}

func TestSyncStatefulSetBlockedPet(t *testing.T) {
	psc, fc := newFakeStatefulSetController()
	ps := newStatefulSet(3)
	i, _ := psc.syncStatefulSet(ps, fc.getPodList())
	if i != len(fc.getPodList()) {
		t.Errorf("syncStatefulSet should return actual amount of pods")
	}
}

type fakeClient struct {
	fakeinternal.Clientset
	statefulsetClient *fakeStatefulSetClient
}

func (c *fakeClient) Apps() v1beta1.AppsV1beta1Interface {
	return &fakeApps{c, &fake.FakeAppsV1beta1{}}
}

type fakeApps struct {
	*fakeClient
	*fake.FakeAppsV1beta1
}

func (c *fakeApps) StatefulSets(namespace string) v1beta1.StatefulSetInterface {
	c.statefulsetClient.Namespace = namespace
	return c.statefulsetClient
}

type fakeStatefulSetClient struct {
	*fake.FakeStatefulSets
	Namespace string
	replicas  int32
}

func (f *fakeStatefulSetClient) UpdateStatus(statefulset *apps.StatefulSet) (*apps.StatefulSet, error) {
	f.replicas = statefulset.Status.Replicas
	return statefulset, nil
}

func TestStatefulSetReplicaCount(t *testing.T) {
	fpsc := &fakeStatefulSetClient{}
	psc, _ := newFakeStatefulSetController()
	psc.kubeClient = &fakeClient{
		statefulsetClient: fpsc,
	}

	ps := newStatefulSet(3)
	psKey := fmt.Sprintf("%v/%v", ps.Namespace, ps.Name)
	psc.psStore.Store.Add(ps)

	if err := psc.Sync(psKey); err != nil {
		t.Errorf("Error during sync of deleted statefulset %v", err)
	}

	if fpsc.replicas != 1 {
		t.Errorf("Replicas count sent as status update for StatefulSet should be 1, is %d instead", fpsc.replicas)
	}
}
