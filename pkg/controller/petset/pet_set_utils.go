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
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

// overlappingPetSets sorts a list of PetSets by creation timestamp, using their names as a tie breaker.
// Generally used to tie break between PetSets that have overlapping selectors.
type overlappingPetSets []apps.PetSet

func (o overlappingPetSets) Len() int      { return len(o) }
func (o overlappingPetSets) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o overlappingPetSets) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

// updatePetCount attempts to update the Status.Replicas of the given PetSet, with a single GET/PUT retry.
func updatePetCount(kubeClient *client.Client, ps apps.PetSet, numPets int) (updateErr error) {
	if ps.Status.Replicas == numPets || kubeClient == nil {
		return nil
	}
	psClient := kubeClient.Apps().PetSets(ps.Namespace)
	var getErr error
	for i, ps := 0, &ps; ; i++ {
		glog.V(4).Infof(fmt.Sprintf("Updating replica count for PetSet: %s/%s, ", ps.Namespace, ps.Name) +
			fmt.Sprintf("replicas %d->%d (need %d), ", ps.Status.Replicas, numPets, ps.Spec.Replicas))

		ps.Status = apps.PetSetStatus{Replicas: numPets}
		_, updateErr = psClient.UpdateStatus(ps)
		if updateErr == nil || i >= statusUpdateRetries {
			return updateErr
		}
		if ps, getErr = psClient.Get(ps.Name); getErr != nil {
			return getErr
		}
	}
}

// claimClient returns the pvcClient for the given kubeClient/ns.
func claimClient(kubeClient *client.Client, ns string) client.PersistentVolumeClaimInterface {
	return kubeClient.PersistentVolumeClaims(ns)
}

// podClient returns the given podClient for the given kubeClient/ns.
func podClient(kubeClient *client.Client, ns string) client.PodInterface {
	return kubeClient.Pods(ns)
}

// unhealthyPetTracker tracks unhealthy pets for petsets.
type unhealthyPetTracker struct {
	pc        petClient
	store     cache.Store
	storeLock sync.Mutex
}

// Get returns a previously recorded blocking pet for the given petset.
func (u *unhealthyPetTracker) Get(ps *apps.PetSet, knownPets []*api.Pod) (*pcb, error) {
	u.storeLock.Lock()
	defer u.storeLock.Unlock()

	// We "Get" by key but "Add" by object because the store interface doesn't
	// allow us to Get/Add a related obj (eg petset: blocking pet).
	key, err := controller.KeyFunc(ps)
	if err != nil {
		return nil, err
	}
	obj, exists, err := u.store.GetByKey(key)
	if err != nil {
		return nil, err
	}

	hc := defaultPetHealthChecker{}
	// There's no unhealthy pet blocking a scale event, but this might be
	// a controller manager restart. If it is, knownPets can be trusted.
	if !exists {
		for _, p := range knownPets {
			if hc.isHealthy(p) && !hc.isDying(p) {
				glog.V(4).Infof("Ignoring healthy pet %v for PetSet %v", p.Name, ps.Name)
				continue
			}
			glog.Infof("No recorded blocking pet, but found unhealthy pet %v for PetSet %v", p.Name, ps.Name)
			return &pcb{pod: p, parent: ps}, nil
		}
		return nil, nil
	}

	// This is a pet that's blocking further creates/deletes of a petset. If it
	// disappears, it's no longer blocking. If it exists, it continues to block
	// till it turns healthy or disappears.
	bp := obj.(*pcb)
	blockingPet, exists, err := u.pc.Get(bp)
	if err != nil {
		return nil, err
	}
	if !exists {
		glog.V(4).Infof("Clearing blocking pet %v for PetSet %v because it's been deleted", bp.pod.Name, ps.Name)
		return nil, nil
	}
	blockingPetPod := blockingPet.pod
	if hc.isHealthy(blockingPetPod) && !hc.isDying(blockingPetPod) {
		glog.V(4).Infof("Clearing blocking pet %v for PetSet %v because it's healthy", bp.pod.Name, ps.Name)
		u.store.Delete(blockingPet)
		blockingPet = nil
	}
	return blockingPet, nil
}

// Add records the given pet as a blocking pet.
func (u *unhealthyPetTracker) Add(blockingPet *pcb) error {
	u.storeLock.Lock()
	defer u.storeLock.Unlock()

	if blockingPet == nil {
		return nil
	}
	glog.V(4).Infof("Adding blocking pet %v for PetSet %v", blockingPet.pod.Name, blockingPet.parent.Name)
	return u.store.Add(blockingPet)
}

// newUnHealthyPetTracker tracks unhealthy pets that block progress of petsets.
func newUnHealthyPetTracker(pc petClient) *unhealthyPetTracker {
	return &unhealthyPetTracker{pc: pc, store: cache.NewStore(pcbKeyFunc)}
}

// pcbKeyFunc computes the key for a given pcb.
// If it's given a key, it simply returns it.
func pcbKeyFunc(obj interface{}) (string, error) {
	if key, ok := obj.(string); ok {
		return key, nil
	}
	p, ok := obj.(*pcb)
	if !ok {
		return "", fmt.Errorf("not a valid pet control block %#v", p)
	}
	if p.parent == nil {
		return "", fmt.Errorf("cannot compute pet control block key without parent pointer %#v", p)
	}
	return controller.KeyFunc(p.parent)
}
