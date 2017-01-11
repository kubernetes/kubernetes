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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	appsclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/apps/v1beta1"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

// overlappingStatefulSets sorts a list of StatefulSets by creation timestamp, using their names as a tie breaker.
// Generally used to tie break between StatefulSets that have overlapping selectors.
type overlappingStatefulSets []apps.StatefulSet

func (o overlappingStatefulSets) Len() int      { return len(o) }
func (o overlappingStatefulSets) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o overlappingStatefulSets) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

// updatePetCount attempts to update the Status.Replicas of the given StatefulSet, with a single GET/PUT retry.
func updatePetCount(psClient appsclientset.StatefulSetsGetter, ps apps.StatefulSet, numPets int) (updateErr error) {
	if ps.Status.Replicas == int32(numPets) || psClient == nil {
		return nil
	}
	var getErr error
	for i, ps := 0, &ps; ; i++ {
		glog.V(4).Infof(fmt.Sprintf("Updating replica count for StatefulSet: %s/%s, ", ps.Namespace, ps.Name) +
			fmt.Sprintf("replicas %d->%d (need %d), ", ps.Status.Replicas, numPets, *(ps.Spec.Replicas)))

		ps.Status = apps.StatefulSetStatus{Replicas: int32(numPets)}
		_, updateErr = psClient.StatefulSets(ps.Namespace).UpdateStatus(ps)
		if updateErr == nil || i >= statusUpdateRetries {
			return updateErr
		}
		if ps, getErr = psClient.StatefulSets(ps.Namespace).Get(ps.Name, metav1.GetOptions{}); getErr != nil {
			return getErr
		}
	}
}

// unhealthyPetTracker tracks unhealthy pets for statefulsets.
type unhealthyPetTracker struct {
	pc        petClient
	store     cache.Store
	storeLock sync.Mutex
}

// Get returns a previously recorded blocking pet for the given statefulset.
func (u *unhealthyPetTracker) Get(ps *apps.StatefulSet, knownPets []*v1.Pod) (*pcb, error) {
	u.storeLock.Lock()
	defer u.storeLock.Unlock()

	// We "Get" by key but "Add" by object because the store interface doesn't
	// allow us to Get/Add a related obj (eg statefulset: blocking pet).
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
				glog.V(4).Infof("Ignoring healthy pod %v for StatefulSet %v", p.Name, ps.Name)
				continue
			}
			glog.V(4).Infof("No recorded blocking pod, but found unhealthy pod %v for StatefulSet %v", p.Name, ps.Name)
			return &pcb{pod: p, parent: ps}, nil
		}
		return nil, nil
	}

	// This is a pet that's blocking further creates/deletes of a statefulset. If it
	// disappears, it's no longer blocking. If it exists, it continues to block
	// till it turns healthy or disappears.
	bp := obj.(*pcb)
	blockingPet, exists, err := u.pc.Get(bp)
	if err != nil {
		return nil, err
	}
	if !exists {
		glog.V(4).Infof("Clearing blocking pod %v for StatefulSet %v because it's been deleted", bp.pod.Name, ps.Name)
		return nil, nil
	}
	blockingPetPod := blockingPet.pod
	if hc.isHealthy(blockingPetPod) && !hc.isDying(blockingPetPod) {
		glog.V(4).Infof("Clearing blocking pod %v for StatefulSet %v because it's healthy", bp.pod.Name, ps.Name)
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
	glog.V(4).Infof("Adding blocking pod %v for StatefulSet %v", blockingPet.pod.Name, blockingPet.parent.Name)
	return u.store.Add(blockingPet)
}

// newUnHealthyPetTracker tracks unhealthy pets that block progress of statefulsets.
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
		return "", fmt.Errorf("not a valid pod control block %#v", p)
	}
	if p.parent == nil {
		return "", fmt.Errorf("cannot compute pod control block key without parent pointer %#v", p)
	}
	return controller.KeyFunc(p.parent)
}
