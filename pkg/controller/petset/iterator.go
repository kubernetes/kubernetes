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
	"sort"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/controller"
)

// newPCB generates a new PCB using the id string as a unique qualifier
func newPCB(id string, ps *apps.StatefulSet) (*pcb, error) {
	petPod, err := controller.GetPodFromTemplate(&ps.Spec.Template, ps, nil)
	if err != nil {
		return nil, err
	}
	for _, im := range newIdentityMappers(ps) {
		im.SetIdentity(id, petPod)
	}
	petPVCs := []v1.PersistentVolumeClaim{}
	vMapper := &VolumeIdentityMapper{ps}
	for _, c := range vMapper.GetClaims(id) {
		petPVCs = append(petPVCs, c)
	}
	// TODO: Replace id field with IdentityHash, since id is more than just an index.
	return &pcb{pod: petPod, pvcs: petPVCs, id: id, parent: ps}, nil
}

// petQueue is a custom datastructure that's resembles a queue of pets.
type petQueue struct {
	pets     []*pcb
	idMapper identityMapper
}

// enqueue enqueues the given pet, evicting any pets with the same id
func (pt *petQueue) enqueue(p *pcb) {
	if p == nil {
		pt.pets = append(pt.pets, nil)
		return
	}
	// Pop an existing pet from the know list, append the new pet to the end.
	petList := []*pcb{}
	petID := pt.idMapper.Identity(p.pod)
	for i := range pt.pets {
		if petID != pt.idMapper.Identity(pt.pets[i].pod) {
			petList = append(petList, pt.pets[i])
		}
	}
	pt.pets = petList
	p.event = syncPet
	pt.pets = append(pt.pets, p)
}

// dequeue returns the last element of the queue
func (pt *petQueue) dequeue() *pcb {
	if pt.empty() {
		glog.Warningf("Dequeue invoked on an empty queue")
		return nil
	}
	l := len(pt.pets) - 1
	pet := pt.pets[l]
	pt.pets = pt.pets[:l]
	return pet
}

// empty returns true if the pet queue is empty.
func (pt *petQueue) empty() bool {
	return len(pt.pets) == 0
}

// NewPetQueue returns a queue for tracking pets
func NewPetQueue(ps *apps.StatefulSet, podList []*v1.Pod) *petQueue {
	pt := petQueue{pets: []*pcb{}, idMapper: &NameIdentityMapper{ps}}
	// Seed the queue with existing pets. Assume all pets are scheduled for
	// deletion, enqueuing a pet will "undelete" it. We always want to delete
	// from the higher ids, so sort by creation timestamp.

	sort.Sort(PodsByCreationTimestamp(podList))
	vMapper := VolumeIdentityMapper{ps}
	for i := range podList {
		pod := podList[i]
		pt.pets = append(pt.pets, &pcb{pod: pod, pvcs: vMapper.GetClaimsForPet(pod), parent: ps, event: deletePet, id: fmt.Sprintf("%v", i)})
	}
	return &pt
}

// statefulsetIterator implements a simple iterator over pets in the given statefulset.
type statefulSetIterator struct {
	// ps is the statefulset for this iterator.
	ps *apps.StatefulSet
	// queue contains the elements to iterate over.
	queue *petQueue
	// errs is a list because we always want the iterator to drain.
	errs []error
	// petCount is the number of pets iterated over.
	petCount int32
}

// Next returns true for as long as there are elements in the underlying queue.
func (pi *statefulSetIterator) Next() bool {
	var pet *pcb
	var err error
	if pi.petCount < *(pi.ps.Spec.Replicas) {
		pet, err = newPCB(fmt.Sprintf("%d", pi.petCount), pi.ps)
		if err != nil {
			pi.errs = append(pi.errs, err)
			// Don't stop iterating over the set on errors. Caller handles nil.
			pet = nil
		}
		pi.queue.enqueue(pet)
		pi.petCount++
	}
	// Keep the iterator running till we've deleted pets in the queue.
	return !pi.queue.empty()
}

// Value dequeues an element from the queue.
func (pi *statefulSetIterator) Value() *pcb {
	return pi.queue.dequeue()
}

// NewStatefulSetIterator returns a new iterator. All pods in the given podList
// are used to seed the queue of the iterator.
func NewStatefulSetIterator(ps *apps.StatefulSet, podList []*v1.Pod) *statefulSetIterator {
	pi := &statefulSetIterator{
		ps:       ps,
		queue:    NewPetQueue(ps, podList),
		errs:     []error{},
		petCount: 0,
	}
	return pi
}

// PodsByCreationTimestamp sorts a list of Pods by creation timestamp, using their names as a tie breaker.
type PodsByCreationTimestamp []*v1.Pod

func (o PodsByCreationTimestamp) Len() int      { return len(o) }
func (o PodsByCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o PodsByCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}
