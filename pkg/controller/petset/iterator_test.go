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

	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestPetQueueCreates(t *testing.T) {
	replicas := 3
	ps := newStatefulSet(replicas)
	q := NewPetQueue(ps, []*api.Pod{})
	for i := 0; i < replicas; i++ {
		pet, _ := newPCB(fmt.Sprintf("%v", i), ps)
		q.enqueue(pet)
		p := q.dequeue()
		if p.event != syncPet {
			t.Errorf("Failed to retrieve sync event from queue")
		}
	}
	if q.dequeue() != nil {
		t.Errorf("Expected no pods")
	}
}

func TestPetQueueScaleDown(t *testing.T) {
	replicas := 1
	ps := newStatefulSet(replicas)

	// knownPods are the pods in the system
	knownPods := newPodList(ps, 3)

	q := NewPetQueue(ps, knownPods)

	// The iterator will insert a single replica, the enqueue
	// mimics that behavior.
	pet, _ := newPCB(fmt.Sprintf("%v", 0), ps)
	q.enqueue(pet)

	deletes := sets.NewString(fmt.Sprintf("%v-1", ps.Name), fmt.Sprintf("%v-2", ps.Name))
	syncs := sets.NewString(fmt.Sprintf("%v-0", ps.Name))

	// Confirm that 2 known pods are deleted
	for i := 0; i < 3; i++ {
		p := q.dequeue()
		switch p.event {
		case syncPet:
			if !syncs.Has(p.pod.Name) {
				t.Errorf("Unexpected sync %v expecting %+v", p.pod.Name, syncs)
			}
		case deletePet:
			if !deletes.Has(p.pod.Name) {
				t.Errorf("Unexpected deletes %v expecting %+v", p.pod.Name, deletes)
			}
		}
	}
	if q.dequeue() != nil {
		t.Errorf("Expected no pods")
	}
}

func TestPetQueueScaleUp(t *testing.T) {
	replicas := 5
	ps := newStatefulSet(replicas)

	// knownPods are pods in the system
	knownPods := newPodList(ps, 2)

	q := NewPetQueue(ps, knownPods)
	for i := 0; i < 5; i++ {
		pet, _ := newPCB(fmt.Sprintf("%v", i), ps)
		q.enqueue(pet)
	}
	for i := 4; i >= 0; i-- {
		pet := q.dequeue()
		expectedName := fmt.Sprintf("%v-%d", ps.Name, i)
		if pet.event != syncPet || pet.pod.Name != expectedName {
			t.Errorf("Unexpected pod %+v, expected %v", pet.pod.Name, expectedName)
		}
	}
}

func TestStatefulSetIteratorRelist(t *testing.T) {
	replicas := 5
	ps := newStatefulSet(replicas)

	// knownPods are pods in the system
	knownPods := newPodList(ps, 5)
	for i := range knownPods {
		knownPods[i].Spec.NodeName = fmt.Sprintf("foo-node-%v", i)
		knownPods[i].Status.Phase = api.PodRunning
	}
	pi := NewStatefulSetIterator(ps, knownPods)

	// A simple resync should not change identity of pods in the system
	i := 0
	for pi.Next() {
		p := pi.Value()
		if identityHash(ps, p.pod) != identityHash(ps, knownPods[i]) {
			t.Errorf("Got unexpected identity hash from iterator.")
		}
		if p.event != syncPet {
			t.Errorf("Got unexpected sync event for %v: %v", p.pod.Name, p.event)
		}
		i++
	}
	if i != 5 {
		t.Errorf("Unexpected iterations %v, this probably means too many/few pods", i)
	}

	// Scale to 0 should delete all pods in system
	ps.Spec.Replicas = 0
	pi = NewStatefulSetIterator(ps, knownPods)
	i = 0
	for pi.Next() {
		p := pi.Value()
		if p.event != deletePet {
			t.Errorf("Got unexpected sync event for %v: %v", p.pod.Name, p.event)
		}
		i++
	}
	if i != 5 {
		t.Errorf("Unexpected iterations %v, this probably means too many/few pods", i)
	}

	// Relist with 0 replicas should no-op
	pi = NewStatefulSetIterator(ps, []*api.Pod{})
	if pi.Next() != false {
		t.Errorf("Unexpected iteration without any replicas or pods in system")
	}
}
