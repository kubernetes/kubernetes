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
	"time"

	"k8s.io/kubernetes/pkg/api"
	api_pod "k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
	"speter.net/go/exp/math/dec/inf"
)

func dec(i int64, exponent int) *inf.Dec {
	return inf.NewDec(i, inf.Scale(-exponent))
}

func newPVC(name string) api.PersistentVolumeClaim {
	return api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeClaimSpec{
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceStorage: resource.Quantity{
						Amount: dec(1, 0),
						Format: resource.BinarySI,
					},
				},
			},
		},
	}
}

func newPetSetWithVolumes(replicas int, name string, petMounts []api.VolumeMount, podMounts []api.VolumeMount) *apps.PetSet {
	mounts := append(petMounts, podMounts...)
	claims := []api.PersistentVolumeClaim{}
	for _, m := range petMounts {
		claims = append(claims, newPVC(m.Name))
	}

	vols := []api.Volume{}
	for _, m := range podMounts {
		vols = append(vols, api.Volume{
			Name: m.Name,
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: fmt.Sprintf("/tmp/%v", m.Name),
				},
			},
		})
	}

	return &apps.PetSet{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "PetSet",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
			UID:       types.UID("test"),
		},
		Spec: apps.PetSetSpec{
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
			Replicas: replicas,
			Template: api.PodTemplateSpec{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:         "nginx",
							Image:        "nginx",
							VolumeMounts: mounts,
						},
					},
					Volumes: vols,
				},
			},
			VolumeClaimTemplates: claims,
			ServiceName:          "governingsvc",
		},
	}
}

func runningPod(ns, name string) *api.Pod {
	p := &api.Pod{Status: api.PodStatus{Phase: api.PodRunning}}
	p.Namespace = ns
	p.Name = name
	return p
}

func newPodList(ps *apps.PetSet, num int) []*api.Pod {
	// knownPods are pods in the system
	knownPods := []*api.Pod{}
	for i := 0; i < num; i++ {
		k, _ := newPCB(fmt.Sprintf("%v", i), ps)
		knownPods = append(knownPods, k.pod)
	}
	return knownPods
}

func newPetSet(replicas int) *apps.PetSet {
	petMounts := []api.VolumeMount{
		{Name: "datadir", MountPath: "/tmp/zookeeper"},
	}
	podMounts := []api.VolumeMount{
		{Name: "home", MountPath: "/home"},
	}
	return newPetSetWithVolumes(replicas, "foo", petMounts, podMounts)
}

func checkPodForMount(pod *api.Pod, mountName string) error {
	for _, c := range pod.Spec.Containers {
		for _, v := range c.VolumeMounts {
			if v.Name == mountName {
				return nil
			}
		}
	}
	return fmt.Errorf("Found volume but no associated mount %v in pod %v", mountName, pod.Name)
}

func newFakePetClient() *fakePetClient {
	return &fakePetClient{
		pets:             []*pcb{},
		claims:           []api.PersistentVolumeClaim{},
		recorder:         &record.FakeRecorder{},
		petHealthChecker: &defaultPetHealthChecker{},
	}
}

type fakePetClient struct {
	pets                         []*pcb
	claims                       []api.PersistentVolumeClaim
	petsCreated, petsDeleted     int
	claimsCreated, claimsDeleted int
	recorder                     record.EventRecorder
	petHealthChecker
}

// Delete fakes pet client deletion.
func (f *fakePetClient) Delete(p *pcb) error {
	pets := []*pcb{}
	found := false
	for i, pet := range f.pets {
		if p.pod.Name == pet.pod.Name {
			found = true
			f.recorder.Eventf(pet.parent, api.EventTypeNormal, "SuccessfulDelete", "pet: %v", pet.pod.Name)
			continue
		}
		pets = append(pets, f.pets[i])
	}
	if !found {
		// TODO: Return proper not found error
		return fmt.Errorf("Delete failed: pet %v doesn't exist", p.pod.Name)
	}
	f.pets = pets
	f.petsDeleted++
	return nil
}

// Get fakes getting pets.
func (f *fakePetClient) Get(p *pcb) (*pcb, bool, error) {
	for i, pet := range f.pets {
		if p.pod.Name == pet.pod.Name {
			return f.pets[i], true, nil
		}
	}
	return nil, false, nil
}

// Create fakes pet creation.
func (f *fakePetClient) Create(p *pcb) error {
	for _, pet := range f.pets {
		if p.pod.Name == pet.pod.Name {
			return fmt.Errorf("Create failed: pet %v already exists", p.pod.Name)
		}
	}
	f.recorder.Eventf(p.parent, api.EventTypeNormal, "SuccessfulCreate", "pet: %v", p.pod.Name)
	f.pets = append(f.pets, p)
	f.petsCreated++
	return nil
}

// Update fakes pet updates.
func (f *fakePetClient) Update(expected, wanted *pcb) error {
	found := false
	pets := []*pcb{}
	for i, pet := range f.pets {
		if wanted.pod.Name == pet.pod.Name {
			f.pets[i].pod.Annotations[api_pod.PodHostnameAnnotation] = wanted.pod.Annotations[api_pod.PodHostnameAnnotation]
			f.pets[i].pod.Annotations[api_pod.PodSubdomainAnnotation] = wanted.pod.Annotations[api_pod.PodSubdomainAnnotation]
			f.pets[i].pod.Spec = wanted.pod.Spec
			found = true
		}
		pets = append(pets, f.pets[i])
	}
	f.pets = pets
	if !found {
		return fmt.Errorf("Cannot update pet %v not found", wanted.pod.Name)
	}
	// TODO: Delete pvcs/volumes that are in wanted but not in expected.
	return nil
}

func (f *fakePetClient) getPodList() []*api.Pod {
	p := []*api.Pod{}
	for i, pet := range f.pets {
		if pet.pod == nil {
			continue
		}
		p = append(p, f.pets[i].pod)
	}
	return p
}

func (f *fakePetClient) deletePetAtIndex(index int) {
	p := []*pcb{}
	for i := range f.pets {
		if i != index {
			p = append(p, f.pets[i])
		}
	}
	f.pets = p
}

func (f *fakePetClient) setHealthy(index int) error {
	if len(f.pets) < index {
		return fmt.Errorf("Index out of range, len %v index %v", len(f.pets), index)
	}
	f.pets[index].pod.Status.Phase = api.PodRunning
	f.pets[index].pod.Annotations[PetSetInitAnnotation] = "true"
	return nil
}

// isHealthy is a convenience wrapper around the default health checker.
// The first invocation returns not-healthy, but marks the pet healthy so
// subsequent invocations see it as healthy.
func (f *fakePetClient) isHealthy(pod *api.Pod) bool {
	if f.petHealthChecker.isHealthy(pod) {
		return true
	}
	return false
}

func (f *fakePetClient) setDeletionTimestamp(index int) error {
	if len(f.pets) < index {
		return fmt.Errorf("Index out of range, len %v index %v", len(f.pets), index)
	}
	f.pets[index].pod.DeletionTimestamp = &unversioned.Time{Time: time.Now()}
	return nil
}

// SyncPVCs fakes pvc syncing.
func (f *fakePetClient) SyncPVCs(pet *pcb) error {
	v := pet.pvcs
	updateClaims := map[string]api.PersistentVolumeClaim{}
	for i, update := range v {
		updateClaims[update.Name] = v[i]
	}
	claimList := []api.PersistentVolumeClaim{}
	for i, existing := range f.claims {
		if update, ok := updateClaims[existing.Name]; ok {
			claimList = append(claimList, update)
			delete(updateClaims, existing.Name)
		} else {
			claimList = append(claimList, f.claims[i])
		}
	}
	for _, remaining := range updateClaims {
		claimList = append(claimList, remaining)
		f.claimsCreated++
		f.recorder.Eventf(pet.parent, api.EventTypeNormal, "SuccessfulCreate", "pvc: %v", remaining.Name)
	}
	f.claims = claimList
	return nil
}

// DeletePVCs fakes pvc deletion.
func (f *fakePetClient) DeletePVCs(pet *pcb) error {
	claimsToDelete := pet.pvcs
	deleteClaimNames := sets.NewString()
	for _, c := range claimsToDelete {
		deleteClaimNames.Insert(c.Name)
	}
	pvcs := []api.PersistentVolumeClaim{}
	for i, existing := range f.claims {
		if deleteClaimNames.Has(existing.Name) {
			deleteClaimNames.Delete(existing.Name)
			f.claimsDeleted++
			f.recorder.Eventf(pet.parent, api.EventTypeNormal, "SuccessfulDelete", "pvc: %v", existing.Name)
			continue
		}
		pvcs = append(pvcs, f.claims[i])
	}
	f.claims = pvcs
	if deleteClaimNames.Len() != 0 {
		return fmt.Errorf("Claims %+v don't exist. Failed deletion.", deleteClaimNames)
	}
	return nil
}
