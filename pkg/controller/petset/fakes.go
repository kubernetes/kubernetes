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
	"time"

	inf "gopkg.in/inf.v0"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
)

func dec(i int64, exponent int) *inf.Dec {
	return inf.NewDec(i, inf.Scale(-exponent))
}

func newPVC(name string) v1.PersistentVolumeClaim {
	return v1.PersistentVolumeClaim{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: *resource.NewQuantity(1, resource.BinarySI),
				},
			},
		},
	}
}

func newStatefulSetWithVolumes(replicas int, name string, petMounts []v1.VolumeMount, podMounts []v1.VolumeMount) *apps.StatefulSet {
	mounts := append(petMounts, podMounts...)
	claims := []v1.PersistentVolumeClaim{}
	for _, m := range petMounts {
		claims = append(claims, newPVC(m.Name))
	}

	vols := []v1.Volume{}
	for _, m := range podMounts {
		vols = append(vols, v1.Volume{
			Name: m.Name,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: fmt.Sprintf("/tmp/%v", m.Name),
				},
			},
		})
	}

	return &apps.StatefulSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "StatefulSet",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
			UID:       types.UID("test"),
		},
		Spec: apps.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
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

func runningPod(ns, name string) *v1.Pod {
	p := &v1.Pod{Status: v1.PodStatus{Phase: v1.PodRunning}}
	p.Namespace = ns
	p.Name = name
	return p
}

func newPodList(ps *apps.StatefulSet, num int) []*v1.Pod {
	// knownPods are pods in the system
	knownPods := []*v1.Pod{}
	for i := 0; i < num; i++ {
		k, _ := newPCB(fmt.Sprintf("%v", i), ps)
		knownPods = append(knownPods, k.pod)
	}
	return knownPods
}

func newStatefulSet(replicas int) *apps.StatefulSet {
	petMounts := []v1.VolumeMount{
		{Name: "datadir", MountPath: "/tmp/zookeeper"},
	}
	podMounts := []v1.VolumeMount{
		{Name: "home", MountPath: "/home"},
	}
	return newStatefulSetWithVolumes(replicas, "foo", petMounts, podMounts)
}

func checkPodForMount(pod *v1.Pod, mountName string) error {
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
		claims:           []v1.PersistentVolumeClaim{},
		recorder:         &record.FakeRecorder{},
		petHealthChecker: &defaultPetHealthChecker{},
	}
}

type fakePetClient struct {
	pets          []*pcb
	claims        []v1.PersistentVolumeClaim
	petsCreated   int
	petsDeleted   int
	claimsCreated int
	claimsDeleted int
	recorder      record.EventRecorder
	petHealthChecker
}

// Delete fakes pet client deletion.
func (f *fakePetClient) Delete(p *pcb) error {
	pets := []*pcb{}
	found := false
	for i, pet := range f.pets {
		if p.pod.Name == pet.pod.Name {
			found = true
			f.recorder.Eventf(pet.parent, v1.EventTypeNormal, "SuccessfulDelete", "pod: %v", pet.pod.Name)
			continue
		}
		pets = append(pets, f.pets[i])
	}
	if !found {
		// TODO: Return proper not found error
		return fmt.Errorf("Delete failed: pod %v doesn't exist", p.pod.Name)
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
			return fmt.Errorf("Create failed: pod %v already exists", p.pod.Name)
		}
	}
	f.recorder.Eventf(p.parent, v1.EventTypeNormal, "SuccessfulCreate", "pod: %v", p.pod.Name)
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
			f.pets[i].pod.Annotations[apipod.PodHostnameAnnotation] = wanted.pod.Annotations[apipod.PodHostnameAnnotation]
			f.pets[i].pod.Annotations[apipod.PodSubdomainAnnotation] = wanted.pod.Annotations[apipod.PodSubdomainAnnotation]
			f.pets[i].pod.Spec = wanted.pod.Spec
			found = true
		}
		pets = append(pets, f.pets[i])
	}
	f.pets = pets
	if !found {
		return fmt.Errorf("Cannot update pod %v not found", wanted.pod.Name)
	}
	// TODO: Delete pvcs/volumes that are in wanted but not in expected.
	return nil
}

func (f *fakePetClient) getPodList() []*v1.Pod {
	p := []*v1.Pod{}
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
	if len(f.pets) <= index {
		return fmt.Errorf("Index out of range, len %v index %v", len(f.pets), index)
	}
	f.pets[index].pod.Status.Phase = v1.PodRunning
	f.pets[index].pod.Annotations[StatefulSetInitAnnotation] = "true"
	f.pets[index].pod.Status.Conditions = []v1.PodCondition{
		{Type: v1.PodReady, Status: v1.ConditionTrue},
	}
	return nil
}

// isHealthy is a convenience wrapper around the default health checker.
// The first invocation returns not-healthy, but marks the pet healthy so
// subsequent invocations see it as healthy.
func (f *fakePetClient) isHealthy(pod *v1.Pod) bool {
	if f.petHealthChecker.isHealthy(pod) {
		return true
	}
	return false
}

func (f *fakePetClient) setDeletionTimestamp(index int) error {
	if len(f.pets) <= index {
		return fmt.Errorf("Index out of range, len %v index %v", len(f.pets), index)
	}
	f.pets[index].pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	return nil
}

// SyncPVCs fakes pvc syncing.
func (f *fakePetClient) SyncPVCs(pet *pcb) error {
	v := pet.pvcs
	updateClaims := map[string]v1.PersistentVolumeClaim{}
	for i, update := range v {
		updateClaims[update.Name] = v[i]
	}
	claimList := []v1.PersistentVolumeClaim{}
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
		f.recorder.Eventf(pet.parent, v1.EventTypeNormal, "SuccessfulCreate", "pvc: %v", remaining.Name)
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
	pvcs := []v1.PersistentVolumeClaim{}
	for i, existing := range f.claims {
		if deleteClaimNames.Has(existing.Name) {
			deleteClaimNames.Delete(existing.Name)
			f.claimsDeleted++
			f.recorder.Eventf(pet.parent, v1.EventTypeNormal, "SuccessfulDelete", "pvc: %v", existing.Name)
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
