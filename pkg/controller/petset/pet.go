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
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/golang/glog"
)

// petLifeCycleEvent is used to communicate high level actions the controller
// needs to take on a given pet. It's recorded in the pcb. The recognized values
// are listed below.
type petLifeCycleEvent string

const (
	syncPet   petLifeCycleEvent = "sync"
	deletePet petLifeCycleEvent = "delete"
	// updateRetries is the number of Get/Update cycles we perform when an
	// update fails.
	updateRetries = 3
	// PetSetInitAnnotation is an annotation which when set, indicates that the
	// pet has finished initializing itself.
	// TODO: Replace this with init container status.
	PetSetInitAnnotation = "pod.alpha.kubernetes.io/initialized"
)

// pcb is the control block used to transmit all updates about a single pet.
// It serves as the manifest for a single pet. Users must populate the pod
// and parent fields to pass it around safely.
type pcb struct {
	// pod is the desired pet pod.
	pod *api.Pod
	// pvcs is a list of desired persistent volume claims for the pet pod.
	pvcs []api.PersistentVolumeClaim
	// event is the lifecycle event associated with this update.
	event petLifeCycleEvent
	// id is the identity index of this pet.
	id string
	// parent is a pointer to the parent petset.
	parent *apps.PetSet
}

// pvcClient is a client for managing persistent volume claims.
type pvcClient interface {
	// DeletePVCs deletes the pvcs in the given pcb.
	DeletePVCs(*pcb) error
	// SyncPVCs creates/updates pvcs in the given pcb.
	SyncPVCs(*pcb) error
}

// petSyncer syncs a single pet.
type petSyncer struct {
	petClient

	// blockingPet is an unhealthy pet either from this iteration or a previous
	// iteration, either because it is not yet Running, or being Deleted, that
	// prevents other creates/deletions.
	blockingPet *pcb
}

// Sync syncs the given pet.
func (p *petSyncer) Sync(pet *pcb) error {
	if pet == nil {
		return nil
	}
	realPet, exists, err := p.Get(pet)
	if err != nil {
		return err
	}
	// There is not constraint except quota on the number of pvcs created.
	// This is done per pet so we get a working cluster ASAP, even if user
	// runs out of quota.
	if err := p.SyncPVCs(pet); err != nil {
		return err
	}
	if exists {
		if !p.isHealthy(realPet.pod) {
			glog.Infof("PetSet %v waiting on unhealthy pet %v", pet.parent.Name, realPet.pod.Name)
		}
		return p.Update(realPet, pet)
	}
	if p.blockingPet != nil {
		glog.Infof("Create of %v in PetSet %v blocked by unhealthy pet %v", pet.pod.Name, pet.parent.Name, p.blockingPet.pod.Name)
		return nil
	}
	// This is counted as a create, even if it fails. We can't skip indices
	// because some pets might allocate a special role to earlier indices.
	// The returned error will force a requeue.
	// TODO: What's the desired behavior if pet-0 is deleted while pet-1 is
	// not yet healthy? currently pet-0 will wait till pet-1 is healthy,
	// this feels safer, but might lead to deadlock.
	p.blockingPet = pet
	if err := p.Create(pet); err != nil {
		return err
	}
	return nil
}

// Delete deletes the given pet, if no other pet in the petset is blocking a
// scale event.
func (p *petSyncer) Delete(pet *pcb) error {
	if pet == nil {
		return nil
	}
	realPet, exists, err := p.Get(pet)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	if p.blockingPet != nil {
		glog.Infof("Delete of %v in PetSet %v blocked by unhealthy pet %v", realPet.pod.Name, pet.parent.Name, p.blockingPet.pod.Name)
		return nil
	}
	// This is counted as a delete, even if it fails.
	// The returned error will force a requeue.
	p.blockingPet = realPet
	if !p.isDying(realPet.pod) {
		glog.Infof("PetSet %v deleting pet %v", pet.parent.Name, pet.pod.Name)
		return p.petClient.Delete(pet)
	}
	glog.Infof("PetSet %v waiting on pet %v to die in %v", pet.parent.Name, realPet.pod.Name, realPet.pod.DeletionTimestamp)
	return nil
}

// petClient is a client for managing pets.
type petClient interface {
	pvcClient
	petHealthChecker
	Delete(*pcb) error
	Get(*pcb) (*pcb, bool, error)
	Create(*pcb) error
	Update(*pcb, *pcb) error
}

// apiServerPetClient is a petset aware Kubernetes client.
type apiServerPetClient struct {
	c        *client.Client
	recorder record.EventRecorder
	petHealthChecker
}

// Get gets the pet in the pcb from the apiserver.
func (p *apiServerPetClient) Get(pet *pcb) (*pcb, bool, error) {
	found := true
	ns := pet.parent.Namespace
	pod, err := podClient(p.c, ns).Get(pet.pod.Name)
	if errors.IsNotFound(err) {
		found = false
		err = nil
	}
	if err != nil || !found {
		return nil, found, err
	}
	realPet := *pet
	realPet.pod = pod
	return &realPet, true, nil
}

// Delete deletes the pet in the pcb from the apiserver.
func (p *apiServerPetClient) Delete(pet *pcb) error {
	err := podClient(p.c, pet.parent.Namespace).Delete(pet.pod.Name, nil)
	if errors.IsNotFound(err) {
		err = nil
	}
	p.event(pet.parent, "Delete", fmt.Sprintf("pet: %v", pet.pod.Name), err)
	return err
}

// Create creates the pet in the pcb.
func (p *apiServerPetClient) Create(pet *pcb) error {
	_, err := podClient(p.c, pet.parent.Namespace).Create(pet.pod)
	p.event(pet.parent, "Create", fmt.Sprintf("pet: %v", pet.pod.Name), err)
	return err
}

// Update updates the pet in the 'pet' pcb to match the pet in the 'expectedPet' pcb.
func (p *apiServerPetClient) Update(pet *pcb, expectedPet *pcb) (updateErr error) {
	var getErr error
	pc := podClient(p.c, pet.parent.Namespace)

	pod, needsUpdate, err := copyPetID(pet, expectedPet)
	if err != nil || !needsUpdate {
		return err
	}
	glog.Infof("Resetting pet %v to match PetSet %v spec", pod.Name, pet.parent.Name)
	for i, p := 0, &pod; ; i++ {
		_, updateErr = pc.Update(p)
		if updateErr == nil || i >= updateRetries {
			return updateErr
		}
		if p, getErr = pc.Get(pod.Name); getErr != nil {
			return getErr
		}
	}
}

// DeletePVCs should delete PVCs, when implemented.
func (p *apiServerPetClient) DeletePVCs(pet *pcb) error {
	// TODO: Implement this when we delete pvcs.
	return nil
}

func (p *apiServerPetClient) getPVC(pvcName, pvcNamespace string) (*api.PersistentVolumeClaim, bool, error) {
	found := true
	pvc, err := claimClient(p.c, pvcNamespace).Get(pvcName)
	if errors.IsNotFound(err) {
		found = false
	}
	if !found {
		return nil, found, nil
	} else if err != nil {
		return nil, found, err
	}
	return pvc, true, nil
}

func (p *apiServerPetClient) createPVC(pvc *api.PersistentVolumeClaim) error {
	_, err := claimClient(p.c, pvc.Namespace).Create(pvc)
	return err
}

// SyncPVCs syncs pvcs in the given pcb.
func (p *apiServerPetClient) SyncPVCs(pet *pcb) error {
	errMsg := ""
	// Create new claims.
	for i, pvc := range pet.pvcs {
		_, exists, err := p.getPVC(pvc.Name, pet.parent.Namespace)
		if !exists {
			var err error
			if err = p.createPVC(&pet.pvcs[i]); err != nil {
				errMsg += fmt.Sprintf("Failed to create %v: %v", pvc.Name, err)
			}
			p.event(pet.parent, "Create", fmt.Sprintf("pvc: %v", pvc.Name), err)
		} else if err != nil {
			errMsg += fmt.Sprintf("Error trying to get pvc %v, %v.", pvc.Name, err)
		}
		// TODO: Check resource requirements and accessmodes, update if necessary
	}
	if len(errMsg) != 0 {
		return fmt.Errorf("%v", errMsg)
	}
	return nil
}

// event formats an event for the given runtime object.
func (p *apiServerPetClient) event(obj runtime.Object, reason, msg string, err error) {
	if err != nil {
		p.recorder.Eventf(obj, api.EventTypeWarning, fmt.Sprintf("Failed%v", reason), fmt.Sprintf("%v, error: %v", msg, err))
	} else {
		p.recorder.Eventf(obj, api.EventTypeNormal, fmt.Sprintf("Successful%v", reason), msg)
	}
}

// petHealthChecker is an interface to check pet health. It makes a boolean
// decision based on the given pod.
type petHealthChecker interface {
	isHealthy(*api.Pod) bool
	isDying(*api.Pod) bool
}

// defaultPetHealthChecks does basic health checking.
// It doesn't update, probe or get the pod.
type defaultPetHealthChecker struct{}

// isHealthy returns true if the pod is running and has the
// "pod.alpha.kubernetes.io/initialized" set to "true".
func (d *defaultPetHealthChecker) isHealthy(pod *api.Pod) bool {
	if pod == nil || pod.Status.Phase != api.PodRunning {
		return false
	}
	initialized, ok := pod.Annotations[PetSetInitAnnotation]
	if !ok {
		glog.Infof("PetSet pod %v in %v, waiting on annotation %v", api.PodRunning, pod.Name, PetSetInitAnnotation)
		return false
	}
	b, err := strconv.ParseBool(initialized)
	if err != nil {
		return false
	}
	return b && api.IsPodReady(pod)
}

// isDying returns true if the pod has a non-nil deletion timestamp. Since the
// timestamp can only decrease, once this method returns true for a given pet, it
// will never return false.
func (d *defaultPetHealthChecker) isDying(pod *api.Pod) bool {
	return pod != nil && pod.DeletionTimestamp != nil
}
