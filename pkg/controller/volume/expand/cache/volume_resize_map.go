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

package cache

import (
	"fmt"
	"sync"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

type VolumeResizeMap interface {
	AddPvcUpdate(newPvc *v1.PersistentVolumeClaim, oldPvc *v1.PersistentVolumeClaim, spec *volume.Spec)
	GetPvcsWithResizeRequest() []*PvcWithResizeRequest
	GetPvcsWithUpdateNeeded() []*PvcWithResizeRequest
	// Mark this volume as resize
	MarkAsResized(*PvcWithResizeRequest) error
	MarkForFileSystemResize(*PvcWithResizeRequest) error
	MarkResizeFailed(*PvcWithResizeRequest, string) error
}

type volumeResizeMap struct {
	pvcrs      map[types.UniquePvcName]*PvcWithResizeRequest
	kubeClient clientset.Interface
	sync.RWMutex
}

type PvcWithResizeRequest struct {
	PVC          *v1.PersistentVolumeClaim
	VolumeSpec   *volume.Spec
	CurrentSize  resource.Quantity
	ExpectedSize resource.Quantity
	ResizeDone   bool
	UpdateNeeded *updateNeeded
	FailedReason *string
}

type updateNeeded string

const (
	Resized      updateNeeded = "resized"
	ResizeFailed updateNeeded = "resizeFailed"
	FsResize     updateNeeded = "fsResize"
)

func (pvcr *PvcWithResizeRequest) UniquePvcKey() types.UniquePvcName {
	return types.UniquePvcName(pvcr.PVC.UID)
}

func (pvcr *PvcWithResizeRequest) QualifiedName() string {
	return strings.JoinQualifiedName(pvcr.PVC.Namespace, pvcr.PVC.Name)
}

func NewVolumeResizeMap(kubeClient clientset.Interface) VolumeResizeMap {
	resizeMap := &volumeResizeMap{}
	resizeMap.pvcrs = make(map[types.UniquePvcName]*PvcWithResizeRequest)
	resizeMap.kubeClient = kubeClient
	return resizeMap
}

func (resizeMap *volumeResizeMap) AddPvcUpdate(newPvc *v1.PersistentVolumeClaim, oldPvc *v1.PersistentVolumeClaim, spec *volume.Spec) {
	resizeMap.Lock()
	defer resizeMap.Unlock()

	newSize := newPvc.Spec.Resources.Requests[v1.ResourceStorage]
	oldSize := oldPvc.Spec.Resources.Requests[v1.ResourceStorage]
	glog.Infof(" Checking size of stuff new %v vs old %v", newSize, oldSize)

	if newSize.Cmp(oldSize) > 0 {
		pvcRequest := &PvcWithResizeRequest{
			PVC:          newPvc,
			CurrentSize:  newPvc.Status.Capacity[v1.ResourceStorage],
			ExpectedSize: newSize,
			VolumeSpec:   spec,
			ResizeDone:   false,
		}
		resizeMap.pvcrs[types.UniquePvcName(newPvc.UID)] = pvcRequest
	}
}

// Return Pvcrs that require resize
func (resizeMap *volumeResizeMap) GetPvcsWithResizeRequest() []*PvcWithResizeRequest {
	resizeMap.Lock()
	defer resizeMap.Unlock()

	pvcrs := []*PvcWithResizeRequest{}
	for _, pvcr := range resizeMap.pvcrs {
		if !pvcr.ResizeDone {
			pvcrs = append(pvcrs, pvcr)
		}
	}
	return pvcrs
}

// Return Pvcrs that require update of their API objects
func (resizeMap *volumeResizeMap) GetPvcsWithUpdateNeeded() []*PvcWithResizeRequest {
	resizeMap.Lock()
	defer resizeMap.Unlock()

	pvcrs := []*PvcWithResizeRequest{}
	for _, pvcr := range resizeMap.pvcrs {
		// TODO: when to delete pvcr from resizeMap
		if pvcr.ResizeDone && pvcr.UpdateNeeded != nil {
			pvcrs = append(pvcrs, pvcr)
		}
	}
	return pvcrs
}

func (resizeMap *volumeResizeMap) MarkAsResized(pvcr *PvcWithResizeRequest) error {
	resizeMap.Lock()
	defer resizeMap.Unlock()

	pvcUniqueName := pvcr.UniquePvcKey()

	if pvcr, ok := resizeMap.pvcrs[pvcUniqueName]; ok {
		pvcr.ResizeDone = true
		update := Resized
		pvcr.UpdateNeeded = &update
	}

	// This needs to be done atomically somehow so as these operations succeed or fail together. :(
	// If any part of this update fails, sync should GetPvcsWithUpdateNeeded and update each

	err := resizeMap.updatePvSize(pvcr, pvcr.ExpectedSize)
	if err != nil {
		glog.V(4).Infof("Error updating PV spec capacity for volume %q with : %v", pvcr.QualifiedName(), err)
		return err
	}

	readyCondition := v1.PvcCondition{
		Type:   v1.PvcReady,
		Status: v1.ConditionTrue,
	}

	err = resizeMap.updatePvcCapacityAndConditions(pvcr, pvcr.ExpectedSize, readyCondition)
	if err != nil {
		glog.V(4).Infof("Error updating PV spec capacity for volume %q with : %v", pvcr.QualifiedName(), err)
		return err
	}

	if pvcr, ok := resizeMap.pvcrs[pvcUniqueName]; ok {
		pvcr.UpdateNeeded = nil
	}
	return nil
}

func (resizeMap *volumeResizeMap) MarkResizeFailed(pvcr *PvcWithResizeRequest, reason string) error {
	resizeMap.Lock()
	defer resizeMap.Unlock()

	pvcUniqueName := pvcr.UniquePvcKey()

	if pvcr, ok := resizeMap.pvcrs[pvcUniqueName]; ok {
		pvcr.ResizeDone = true
		update := ResizeFailed
		pvcr.UpdateNeeded = &update
		pvcr.FailedReason = &reason
	}

	// This needs to be done atomically somehow so as these operations succeed or fail together. :(
	// If any part of this update fails, sync should GetPvcsWithUpdateNeeded and update each

	failedCondition := v1.PvcCondition{
		Type:   v1.PvcResizeFailed,
		Status: v1.ConditionTrue,
		Reason: reason,
	}

	err := resizeMap.updatePvcCondition(pvcr, failedCondition)
	if err != nil {
		glog.V(4).Infof("Error updating pvc condition for volume %q with : %v", pvcr.QualifiedName(), err)
		return err
	}

	if pvcr, ok := resizeMap.pvcrs[pvcUniqueName]; ok {
		pvcr.UpdateNeeded = nil
	}
	return nil
}

func (resizeMap *volumeResizeMap) MarkForFileSystemResize(pvcr *PvcWithResizeRequest) error {
	resizeMap.Lock()
	defer resizeMap.Unlock()

	pvcUniqueName := pvcr.UniquePvcKey()

	if pvcr, ok := resizeMap.pvcrs[pvcUniqueName]; ok {
		pvcr.ResizeDone = true
		update := FsResize
		pvcr.UpdateNeeded = &update
	}

	err := resizeMap.updatePvSize(pvcr, pvcr.ExpectedSize)
	if err != nil {
		glog.V(4).Infof("Error updating PV spec capacity for volume %q with : %v", pvcr.QualifiedName(), err)
		return err
	}

	if pvcr, ok := resizeMap.pvcrs[pvcUniqueName]; ok {
		pvcr.UpdateNeeded = nil
	}
	return nil
}

func (resizeMap *volumeResizeMap) updatePvSize(pvcr *PvcWithResizeRequest, newSize resource.Quantity) error {
	oldPv := pvcr.VolumeSpec.PersistentVolume
	clone, err := scheme.Scheme.DeepCopy(oldPv)

	if err != nil {
		return fmt.Errorf("Error cloning PV %q with error : %v", oldPv.Name, err)
	}
	pvClone, ok := clone.(*v1.PersistentVolume)

	if !ok {
		return fmt.Errorf("Unexpected cast error for PV : %v", pvClone)
	}

	pvClone.Spec.Capacity[v1.ResourceStorage] = newSize
	_, updateErr := resizeMap.kubeClient.CoreV1().PersistentVolumes().Update(pvClone)

	if updateErr != nil {
		glog.V(4).Infof("Erro updating pv %q with error : %v", pvClone.Name, updateErr)
		return updateErr
	}
	return nil
}

func (resizeMap *volumeResizeMap) updatePvcCapacity(pvcr *PvcWithResizeRequest, newSize resource.Quantity) error {
	claimClone, err := clonePVC(pvcr.PVC)
	if err != nil {
		return err
	}

	claimClone.Status.Capacity[v1.ResourceStorage] = newSize
	_, updateErr := resizeMap.kubeClient.Core().PersistentVolumeClaims(claimClone.Namespace).UpdateStatus(claimClone)
	if updateErr != nil {
		glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: failed: %v", pvcr.QualifiedName(), updateErr)
		return updateErr
	}
	return nil
}

func (resizeMap *volumeResizeMap) updatePvcCondition(pvcr *PvcWithResizeRequest, pvcCondition v1.PvcCondition) error {
	glog.V(4).Infof("Updating PVC %s with condition % and status %s", pvcr.QualifiedName(), pvcCondition.Type, pvcCondition.Status)

	newConditions := []v1.PvcCondition{pvcCondition}

	claimClone, err := clonePVC(pvcr.PVC)
	if err != nil {
		return err
	}

	claimClone.Status.Conditions = newConditions
	_, updateErr := resizeMap.kubeClient.CoreV1().PersistentVolumeClaims(claimClone.Namespace).UpdateStatus(claimClone)
	if updateErr != nil {
		glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: failed: %v", pvcr.QualifiedName(), updateErr)
		return updateErr
	}
	return nil
}

func (resizeMap *volumeResizeMap) updatePvcCapacityAndConditions(pvcr *PvcWithResizeRequest, newSize resource.Quantity, pvcCondition v1.PvcCondition) error {
	glog.V(4).Infof("Updating PVC %s with condition % and status %s", pvcr.QualifiedName(), pvcCondition.Type, pvcCondition.Status)

	newConditions := []v1.PvcCondition{pvcCondition}
	claimClone, err := clonePVC(pvcr.PVC)
	if err != nil {
		return err
	}

	claimClone.Status.Capacity[v1.ResourceStorage] = newSize
	claimClone.Status.Conditions = newConditions
	_, updateErr := resizeMap.kubeClient.CoreV1().PersistentVolumeClaims(claimClone.Namespace).UpdateStatus(claimClone)
	if updateErr != nil {
		glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: failed: %v", pvcr.QualifiedName(), updateErr)
		return updateErr
	}
	return nil
}

func clonePVC(oldPvc *v1.PersistentVolumeClaim) (*v1.PersistentVolumeClaim, error) {
	clone, err := scheme.Scheme.DeepCopy(oldPvc)

	if err != nil {
		return nil, fmt.Errorf("Error cloning claim %s : %v", claimToClaimKey(oldPvc), err)
	}

	claimClone, ok := clone.(*v1.PersistentVolumeClaim)

	if !ok {
		return nil, fmt.Errorf("Unexpected claim cast error : %v", claimClone)
	}
	return claimClone, nil

}

func claimToClaimKey(claim *v1.PersistentVolumeClaim) string {
	return fmt.Sprintf("%s/%s", claim.Namespace, claim.Name)
}
