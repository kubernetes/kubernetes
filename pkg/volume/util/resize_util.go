/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"context"
	"encoding/json"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/mount-utils"
	"k8s.io/utils/exec"
)

var (
	knownResizeConditions = map[v1.PersistentVolumeClaimConditionType]bool{
		v1.PersistentVolumeClaimFileSystemResizePending: true,
		v1.PersistentVolumeClaimResizing:                true,
		v1.PersistentVolumeClaimControllerResizeError:   true,
		v1.PersistentVolumeClaimNodeResizeError:         true,
	}

	// AnnPreResizeCapacity annotation is added to a PV when expanding volume.
	// Its value is status capacity of the PVC prior to the volume expansion
	// Its value will be set by the external-resizer when it deems that filesystem resize is required after resizing volume.
	// Its value will be used by pv_controller to determine pvc's status capacity when binding pvc and pv.
	AnnPreResizeCapacity = "volume.alpha.kubernetes.io/pre-resize-capacity"
)

type resizeProcessStatus struct {
	condition v1.PersistentVolumeClaimCondition
	processed bool
}

// UpdatePVSize updates just pv size after cloudprovider resizing is successful
func UpdatePVSize(
	pv *v1.PersistentVolume,
	newSize resource.Quantity,
	kubeClient clientset.Interface) (*v1.PersistentVolume, error) {
	pvClone := pv.DeepCopy()
	pvClone.Spec.Capacity[v1.ResourceStorage] = newSize

	return PatchPV(pv, pvClone, kubeClient)
}

// AddAnnPreResizeCapacity adds volume.alpha.kubernetes.io/pre-resize-capacity from the pv
func AddAnnPreResizeCapacity(
	pv *v1.PersistentVolume,
	oldCapacity resource.Quantity,
	kubeClient clientset.Interface) error {
	// if the pv already has a resize annotation skip the process
	if metav1.HasAnnotation(pv.ObjectMeta, AnnPreResizeCapacity) {
		return nil
	}

	pvClone := pv.DeepCopy()
	if pvClone.ObjectMeta.Annotations == nil {
		pvClone.ObjectMeta.Annotations = make(map[string]string)
	}
	pvClone.ObjectMeta.Annotations[AnnPreResizeCapacity] = oldCapacity.String()

	_, err := PatchPV(pv, pvClone, kubeClient)
	return err
}

// DeleteAnnPreResizeCapacity deletes volume.alpha.kubernetes.io/pre-resize-capacity from the pv
func DeleteAnnPreResizeCapacity(
	pv *v1.PersistentVolume,
	kubeClient clientset.Interface) error {
	// if the pv does not have a resize annotation skip the entire process
	if !metav1.HasAnnotation(pv.ObjectMeta, AnnPreResizeCapacity) {
		return nil
	}
	pvClone := pv.DeepCopy()
	delete(pvClone.ObjectMeta.Annotations, AnnPreResizeCapacity)
	_, err := PatchPV(pv, pvClone, kubeClient)
	return err
}

// PatchPV creates and executes a patch for pv
func PatchPV(
	oldPV *v1.PersistentVolume,
	newPV *v1.PersistentVolume,
	kubeClient clientset.Interface) (*v1.PersistentVolume, error) {
	oldData, err := json.Marshal(oldPV)
	if err != nil {
		return oldPV, fmt.Errorf("unexpected error marshaling old PV %q with error : %v", oldPV.Name, err)
	}

	newData, err := json.Marshal(newPV)
	if err != nil {
		return oldPV, fmt.Errorf("unexpected error marshaling new PV %q with error : %v", newPV.Name, err)
	}

	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, oldPV)
	if err != nil {
		return oldPV, fmt.Errorf("error Creating two way merge patch for PV %q with error : %v", oldPV.Name, err)
	}

	updatedPV, err := kubeClient.CoreV1().PersistentVolumes().Patch(context.TODO(), oldPV.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	if err != nil {
		return oldPV, fmt.Errorf("error Patching PV %q with error : %v", oldPV.Name, err)
	}
	return updatedPV, nil
}

// MarkResizeInProgressWithResizer marks cloudprovider resizing as in progress
// and also annotates the PVC with the name of the resizer.
func MarkResizeInProgressWithResizer(
	pvc *v1.PersistentVolumeClaim,
	resizerName string,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	// Mark PVC as Resize Started
	progressCondition := v1.PersistentVolumeClaimCondition{
		Type:               v1.PersistentVolumeClaimResizing,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
	}
	conditions := []v1.PersistentVolumeClaimCondition{progressCondition}
	newPVC := pvc.DeepCopy()
	newPVC = MergeResizeConditionOnPVC(newPVC, conditions, false /* keepOldResizeConditions */)
	newPVC = setResizer(newPVC, resizerName)
	return PatchPVCStatus(pvc /*oldPVC*/, newPVC, kubeClient)
}

func MarkControllerReisizeInProgress(pvc *v1.PersistentVolumeClaim, resizerName string, newSize resource.Quantity, kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	// Mark PVC as Resize Started
	progressCondition := v1.PersistentVolumeClaimCondition{
		Type:               v1.PersistentVolumeClaimResizing,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
	}
	conditions := []v1.PersistentVolumeClaimCondition{progressCondition}
	newPVC := pvc.DeepCopy()
	newPVC = MergeResizeConditionOnPVC(newPVC, conditions, false /* keepOldResizeConditions */)
	newPVC = mergeStorageResourceStatus(newPVC, v1.PersistentVolumeClaimControllerResizeInProgress)
	newPVC = mergeStorageAllocatedResources(newPVC, newSize)
	newPVC = setResizer(newPVC, resizerName)
	return PatchPVCStatus(pvc /*oldPVC*/, newPVC, kubeClient)
}

// SetClaimResizer sets resizer annotation on PVC
func SetClaimResizer(
	pvc *v1.PersistentVolumeClaim,
	resizerName string,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	newPVC := pvc.DeepCopy()
	newPVC = setResizer(newPVC, resizerName)
	return PatchPVCStatus(pvc /*oldPVC*/, newPVC, kubeClient)
}

func setResizer(pvc *v1.PersistentVolumeClaim, resizerName string) *v1.PersistentVolumeClaim {
	if val, ok := pvc.Annotations[volumetypes.VolumeResizerKey]; ok && val == resizerName {
		return pvc
	}
	metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, volumetypes.VolumeResizerKey, resizerName)
	return pvc
}

// MarkForFSResize marks file system resizing as pending
func MarkForFSResize(
	pvc *v1.PersistentVolumeClaim,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	pvcCondition := v1.PersistentVolumeClaimCondition{
		Type:               v1.PersistentVolumeClaimFileSystemResizePending,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Message:            "Waiting for user to (re-)start a pod to finish file system resize of volume on node.",
	}
	conditions := []v1.PersistentVolumeClaimCondition{pvcCondition}
	newPVC := pvc.DeepCopy()

	if utilfeature.DefaultFeatureGate.Enabled(features.RecoverVolumeExpansionFailure) {
		newPVC = mergeStorageResourceStatus(newPVC, v1.PersistentVolumeClaimNodeResizePending)
	}

	newPVC = MergeResizeConditionOnPVC(newPVC, conditions, true /* keepOldResizeConditions */)
	updatedPVC, err := PatchPVCStatus(pvc /*oldPVC*/, newPVC, kubeClient)
	return updatedPVC, err
}

// MarkResizeFinished marks all resizing as done
func MarkResizeFinished(
	pvc *v1.PersistentVolumeClaim,
	newSize resource.Quantity,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	return MarkFSResizeFinished(pvc, newSize, kubeClient)
}

// MarkFSResizeFinished marks file system resizing as done
func MarkFSResizeFinished(
	pvc *v1.PersistentVolumeClaim,
	newSize resource.Quantity,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	newPVC := pvc.DeepCopy()

	newPVC.Status.Capacity[v1.ResourceStorage] = newSize

	// if RecoverVolumeExpansionFailure is enabled, we need to reset ResizeStatus back to nil
	if utilfeature.DefaultFeatureGate.Enabled(features.RecoverVolumeExpansionFailure) {
		allocatedResourceStatusMap := newPVC.Status.AllocatedResourceStatuses
		delete(allocatedResourceStatusMap, v1.ResourceStorage)
		if len(allocatedResourceStatusMap) == 0 {
			newPVC.Status.AllocatedResourceStatuses = nil
		} else {
			newPVC.Status.AllocatedResourceStatuses = allocatedResourceStatusMap
		}
	}

	newPVC = MergeResizeConditionOnPVC(newPVC, []v1.PersistentVolumeClaimCondition{}, false /* keepOldResizeConditions */)
	updatedPVC, err := PatchPVCStatus(pvc /*oldPVC*/, newPVC, kubeClient)
	return updatedPVC, err
}

func MarkNodeExpansionFinishedWithRecovery(
	pvc *v1.PersistentVolumeClaim,
	newSize resource.Quantity,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	newPVC := pvc.DeepCopy()

	newPVC.Status.Capacity[v1.ResourceStorage] = newSize

	allocatedResourceStatusMap := newPVC.Status.AllocatedResourceStatuses
	delete(allocatedResourceStatusMap, v1.ResourceStorage)
	if len(allocatedResourceStatusMap) == 0 {
		newPVC.Status.AllocatedResourceStatuses = nil
	} else {
		newPVC.Status.AllocatedResourceStatuses = allocatedResourceStatusMap
	}

	newPVC = MergeResizeConditionOnPVC(newPVC, []v1.PersistentVolumeClaimCondition{}, false /* keepOldResizeConditions */)
	updatedPVC, err := PatchPVCStatus(pvc /*oldPVC*/, newPVC, kubeClient)
	return updatedPVC, err
}

// MarkNodeExpansionInfeasible marks a PVC for node expansion as failed. Kubelet should not retry expansion
// of volumes which are in failed state.
func MarkNodeExpansionInfeasible(pvc *v1.PersistentVolumeClaim, kubeClient clientset.Interface, err error) (*v1.PersistentVolumeClaim, error) {
	newPVC := pvc.DeepCopy()
	newPVC = mergeStorageResourceStatus(newPVC, v1.PersistentVolumeClaimNodeResizeInfeasible)
	errorCondition := v1.PersistentVolumeClaimCondition{
		Type:               v1.PersistentVolumeClaimNodeResizeError,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Message:            fmt.Sprintf("failed to expand pvc with %v", err),
	}
	newPVC = MergeResizeConditionOnPVC(newPVC,
		[]v1.PersistentVolumeClaimCondition{errorCondition},
		true /* keepOldResizeConditions */)

	patchBytes, err := createPVCPatch(pvc, newPVC, false /* addResourceVersionCheck */)
	if err != nil {
		return pvc, fmt.Errorf("patchPVCStatus failed to patch PVC %q: %v", pvc.Name, err)
	}

	updatedClaim, updateErr := kubeClient.CoreV1().PersistentVolumeClaims(pvc.Namespace).
		Patch(context.TODO(), pvc.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
	if updateErr != nil {
		return pvc, fmt.Errorf("patchPVCStatus failed to patch PVC %q: %v", pvc.Name, updateErr)
	}
	return updatedClaim, nil
}

func MarkNodeExpansionFailedCondition(pvc *v1.PersistentVolumeClaim, kubeClient clientset.Interface, err error) (*v1.PersistentVolumeClaim, error) {
	newPVC := pvc.DeepCopy()
	errorCondition := v1.PersistentVolumeClaimCondition{
		Type:               v1.PersistentVolumeClaimNodeResizeError,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Message:            fmt.Sprintf("failed to expand pvc with %v", err),
	}
	newPVC = MergeResizeConditionOnPVC(newPVC,
		[]v1.PersistentVolumeClaimCondition{errorCondition},
		true /* keepOldResizeConditions */)
	patchBytes, err := createPVCPatch(pvc, newPVC, false /* addResourceVersionCheck */)
	if err != nil {
		return pvc, fmt.Errorf("patchPVCStatus failed to patch PVC %q: %w", pvc.Name, err)
	}

	updatedClaim, updateErr := kubeClient.CoreV1().PersistentVolumeClaims(pvc.Namespace).
		Patch(context.TODO(), pvc.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
	if updateErr != nil {
		return pvc, fmt.Errorf("patchPVCStatus failed to patch PVC %q: %w", pvc.Name, updateErr)
	}
	return updatedClaim, nil
}

// MarkNodeExpansionInProgress marks pvc expansion in progress on node
func MarkNodeExpansionInProgress(pvc *v1.PersistentVolumeClaim, kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	newPVC := pvc.DeepCopy()
	newPVC = mergeStorageResourceStatus(newPVC, v1.PersistentVolumeClaimNodeResizeInProgress)
	updatedPVC, err := PatchPVCStatus(pvc /* oldPVC */, newPVC, kubeClient)
	return updatedPVC, err
}

// PatchPVCStatus updates PVC status using PATCH verb
// Don't use Update because this can be called from kubelet and if kubelet has an older client its
// Updates will overwrite new fields. And to avoid writing to a stale object, add ResourceVersion
// to the patch so that Patch will fail if the patch's RV != actual up-to-date RV like Update would
func PatchPVCStatus(
	oldPVC *v1.PersistentVolumeClaim,
	newPVC *v1.PersistentVolumeClaim,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	patchBytes, err := createPVCPatch(oldPVC, newPVC, true /* addResourceVersionCheck */)
	if err != nil {
		return oldPVC, fmt.Errorf("patchPVCStatus failed to patch PVC %q: %v", oldPVC.Name, err)
	}

	updatedClaim, updateErr := kubeClient.CoreV1().PersistentVolumeClaims(oldPVC.Namespace).
		Patch(context.TODO(), oldPVC.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
	if updateErr != nil {
		return oldPVC, fmt.Errorf("patchPVCStatus failed to patch PVC %q: %v", oldPVC.Name, updateErr)
	}
	return updatedClaim, nil
}

func createPVCPatch(
	oldPVC *v1.PersistentVolumeClaim,
	newPVC *v1.PersistentVolumeClaim, addResourceVersionCheck bool) ([]byte, error) {
	oldData, err := json.Marshal(oldPVC)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal old data: %v", err)
	}

	newData, err := json.Marshal(newPVC)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal new data: %v", err)
	}

	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, oldPVC)
	if err != nil {
		return nil, fmt.Errorf("failed to create 2 way merge patch: %v", err)
	}

	if addResourceVersionCheck {
		patchBytes, err = addResourceVersion(patchBytes, oldPVC.ResourceVersion)
		if err != nil {
			return nil, fmt.Errorf("failed to add resource version: %v", err)
		}
	}

	return patchBytes, nil
}

func addResourceVersion(patchBytes []byte, resourceVersion string) ([]byte, error) {
	var patchMap map[string]interface{}
	err := json.Unmarshal(patchBytes, &patchMap)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling patch: %v", err)
	}
	u := unstructured.Unstructured{Object: patchMap}
	a, err := meta.Accessor(&u)
	if err != nil {
		return nil, fmt.Errorf("error creating accessor: %v", err)
	}
	a.SetResourceVersion(resourceVersion)
	versionBytes, err := json.Marshal(patchMap)
	if err != nil {
		return nil, fmt.Errorf("error marshalling json patch: %v", err)
	}
	return versionBytes, nil
}

// MergeResizeConditionOnPVC updates pvc with requested resize conditions
// leaving other conditions untouched.
func MergeResizeConditionOnPVC(
	pvc *v1.PersistentVolumeClaim,
	resizeConditions []v1.PersistentVolumeClaimCondition, keepOldResizeConditions bool) *v1.PersistentVolumeClaim {
	resizeConditionMap := map[v1.PersistentVolumeClaimConditionType]*resizeProcessStatus{}

	for _, condition := range resizeConditions {
		resizeConditionMap[condition.Type] = &resizeProcessStatus{condition, false}
	}

	oldConditions := pvc.Status.Conditions
	newConditions := []v1.PersistentVolumeClaimCondition{}
	for _, condition := range oldConditions {
		// If Condition is of not resize type, we keep it.
		if _, ok := knownResizeConditions[condition.Type]; !ok {
			newConditions = append(newConditions, condition)
			continue
		}

		if newCondition, ok := resizeConditionMap[condition.Type]; ok {
			if newCondition.condition.Status != condition.Status {
				newConditions = append(newConditions, newCondition.condition)
			} else {
				newConditions = append(newConditions, condition)
			}
			newCondition.processed = true
		} else if keepOldResizeConditions {
			// if keepOldResizeConditions is true, we keep the old resize conditions that were present in the
			// existing pvc.Status.Conditions field.
			newConditions = append(newConditions, condition)
		}
	}

	// append all unprocessed conditions
	for _, newCondition := range resizeConditionMap {
		if !newCondition.processed {
			newConditions = append(newConditions, newCondition.condition)
		}
	}
	pvc.Status.Conditions = newConditions
	return pvc
}

func mergeStorageResourceStatus(pvc *v1.PersistentVolumeClaim, status v1.ClaimResourceStatus) *v1.PersistentVolumeClaim {
	allocatedResourceStatusMap := pvc.Status.AllocatedResourceStatuses
	if allocatedResourceStatusMap == nil {
		pvc.Status.AllocatedResourceStatuses = map[v1.ResourceName]v1.ClaimResourceStatus{
			v1.ResourceStorage: status,
		}
		return pvc
	}
	allocatedResourceStatusMap[v1.ResourceStorage] = status
	pvc.Status.AllocatedResourceStatuses = allocatedResourceStatusMap
	return pvc
}

func mergeStorageAllocatedResources(pvc *v1.PersistentVolumeClaim, size resource.Quantity) *v1.PersistentVolumeClaim {
	allocatedResourcesMap := pvc.Status.AllocatedResources
	if allocatedResourcesMap == nil {
		pvc.Status.AllocatedResources = map[v1.ResourceName]resource.Quantity{
			v1.ResourceStorage: size,
		}
		return pvc
	}
	allocatedResourcesMap[v1.ResourceStorage] = size
	pvc.Status.AllocatedResources = allocatedResourcesMap
	return pvc
}

// GenericResizeFS : call generic filesystem resizer for plugins that don't have any special filesystem resize requirements
func GenericResizeFS(host volume.VolumeHost, devicePath, deviceMountPath string) (bool, error) {
	resizer := mount.NewResizeFs(exec.New())
	return resizer.Resize(devicePath, deviceMountPath)
}
