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
	"encoding/json"
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/resizefs"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

var (
	knownResizeConditions = map[v1.PersistentVolumeClaimConditionType]bool{
		v1.PersistentVolumeClaimFileSystemResizePending: true,
		v1.PersistentVolumeClaimResizing:                true,
	}
)

type resizeProcessStatus struct {
	condition v1.PersistentVolumeClaimCondition
	processed bool
}

// ClaimToClaimKey return namespace/name string for pvc
func ClaimToClaimKey(claim *v1.PersistentVolumeClaim) string {
	return fmt.Sprintf("%s/%s", claim.Namespace, claim.Name)
}

// UpdatePVSize updates just pv size after cloudprovider resizing is successful
func UpdatePVSize(
	pv *v1.PersistentVolume,
	newSize resource.Quantity,
	kubeClient clientset.Interface) error {
	pvClone := pv.DeepCopy()

	oldData, err := json.Marshal(pvClone)
	if err != nil {
		return fmt.Errorf("unexpected error marshaling old PV %q with error : %v", pvClone.Name, err)
	}

	pvClone.Spec.Capacity[v1.ResourceStorage] = newSize

	newData, err := json.Marshal(pvClone)
	if err != nil {
		return fmt.Errorf("unexpected error marshaling new PV %q with error : %v", pvClone.Name, err)
	}

	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, pvClone)
	if err != nil {
		return fmt.Errorf("error Creating two way merge patch for PV %q with error : %v", pvClone.Name, err)
	}

	_, err = kubeClient.CoreV1().PersistentVolumes().Patch(pvClone.Name, types.StrategicMergePatchType, patchBytes)
	if err != nil {
		return fmt.Errorf("error Patching PV %q with error : %v", pvClone.Name, err)
	}
	return nil
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
	newPVC = MergeResizeConditionOnPVC(newPVC, conditions)
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
	kubeClient clientset.Interface) error {
	pvcCondition := v1.PersistentVolumeClaimCondition{
		Type:               v1.PersistentVolumeClaimFileSystemResizePending,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Message:            "Waiting for user to (re-)start a pod to finish file system resize of volume on node.",
	}
	conditions := []v1.PersistentVolumeClaimCondition{pvcCondition}
	newPVC := pvc.DeepCopy()
	newPVC = MergeResizeConditionOnPVC(newPVC, conditions)
	_, err := PatchPVCStatus(pvc /*oldPVC*/, newPVC, kubeClient)
	return err
}

// MarkResizeFinished marks all resizing as done
func MarkResizeFinished(
	pvc *v1.PersistentVolumeClaim,
	newSize resource.Quantity,
	kubeClient clientset.Interface) error {
	return MarkFSResizeFinished(pvc, newSize, kubeClient)
}

// MarkFSResizeFinished marks file system resizing as done
func MarkFSResizeFinished(
	pvc *v1.PersistentVolumeClaim,
	newSize resource.Quantity,
	kubeClient clientset.Interface) error {
	newPVC := pvc.DeepCopy()
	newPVC.Status.Capacity[v1.ResourceStorage] = newSize
	newPVC = MergeResizeConditionOnPVC(newPVC, []v1.PersistentVolumeClaimCondition{})
	_, err := PatchPVCStatus(pvc /*oldPVC*/, newPVC, kubeClient)
	return err
}

// PatchPVCStatus updates PVC status using PATCH verb
// Don't use Update because this can be called from kubelet and if kubelet has an older client its
// Updates will overwrite new fields. And to avoid writing to a stale object, add ResourceVersion
// to the patch so that Patch will fail if the patch's RV != actual up-to-date RV like Update would
func PatchPVCStatus(
	oldPVC *v1.PersistentVolumeClaim,
	newPVC *v1.PersistentVolumeClaim,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	patchBytes, err := createPVCPatch(oldPVC, newPVC)
	if err != nil {
		return nil, fmt.Errorf("patchPVCStatus failed to patch PVC %q: %v", oldPVC.Name, err)
	}

	updatedClaim, updateErr := kubeClient.CoreV1().PersistentVolumeClaims(oldPVC.Namespace).
		Patch(oldPVC.Name, types.StrategicMergePatchType, patchBytes, "status")
	if updateErr != nil {
		return nil, fmt.Errorf("patchPVCStatus failed to patch PVC %q: %v", oldPVC.Name, updateErr)
	}
	return updatedClaim, nil
}

func createPVCPatch(
	oldPVC *v1.PersistentVolumeClaim,
	newPVC *v1.PersistentVolumeClaim) ([]byte, error) {
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

	patchBytes, err = addResourceVersion(patchBytes, oldPVC.ResourceVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to add resource version: %v", err)
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
	resizeConditions []v1.PersistentVolumeClaimCondition) *v1.PersistentVolumeClaim {
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

// GenericResizeFS : call generic filesystem resizer for plugins that don't have any special filesystem resize requirements
func GenericResizeFS(host volume.VolumeHost, pluginName, devicePath, deviceMountPath string) (bool, error) {
	mounter := host.GetMounter(pluginName)
	diskFormatter := &mount.SafeFormatAndMount{
		Interface: mounter,
		Exec:      host.GetExec(pluginName),
	}
	resizer := resizefs.NewResizeFs(diskFormatter)
	return resizer.Resize(devicePath, deviceMountPath)
}
