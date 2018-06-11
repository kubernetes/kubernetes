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

package snapshot

import (
	"fmt"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"strconv"
)

var (
	keyFunc = cache.DeletionHandlingMetaNamespaceKeyFunc
)

func SnapshotToSnapshotKey(snapshot *storage.VolumeSnapshot) string {
	return fmt.Sprintf("%s/%s", snapshot.Namespace, snapshot.Name)
}

func SnapshotRefToSnapshotKey(snapshotref *v1.ObjectReference) string {
	return fmt.Sprintf("%s/%s", snapshotref.Namespace, snapshotref.Name)
}

// isSnapshotDataBoundToSnapshot returns true, if given SnapshotData is pre-bound or bound
// to specific Snapshot. Both claim.Name and claim.Namespace must be equal.
// If claim.UID is present in volume.Spec.ClaimRef, it must be equal too.
func isSnapshotDataBoundToSnapshot(snapshotData *storage.VolumeSnapshotData, snapshot *storage.VolumeSnapshot) bool {
	if snapshotData.Spec.VolumeSnapshotRef == nil {
		return false
	}
	if snapshot.Name != snapshotData.Spec.VolumeSnapshotRef.Name || snapshot.Namespace != snapshotData.Spec.VolumeSnapshotRef.Namespace {
		return false
	}
	if snapshotData.Spec.VolumeSnapshotRef.UID != "" && snapshot.UID != snapshotData.Spec.VolumeSnapshotRef.UID {
		return false
	}
	return true
}

// IsSnapshotReady returns true if a VolumeSnapshot is ready; false otherwise.
func IsSnapshotReady(volumeSnapshot *storage.VolumeSnapshot) bool {
	return IsSnapshotReadyConditionTrue(volumeSnapshot.Status)
}

// IsVolumeSnapshotReadyConditionTrue returns true if a VolumeSnapshot is ready; false otherwise.
func IsSnapshotReadyConditionTrue(status storage.VolumeSnapshotStatus) bool {
	condition := GetSnapshotReadyCondition(status)
	return condition != nil && condition.Status == v1.ConditionTrue
}

// GetSnapshotReadyCondition extracts the VolumeSnapshot ready condition from the given status and returns that.
// Returns nil if the condition is not present.
func GetSnapshotReadyCondition(status storage.VolumeSnapshotStatus) *storage.VolumeSnapshotCondition {
	_, condition := GetSnapshotCondition(&status, storage.VolumeSnapshotConditionReady)
	return condition
}

// IsSnapshotDataReady returns true if a volumeSnapshotData is ready; false otherwise.
func IsSnapshotDataReady(volumeSnapshotData *storage.VolumeSnapshotData) bool {
	return IsSnapshotDataReadyConditionTrue(volumeSnapshotData.Status)
}

// IsSnapshotDataReadyConditionTrue returns true if a VolumeSnapshot is ready; false otherwise.
func IsSnapshotDataReadyConditionTrue(status storage.VolumeSnapshotDataStatus) bool {
	condition := GetSnapshotDataReadyCondition(status)
	return condition != nil && condition.Status == v1.ConditionTrue
}

// GetSnapshotReadyCondition extracts the VolumeSnapshot ready condition from the given status and returns that.
// Returns nil if the condition is not present.
func GetSnapshotDataReadyCondition(status storage.VolumeSnapshotDataStatus) *storage.VolumeSnapshotDataCondition {
	_, condition := GetSnapshotDataCondition(&status, storage.VolumeSnapshotDataConditionReady)
	return condition
}

// GetSnapshotCondition extracts the provided condition from the given status and returns that.
// Returns nil and -1 if the condition is not present, and the index of the located condition.
func GetSnapshotDataCondition(status *storage.VolumeSnapshotDataStatus, conditionType storage.VolumeSnapshotDataConditionType) (int, *storage.VolumeSnapshotDataCondition) {
	if status == nil {
		return -1, nil
	}
	for i := range status.Conditions {
		if status.Conditions[i].Type == conditionType {
			return i, &status.Conditions[i]
		}
	}
	return -1, nil
}

// GetSnapshotCondition extracts the provided condition from the given status and returns that.
// Returns nil and -1 if the condition is not present, and the index of the located condition.
func GetSnapshotCondition(status *storage.VolumeSnapshotStatus, conditionType storage.VolumeSnapshotConditionType) (int, *storage.VolumeSnapshotCondition) {
	if status == nil {
		return -1, nil
	}
	for i := range status.Conditions {
		if status.Conditions[i].Type == conditionType {
			return i, &status.Conditions[i]
		}
	}
	return -1, nil
}

// UpdateSnapshotCondition updates existing Snapshot condition or creates a new one. Sets LastTransitionTime to now if the
// status has changed.
// Returns true if Snapshot condition has changed or has been added.
func UpdateSnapshotCondition(status *storage.VolumeSnapshotStatus, condition *storage.VolumeSnapshotCondition) bool {
	condition.LastTransitionTime = metav1.Now()
	// Try to find this pod condition.
	conditionIndex, oldCondition := GetSnapshotCondition(status, condition.Type)

	if oldCondition == nil {
		// We are adding new pod condition.
		status.Conditions = append(status.Conditions, *condition)
		return true
	}
	// We are updating an existing condition, so we need to check if it has changed.
	if condition.Status == oldCondition.Status {
		condition.LastTransitionTime = oldCondition.LastTransitionTime
	}

	isEqual := condition.Status == oldCondition.Status &&
		condition.Reason == oldCondition.Reason &&
		condition.Message == oldCondition.Message &&
		condition.LastTransitionTime.Equal(&oldCondition.LastTransitionTime)

	status.Conditions[conditionIndex] = *condition
	// Return true if one of the fields have changed.
	return !isEqual
}

// UpdateSnapshotDataCondition updates existing Snapshot data condition or creates a new one. Sets LastTransitionTime to now if the
// status has changed.
// Returns true if Snapshot data condition has changed or has been added.
func UpdateSnapshotDataCondition(status *storage.VolumeSnapshotDataStatus, condition *storage.VolumeSnapshotDataCondition) bool {
	condition.LastTransitionTime = metav1.Now()
	// Try to find this pod condition.
	conditionIndex, oldCondition := GetSnapshotDataCondition(status, condition.Type)

	if oldCondition == nil {
		// We are adding new pod condition.
		status.Conditions = append(status.Conditions, *condition)
		return true
	}
	// We are updating an existing condition, so we need to check if it has changed.
	if condition.Status == oldCondition.Status {
		condition.LastTransitionTime = oldCondition.LastTransitionTime
	}

	isEqual := condition.Status == oldCondition.Status &&
		condition.Reason == oldCondition.Reason &&
		condition.Message == oldCondition.Message &&
		condition.LastTransitionTime.Equal(&oldCondition.LastTransitionTime)

	status.Conditions[conditionIndex] = *condition
	// Return true if one of the fields have changed.
	return !isEqual
}

// TransformDataCondition2SnapshotCondition transform snapshotData condition to snapshot condition.
func TransformDataCondition2SnapshotCondition(snapshotDataCon *storage.VolumeSnapshotDataCondition) *storage.VolumeSnapshotCondition {
	if snapshotDataCon == nil {
		return nil
	}
	return &storage.VolumeSnapshotCondition{
		Type:               (storage.VolumeSnapshotConditionType)(snapshotDataCon.Type),
		Status:             snapshotDataCon.Status,
		Message:            snapshotDataCon.Message,
		LastTransitionTime: snapshotDataCon.LastTransitionTime,
		Reason:             snapshotDataCon.Reason,
	}
}

// storeObjectUpdate updates given cache with a new object version from Informer
// callback (i.e. with events from etcd) or with an object modified by the
// controller itself. Returns "true", if the cache was updated, false if the
// object is an old version and should be ignored.
func storeObjectUpdate(store cache.Store, obj interface{}, className string) (bool, error) {
	objName, err := keyFunc(obj)
	if err != nil {
		return false, fmt.Errorf("Couldn't get key for object %+v: %v", obj, err)
	}
	oldObj, found, err := store.Get(obj)
	if err != nil {
		return false, fmt.Errorf("Error finding %s %q in controller cache: %v", className, objName, err)
	}

	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
	}

	if !found {
		// This is a new object
		glog.V(4).Infof("storeObjectUpdate: adding %s %q, version %s", className, objName, objAccessor.GetResourceVersion())
		if err = store.Add(obj); err != nil {
			return false, fmt.Errorf("error adding %s %q to controller cache: %v", className, objName, err)
		}
		return true, nil
	}

	oldObjAccessor, err := meta.Accessor(oldObj)
	if err != nil {
		return false, err
	}

	objResourceVersion, err := strconv.ParseInt(objAccessor.GetResourceVersion(), 10, 64)
	if err != nil {
		return false, fmt.Errorf("error parsing ResourceVersion %q of %s %q: %s", objAccessor.GetResourceVersion(), className, objName, err)
	}
	oldObjResourceVersion, err := strconv.ParseInt(oldObjAccessor.GetResourceVersion(), 10, 64)
	if err != nil {
		return false, fmt.Errorf("error parsing old ResourceVersion %q of %s %q: %s", oldObjAccessor.GetResourceVersion(), className, objName, err)
	}

	// Throw away only older version, let the same version pass - we do want to
	// get periodic sync events.
	if oldObjResourceVersion > objResourceVersion {
		glog.V(4).Infof("storeObjectUpdate: ignoring %s %q version %s", className, objName, objAccessor.GetResourceVersion())
		return false, nil
	}

	glog.V(4).Infof("storeObjectUpdate updating %s %q with version %s", className, objName, objAccessor.GetResourceVersion())
	if err = store.Update(obj); err != nil {
		return false, fmt.Errorf("error updating %s %q in controller cache: %v", className, objName, err)
	}
	return true, nil
}

// Helper function to get PV from VolumeSnapshot
func GetVolumeFromSnapshotData(snapshotData *storage.VolumeSnapshotData, client clientset.Interface) (*v1.PersistentVolume, error) {
	if snapshotData.Spec.PersistentVolumeRef == nil || snapshotData.Spec.PersistentVolumeRef.Name == "" {
		return nil, fmt.Errorf("the PV name is not specified in snapshotData %s", snapshotData.Name)
	}
	pvName := snapshotData.Spec.PersistentVolumeRef.Name
	pv, err := client.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve PV %s from the API server: %q", pvName, err)
	}
	return pv, nil
}

// getSnapshotDataNameForSnapshot returns SnapshotData.Name for the create VolumeSnapshotData.
// The name must be unique.
func GetSnapshotDataNameForSnapshot(snapshot *storage.VolumeSnapshot) string {
	return "vsd-" + string(snapshot.UID)
}
