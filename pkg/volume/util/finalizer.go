/*
Copyright 2017 The Kubernetes Authors.

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
	"k8s.io/api/core/v1"
)

const (
	// Name of finalizer on PVCs that have a running pod.
	PVCProtectionFinalizer = "kubernetes.io/pvc-protection"

	// Name of finalizer on PVs that are bound by PVCs
	PVProtectionFinalizer = "kubernetes.io/pv-protection"
)

// IsPVCBeingDeleted returns:
// true: in case PVC is being deleted, i.e. ObjectMeta.DeletionTimestamp is set
// false: in case PVC is not being deleted, i.e. ObjectMeta.DeletionTimestamp is nil
func IsPVCBeingDeleted(pvc *v1.PersistentVolumeClaim) bool {
	return pvc.ObjectMeta.DeletionTimestamp != nil
}

// IsProtectionFinalizerPresent returns true in case PVCProtectionFinalizer is
// present among the pvc.Finalizers
func IsProtectionFinalizerPresent(pvc *v1.PersistentVolumeClaim) bool {
	for _, finalizer := range pvc.Finalizers {
		if finalizer == PVCProtectionFinalizer {
			return true
		}
	}
	return false
}

// RemoveProtectionFinalizer returns pvc without PVCProtectionFinalizer in case
// it's present in pvc.Finalizers. It expects that pvc is writable (i.e. is not
// informer's cached copy.)
func RemoveProtectionFinalizer(pvc *v1.PersistentVolumeClaim) {
	newFinalizers := make([]string, 0)
	for _, finalizer := range pvc.Finalizers {
		if finalizer != PVCProtectionFinalizer {
			newFinalizers = append(newFinalizers, finalizer)
		}
	}
	if len(newFinalizers) == 0 {
		// Sanitize for unit tests so we don't need to distinguish empty array
		// and nil.
		newFinalizers = nil
	}
	pvc.Finalizers = newFinalizers
}

// AddProtectionFinalizer adds PVCProtectionFinalizer to pvc. It expects that
// pvc is writable (i.e. is not informer's cached copy.)
func AddProtectionFinalizer(pvc *v1.PersistentVolumeClaim) {
	pvc.Finalizers = append(pvc.Finalizers, PVCProtectionFinalizer)
}
