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

package util

import (
	"k8s.io/kubernetes/pkg/api"
)

// IsDefaultStorageClassAnnotation represents a StorageClass annotation that
// marks a class as the default StorageClass
//TODO: Update IsDefaultStorageClassannotation and remove Beta when no longer used
const IsDefaultStorageClassAnnotation = "storageclass.beta.kubernetes.io/is-default-class"
const BetaIsDefaultStorageClassAnnotation = "storageclass.beta.kubernetes.io/is-default-class"

// AlphaStorageClassAnnotation represents the previous alpha storage class
// annotation.  it's no longer used and held here for posterity.
const AlphaStorageClassAnnotation = "volume.alpha.kubernetes.io/storage-class"

// BetaStorageClassAnnotation represents the beta/previous StorageClass annotation.
// It's currently still used and will be held for backwards compatibility
const BetaStorageClassAnnotation = "volume.beta.kubernetes.io/storage-class"

// StorageClassAnnotation represents the storage class associated with a resource.
// It currently matches the Beta value and can change when official is set.
// - in PersistentVolumeClaim it represents required class to match.
//   Only PersistentVolumes with the same class (i.e. annotation with the same
//   value) can be bound to the claim. In case no such volume exists, the
//   controller will provision a new one using StorageClass instance with
//   the same name as the annotation value.
// - in PersistentVolume it represents storage class to which the persistent
//   volume belongs.
//TODO: Update this to final annotation value as it matches BetaStorageClassAnnotation for now
const StorageClassAnnotation = "volume.beta.kubernetes.io/storage-class"

// GetVolumeStorageClass returns value of StorageClassAnnotation or empty string in case
// the annotation does not exist.
// TODO: change to PersistentVolume.Spec.Class value when this attribute is
// introduced.
func GetVolumeStorageClass(volume *api.PersistentVolume) string {
	if class, found := volume.Annotations[StorageClassAnnotation]; found {
		return class
	}

	// 'nil' is interpreted as "", i.e. the volume does not belong to any class.
	return ""
}

// GetClaimStorageClass returns name of class that is requested by given claim.
// Request for `nil` class is interpreted as request for class "",
// i.e. for a classless PV.
// TODO: change to PersistentVolumeClaim.Spec.Class value when this
// attribute is introduced.
func GetClaimStorageClass(claim *api.PersistentVolumeClaim) string {
	if class, found := claim.Annotations[StorageClassAnnotation]; found {
		return class
	}

	return ""
}

// GetStorageClassAnnotation returns the StorageClass value
// if the annotation is set, empty string if not
// TODO: remove Alpha and Beta when no longer used or needed
func GetStorageClassAnnotation(obj api.ObjectMeta) string {
	if class, ok := obj.Annotations[StorageClassAnnotation]; ok {
		return class
	}
	if class, ok := obj.Annotations[BetaStorageClassAnnotation]; ok {
		return class
	}
	if class, ok := obj.Annotations[AlphaStorageClassAnnotation]; ok {
		return class
	}

	return ""
}

// HasStorageClassAnnotation returns a boolean
// if the annotation is set
// TODO: remove Alpha and Beta when no longer used or needed
func HasStorageClassAnnotation(obj api.ObjectMeta) bool {
	if _, found := obj.Annotations[StorageClassAnnotation]; found {
		return found
	}
	if _, found := obj.Annotations[BetaStorageClassAnnotation]; found {
		return found
	}
	if _, found := obj.Annotations[AlphaStorageClassAnnotation]; found {
		return found
	}

	return false

}

// IsDefaultAnnotationText returns a pretty Yes/No String if
// the annotation is set
// TODO: remove Beta when no longer needed
func IsDefaultAnnotationText(obj api.ObjectMeta) string {
	if obj.Annotations[IsDefaultStorageClassAnnotation] == "true" {
		return "Yes"
	}
	if obj.Annotations[BetaIsDefaultStorageClassAnnotation] == "true" {
		return "Yes"
	}

	return "No"
}

// IsDefaultAnnotation returns a boolean if
// the annotation is set
// TODO: remove Beta when no longer needed
func IsDefaultAnnotation(obj api.ObjectMeta) bool {
	if obj.Annotations[IsDefaultStorageClassAnnotation] == "true" {
		return true
	}
	if obj.Annotations[BetaIsDefaultStorageClassAnnotation] == "true" {
		return true
	}

	return false
}
