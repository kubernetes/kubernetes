/*
Copyright 2022 The Kubernetes Authors.

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
	"sort"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	storagev1listers "k8s.io/client-go/listers/storage/v1"
	"k8s.io/klog/v2"
)

const (
	// isDefaultStorageClassAnnotation represents a StorageClass annotation that
	// marks a class as the default StorageClass
	IsDefaultStorageClassAnnotation = "storageclass.kubernetes.io/is-default-class"

	// betaIsDefaultStorageClassAnnotation is the beta version of IsDefaultStorageClassAnnotation.
	// TODO: remove Beta when no longer used
	BetaIsDefaultStorageClassAnnotation = "storageclass.beta.kubernetes.io/is-default-class"
)

// GetDefaultClass returns the default StorageClass from the store, or nil.
func GetDefaultClass(lister storagev1listers.StorageClassLister) (*storagev1.StorageClass, error) {
	list, err := lister.List(labels.Everything())
	if err != nil {
		return nil, err
	}

	defaultClasses := []*storagev1.StorageClass{}
	for _, class := range list {
		if IsDefaultAnnotation(class.ObjectMeta) {
			defaultClasses = append(defaultClasses, class)
			klog.V(4).Infof("GetDefaultClass added: %s", class.Name)
		}
	}

	if len(defaultClasses) == 0 {
		return nil, nil
	}

	// Primary sort by creation timestamp, newest first
	// Secondary sort by class name, ascending order
	sort.Slice(defaultClasses, func(i, j int) bool {
		if defaultClasses[i].CreationTimestamp.UnixNano() == defaultClasses[j].CreationTimestamp.UnixNano() {
			return defaultClasses[i].Name < defaultClasses[j].Name
		}
		return defaultClasses[i].CreationTimestamp.UnixNano() > defaultClasses[j].CreationTimestamp.UnixNano()
	})
	if len(defaultClasses) > 1 {
		klog.V(4).Infof("%d default StorageClasses were found, choosing the newest: %s", len(defaultClasses), defaultClasses[0].Name)
	}

	return defaultClasses[0], nil
}

// IsDefaultAnnotation returns a boolean if the default storage class
// annotation is set
// TODO: remove Beta when no longer needed
func IsDefaultAnnotation(obj metav1.ObjectMeta) bool {
	if obj.Annotations[IsDefaultStorageClassAnnotation] == "true" {
		return true
	}
	if obj.Annotations[BetaIsDefaultStorageClassAnnotation] == "true" {
		return true
	}

	return false
}
