/*
Copyright 2023 The Kubernetes Authors.

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

	storagev1alpha1 "k8s.io/api/storage/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	storagev1alpha1listers "k8s.io/client-go/listers/storage/v1alpha1"
	"k8s.io/klog/v2"
)

const (
	// AlphaIsDefaultVolumeAttributesClassAnnotation is the alpha version of IsDefaultVolumeAttributesClassAnnotation.
	AlphaIsDefaultVolumeAttributesClassAnnotation = "volumeattributesclass.alpha.kubernetes.io/is-default-class"
)

// GetDefaultVolumeAttributesClass returns the default VolumeAttributesClass from the store, or nil.
func GetDefaultVolumeAttributesClass(lister storagev1alpha1listers.VolumeAttributesClassLister, driverName string) (*storagev1alpha1.VolumeAttributesClass, error) {
	list, err := lister.List(labels.Everything())
	if err != nil {
		return nil, err
	}

	defaultClasses := []*storagev1alpha1.VolumeAttributesClass{}
	for _, class := range list {
		if IsDefaultVolumeAttributesClassAnnotation(class.ObjectMeta) && class.DriverName == driverName {
			defaultClasses = append(defaultClasses, class)
			klog.V(4).Infof("GetDefaultVolumeAttributesClass added: %s", class.Name)
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
		klog.V(4).Infof("%d default VolumeAttributesClass were found, choosing: %s", len(defaultClasses), defaultClasses[0].Name)
	}

	return defaultClasses[0], nil
}

// IsDefaultVolumeAttributesClassAnnotation returns a boolean if the default
// volume attributes class annotation is set
func IsDefaultVolumeAttributesClassAnnotation(obj metav1.ObjectMeta) bool {
	return obj.Annotations[AlphaIsDefaultVolumeAttributesClassAnnotation] == "true"
}
