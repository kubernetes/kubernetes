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

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// IsDefaultStorageClassAnnotation represents a StorageClass annotation that
// marks a class as the default StorageClass
const IsDefaultStorageClassAnnotation = "storageclass.kubernetes.io/is-default-class"

// HasDefaultAnnotation returns a boolean if the object metadata has the default annotation set.
func HasDefaultAnnotation(obj metav1.ObjectMeta) bool {
	return obj.Annotations[IsDefaultStorageClassAnnotation] == "true"
}
