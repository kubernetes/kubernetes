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

package polymorphichelpers

import (
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Check whether the kind of resources could be exposed
func canBeExposed(kind schema.GroupKind) error {
	switch kind {
	case
		corev1.SchemeGroupVersion.WithKind("ReplicationController").GroupKind(),
		corev1.SchemeGroupVersion.WithKind("Service").GroupKind(),
		corev1.SchemeGroupVersion.WithKind("Pod").GroupKind(),
		appsv1.SchemeGroupVersion.WithKind("Deployment").GroupKind(),
		appsv1.SchemeGroupVersion.WithKind("ReplicaSet").GroupKind(),
		extensionsv1beta1.SchemeGroupVersion.WithKind("Deployment").GroupKind(),
		extensionsv1beta1.SchemeGroupVersion.WithKind("ReplicaSet").GroupKind():
		// nothing to do here
	default:
		return fmt.Errorf("cannot expose a %s", kind)
	}
	return nil
}
