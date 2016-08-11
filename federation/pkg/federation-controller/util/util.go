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
	"reflect"

	"k8s.io/kubernetes/pkg/api/v1"
)

/*
ObjectMetaIsEquivalent determines whether two ObjectMeta's (typically one from a federated API object,
and the other from a cluster object) are equivalent.
*/
func ObjectMetaIsEquivalent(m1, m2 v1.ObjectMeta) bool {
	// First make all of the read-only fields equal, then perform a deep equality comparison
	m1.SelfLink = m2.SelfLink                   // Might be different in different cluster contexts.
	m1.UID = m2.UID                             // Definitely different in different cluster contexts
	m1.ResourceVersion = m2.ResourceVersion     // Definitely different in different cluster contexts
	m1.Generation = m2.Generation               // Might be different in different cluster contexts.
	m1.CreationTimestamp = m2.CreationTimestamp // Definitely different in different cluster contexts.
	m1.DeletionTimestamp = m2.DeletionTimestamp // Might be different in different cluster contexts.
	m1.OwnerReferences = nil                    // Might be different in different cluster contexts.
	m2.OwnerReferences = nil
	m1.Finalizers = nil // Might be different in different cluster contexts.
	m2.Finalizers = nil

	return reflect.DeepEqual(m1, m2)
}
