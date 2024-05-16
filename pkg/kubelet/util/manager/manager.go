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

package manager

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

// Manager is the interface for registering and unregistering
// objects referenced by pods in the underlying cache and
// extracting those from that cache if needed.
type Manager interface {
	// Get object by its namespace and name.
	GetObject(namespace, name string) (runtime.Object, error)

	// WARNING: Register/UnregisterPod functions should be efficient,
	// i.e. should not block on network operations.

	// RegisterPod registers all objects referenced from a given pod.
	//
	// NOTE: All implementations of RegisterPod should be idempotent.
	RegisterPod(pod *v1.Pod)

	// UnregisterPod unregisters objects referenced from a given pod that are not
	// used by any other registered pod.
	//
	// NOTE: All implementations of UnregisterPod should be idempotent.
	UnregisterPod(pod *v1.Pod)
}

// Store is the interface for a object cache that
// can be used by cacheBasedManager.
type Store interface {
	// AddReference adds a reference from referencedFrom to the object to the store.
	// Note that multiple additions to the store has to be allowed
	// in the implementations and effectively treated as refcounted.
	AddReference(namespace, name string, referencedFrom types.UID)
	// DeleteReference deletes a reference from referencedFrom to the object from the store.
	// Note that object should be deleted only when there was a
	// corresponding Delete call for each of Add calls (effectively
	// when refcount of every referenceFrom was reduced to zero).
	DeleteReference(namespace, name string, referencedFrom types.UID)
	// Get an object from a store.
	Get(namespace, name string) (runtime.Object, error)
}
