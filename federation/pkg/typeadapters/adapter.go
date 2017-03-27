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

package typeadapters

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// FederatedTypeAdapter defines operations for interacting with a
// federated type.  Code written to this interface can then target any
// type for which an implementation of this interface exists.
type FederatedTypeAdapter interface {
	Kind() string
	Equivalent(obj1, obj2 pkgruntime.Object) bool
	ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta
	NamespacedName(obj pkgruntime.Object) types.NamespacedName

	// Fed* operations target the federation control plane
	FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error)
	FedGet(namespacedName types.NamespacedName) (pkgruntime.Object, error)
	FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error)
	FedDelete(namespacedName types.NamespacedName, options *metav1.DeleteOptions) error

	// The following operations are intended to target a cluster that is a member of a federation
	ClusterGet(client clientset.Interface, namespacedName types.NamespacedName) (pkgruntime.Object, error)

	NewTestObject(namespace string) pkgruntime.Object
}

// FederatedTypeAdapterFactory defines the function signature for
// factory methods that create FederatedTypeAdapters.  Such methods
// should be registered with RegisterAdapterFactory to ensure the type
// adapter is discoverable.
type FederatedTypeAdapterFactory func(client federationclientset.Interface) FederatedTypeAdapter

var typeAdapterRegistry = make(map[string]FederatedTypeAdapterFactory)

// RegisterAdapterFactory ensures that the given kind and adapter
// factory will be returned in the results of the AdapterFactories method.
func RegisterAdapterFactory(kind string, factory FederatedTypeAdapterFactory) {
	_, ok := typeAdapterRegistry[kind]
	if ok {
		// TODO Is panicking ok given that this is part of a type-registration mechanism
		panic(fmt.Sprintf("An adapter has already been registered for federated type %q", kind))
	}
	typeAdapterRegistry[kind] = factory
}

// AdapterFactories returns a mapping of known federated type
// (e.g. "secret") to the factory method that will create an adapter
// for that type.
func AdapterFactories() map[string]FederatedTypeAdapterFactory {
	// Return a copy to avoid accidental mutation
	result := make(map[string]FederatedTypeAdapterFactory)
	for key, value := range typeAdapterRegistry {
		result[key] = value
	}
	return result
}
