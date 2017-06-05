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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// FederatedTypeAdapter defines operations for interacting with a
// federated type.  Code written to this interface can then target any
// type for which an implementation of this interface exists.
type FederatedTypeAdapter interface {
	SetClient(client federationclientset.Interface)

	Kind() string
	ObjectType() pkgruntime.Object
	IsExpectedType(obj interface{}) bool
	Copy(obj pkgruntime.Object) pkgruntime.Object
	Equivalent(obj1, obj2 pkgruntime.Object) bool
	NamespacedName(obj pkgruntime.Object) types.NamespacedName
	ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta

	// Fed* operations target the federation control plane
	FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error)
	FedDelete(namespacedName types.NamespacedName, options *metav1.DeleteOptions) error
	FedGet(namespacedName types.NamespacedName) (pkgruntime.Object, error)
	FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error)
	FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error)
	FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error)

	// The following operations are intended to target a cluster that is a member of a federation
	ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error)
	ClusterDelete(client kubeclientset.Interface, nsName types.NamespacedName, options *metav1.DeleteOptions) error
	ClusterGet(client kubeclientset.Interface, namespacedName types.NamespacedName) (pkgruntime.Object, error)
	ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error)
	ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error)
	ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error)

	NewTestObject(namespace string) pkgruntime.Object
}
