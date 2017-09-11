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

package federatedtypes

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
)

// FederatedTypeAdapter defines operations for interacting with a
// federated type.  Code written to this interface can then target any
// type for which an implementation of this interface exists.
type FederatedTypeAdapter interface {
	Kind() string
	ObjectType() pkgruntime.Object
	IsExpectedType(obj interface{}) bool
	Copy(obj pkgruntime.Object) pkgruntime.Object
	Equivalent(obj1, obj2 pkgruntime.Object) bool
	QualifiedName(obj pkgruntime.Object) QualifiedName
	ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta

	// Fed* operations target the federation control plane
	FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error)
	FedDelete(qualifiedName QualifiedName, options *metav1.DeleteOptions) error
	FedGet(qualifiedName QualifiedName) (pkgruntime.Object, error)
	FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error)
	FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error)
	FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error)

	// The following operations are intended to target a cluster that is a member of a federation
	ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error)
	ClusterDelete(client kubeclientset.Interface, qualifiedName QualifiedName, options *metav1.DeleteOptions) error
	ClusterGet(client kubeclientset.Interface, qualifiedName QualifiedName) (pkgruntime.Object, error)
	ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error)
	ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error)
	ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error)

	IsSchedulingAdapter() bool

	NewTestObject(namespace string) pkgruntime.Object
}

// AdapterFactory defines the function signature for factory methods
// that create instances of FederatedTypeAdapter.  Such methods should
// be registered with RegisterAdapterFactory to ensure the type
// adapter is discoverable.
type AdapterFactory func(client federationclientset.Interface, config *restclient.Config, adapterSpecificArgs map[string]interface{}) FederatedTypeAdapter

// SetAnnotation sets the given key and value in the given object's ObjectMeta.Annotations map
func SetAnnotation(adapter FederatedTypeAdapter, obj pkgruntime.Object, key, value string) {
	meta := adapter.ObjectMeta(obj)
	if meta.Annotations == nil {
		meta.Annotations = make(map[string]string)
	}
	meta.Annotations[key] = value
}

// ObjectKey returns a cluster-unique key for the given object
func ObjectKey(adapter FederatedTypeAdapter, obj pkgruntime.Object) string {
	return adapter.QualifiedName(obj).String()
}
