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

package framework

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// ResourceAdapter defines operations for interacting with a federated
// type.  Tests written to target this interface can then target any
// type for which an implementation of this interface exists.
//
// TODO reuse resource adapters defined for use with a generic controller as per
// https://github.com/kubernetes/kubernetes/pull/41050
type ResourceAdapter interface {
	Kind() string
	Equivalent(obj1, obj2 pkgruntime.Object) bool
	ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta
	NamespacedName(obj pkgruntime.Object) types.NamespacedName

	FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error)
	FedGet(namespacedName types.NamespacedName) (pkgruntime.Object, error)
	FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error)
	FedDelete(namespacedName types.NamespacedName, options *metav1.DeleteOptions) error

	Get(client clientset.Interface, namespacedName types.NamespacedName) (pkgruntime.Object, error)
}
