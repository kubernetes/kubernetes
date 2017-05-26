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
	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

const (
	DaemonSetKind           = "daemonset"
	DaemonSetControllerName = "daemonsets"
)

func init() {
	RegisterFederatedType(DaemonSetKind, DaemonSetControllerName, []schema.GroupVersionResource{extensionsv1.SchemeGroupVersion.WithResource(DaemonSetControllerName)}, NewDaemonSetAdapter)
}

type DaemonSetAdapter struct {
	client federationclientset.Interface
}

func NewDaemonSetAdapter(client federationclientset.Interface) FederatedTypeAdapter {
	return &DaemonSetAdapter{client: client}
}

func (a *DaemonSetAdapter) Kind() string {
	return DaemonSetKind
}

func (a *DaemonSetAdapter) ObjectType() pkgruntime.Object {
	return &extensionsv1.DaemonSet{}
}

func (a *DaemonSetAdapter) IsExpectedType(obj interface{}) bool {
	_, ok := obj.(*extensionsv1.DaemonSet)
	return ok
}

func (a *DaemonSetAdapter) Copy(obj pkgruntime.Object) pkgruntime.Object {
	daemonset := obj.(*extensionsv1.DaemonSet)
	return &extensionsv1.DaemonSet{
		ObjectMeta: util.DeepCopyRelevantObjectMeta(daemonset.ObjectMeta),
		Spec:       *(util.DeepCopyApiTypeOrPanic(&daemonset.Spec).(*extensionsv1.DaemonSetSpec)),
	}
}

func (a *DaemonSetAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	daemonset1 := obj1.(*extensionsv1.DaemonSet)
	daemonset2 := obj2.(*extensionsv1.DaemonSet)
	return util.ObjectMetaEquivalent(daemonset1.ObjectMeta, daemonset2.ObjectMeta) && reflect.DeepEqual(daemonset1.Spec, daemonset2.Spec)
}

func (a *DaemonSetAdapter) NamespacedName(obj pkgruntime.Object) types.NamespacedName {
	daemonset := obj.(*extensionsv1.DaemonSet)
	return types.NamespacedName{Namespace: daemonset.Namespace, Name: daemonset.Name}
}

func (a *DaemonSetAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*extensionsv1.DaemonSet).ObjectMeta
}

func (a *DaemonSetAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	daemonset := obj.(*extensionsv1.DaemonSet)
	return a.client.Extensions().DaemonSets(daemonset.Namespace).Create(daemonset)
}

func (a *DaemonSetAdapter) FedDelete(namespacedName types.NamespacedName, options *metav1.DeleteOptions) error {
	return a.client.Extensions().DaemonSets(namespacedName.Namespace).Delete(namespacedName.Name, options)
}

func (a *DaemonSetAdapter) FedGet(namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return a.client.Extensions().DaemonSets(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (a *DaemonSetAdapter) FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return a.client.Extensions().DaemonSets(namespace).List(options)
}

func (a *DaemonSetAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	daemonset := obj.(*extensionsv1.DaemonSet)
	return a.client.Extensions().DaemonSets(daemonset.Namespace).Update(daemonset)
}

func (a *DaemonSetAdapter) FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return a.client.Extensions().DaemonSets(namespace).Watch(options)
}

func (a *DaemonSetAdapter) ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	daemonset := obj.(*extensionsv1.DaemonSet)
	return client.Extensions().DaemonSets(daemonset.Namespace).Create(daemonset)
}

func (a *DaemonSetAdapter) ClusterDelete(client kubeclientset.Interface, nsName types.NamespacedName, options *metav1.DeleteOptions) error {
	return client.Extensions().DaemonSets(nsName.Namespace).Delete(nsName.Name, options)
}

func (a *DaemonSetAdapter) ClusterGet(client kubeclientset.Interface, namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return client.Extensions().DaemonSets(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (a *DaemonSetAdapter) ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return client.Extensions().DaemonSets(namespace).List(options)
}

func (a *DaemonSetAdapter) ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	daemonset := obj.(*extensionsv1.DaemonSet)
	return client.Extensions().DaemonSets(daemonset.Namespace).Update(daemonset)
}

func (a *DaemonSetAdapter) ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return client.Extensions().DaemonSets(namespace).Watch(options)
}

func (a *DaemonSetAdapter) IsSchedulingAdapter() bool {
	return false
}

func (a *DaemonSetAdapter) NewTestObject(namespace string) pkgruntime.Object {
	return &extensionsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-daemonset-",
			Namespace:    namespace,
			Labels:       map[string]string{"app": "test-daemonset"},
		},
		Spec: extensionsv1.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": "test-pod"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "test-daemonset",
							Image: "images/test-daemonset",
							Ports: []v1.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	}
}
