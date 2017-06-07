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

	// Kubernetes daemonset controller writes a daemonset's hash to
	// the object label as an optimization to avoid recomputing it every
	// time. Adding a new label to the object that the federation is
	// unaware of causes problems because federated controllers compare
	// the objects in federation and their equivalents in clusters and
	// try to reconcile them. This leads to a constant fight between the
	// federated daemonset controller and the cluster controllers, and
	// they never reach a stable state.
	//
	// Ideally, cluster components should not update an object's spec or
	// metadata in a way federation cannot replicate. They can update an
	// object's status though. Therefore, this daemonset hash should
	// be a field in daemonset's status, not a label in object meta.
	// @janetkuo says that this label is only a short term solution. In
	// the near future, they are going to replace it with revision numbers
	// in daemonset status. We can then rip this bandaid out.
	//
	// We are deleting the keys here and that should be fine since we are
	// working on object copies. Also, propagating the deleted labels
	// should also be fine because we don't support daemonset rolling
	// update in federation yet.
	delete(daemonset1.ObjectMeta.Labels, extensionsv1.DefaultDaemonSetUniqueLabelKey)
	delete(daemonset2.ObjectMeta.Labels, extensionsv1.DefaultDaemonSetUniqueLabelKey)

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
