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
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

type DaemonSetAdapter struct {
	client federationclientset.Interface
}

func NewDaemonSetAdapter(client federationclientset.Interface) *DaemonSetAdapter {
	return &DaemonSetAdapter{client: client}
}

func (a *DaemonSetAdapter) SetClient(client federationclientset.Interface) {
	a.client = client
}

func (a *DaemonSetAdapter) Kind() string {
	return "daemonset"
}

func (a *DaemonSetAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	daemonSet1 := obj1.(*extensionsv1.DaemonSet)
	daemonSet2 := obj2.(*extensionsv1.DaemonSet)
	return util.DaemonSetsEquivalent(daemonSet1, daemonSet2)
}

func (a *DaemonSetAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*extensionsv1.DaemonSet).ObjectMeta
}

func (a *DaemonSetAdapter) NamespacedName(obj pkgruntime.Object) types.NamespacedName {
	daemonSet := obj.(*extensionsv1.DaemonSet)
	return types.NamespacedName{Namespace: daemonSet.Namespace, Name: daemonSet.Name}
}

func (a *DaemonSetAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	daemonSet := obj.(*extensionsv1.DaemonSet)
	return a.client.ExtensionsV1beta1().DaemonSets(daemonSet.Namespace).Create(daemonSet)
}

func (a *DaemonSetAdapter) FedGet(namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return a.client.ExtensionsV1beta1().DaemonSets(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (a *DaemonSetAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	daemonSet := obj.(*extensionsv1.DaemonSet)
	return a.client.ExtensionsV1beta1().DaemonSets(daemonSet.Namespace).Update(daemonSet)
}

func (a *DaemonSetAdapter) FedDelete(namespacedName types.NamespacedName, options *metav1.DeleteOptions) error {
	return a.client.ExtensionsV1beta1().DaemonSets(namespacedName.Namespace).Delete(namespacedName.Name, options)
}

func (a *DaemonSetAdapter) ClusterGet(client clientset.Interface, namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return client.ExtensionsV1beta1().DaemonSets(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (f *DaemonSetAdapter) NewTestObject(namespace string) pkgruntime.Object {
	return &extensionsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-daemonset-",
			Namespace:    namespace,
		},
		Spec: extensionsv1.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"aaa": "bbb"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "container1",
							Image: "gcr.io/google_containers/serve_hostname:v1.4",
							Ports: []v1.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	}
}
