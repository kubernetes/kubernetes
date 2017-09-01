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
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
)

const (
	ConfigMapKind           = "configmap"
	ConfigMapControllerName = "configmaps"
)

func init() {
	RegisterFederatedType(ConfigMapKind, ConfigMapControllerName, []schema.GroupVersionResource{apiv1.SchemeGroupVersion.WithResource(ConfigMapControllerName)}, NewConfigMapAdapter)
}

type ConfigMapAdapter struct {
	client federationclientset.Interface
}

func NewConfigMapAdapter(client federationclientset.Interface, config *restclient.Config, adapterSpecificArgs map[string]interface{}) FederatedTypeAdapter {
	return &ConfigMapAdapter{client: client}
}

func (a *ConfigMapAdapter) Kind() string {
	return ConfigMapKind
}

func (a *ConfigMapAdapter) ObjectType() pkgruntime.Object {
	return &apiv1.ConfigMap{}
}

func (a *ConfigMapAdapter) IsExpectedType(obj interface{}) bool {
	_, ok := obj.(*apiv1.ConfigMap)
	return ok
}

func (a *ConfigMapAdapter) Copy(obj pkgruntime.Object) pkgruntime.Object {
	configmap := obj.(*apiv1.ConfigMap)
	return &apiv1.ConfigMap{
		ObjectMeta: util.DeepCopyRelevantObjectMeta(configmap.ObjectMeta),
		Data:       configmap.Data,
	}
}

func (a *ConfigMapAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	configmap1 := obj1.(*apiv1.ConfigMap)
	configmap2 := obj2.(*apiv1.ConfigMap)
	return util.ConfigMapEquivalent(configmap1, configmap2)
}

func (a *ConfigMapAdapter) QualifiedName(obj pkgruntime.Object) QualifiedName {
	configmap := obj.(*apiv1.ConfigMap)
	return QualifiedName{Namespace: configmap.Namespace, Name: configmap.Name}
}

func (a *ConfigMapAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*apiv1.ConfigMap).ObjectMeta
}

func (a *ConfigMapAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	configmap := obj.(*apiv1.ConfigMap)
	return a.client.CoreV1().ConfigMaps(configmap.Namespace).Create(configmap)
}

func (a *ConfigMapAdapter) FedDelete(qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return a.client.CoreV1().ConfigMaps(qualifiedName.Namespace).Delete(qualifiedName.Name, options)
}

func (a *ConfigMapAdapter) FedGet(qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return a.client.CoreV1().ConfigMaps(qualifiedName.Namespace).Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *ConfigMapAdapter) FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return a.client.CoreV1().ConfigMaps(namespace).List(options)
}

func (a *ConfigMapAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	configmap := obj.(*apiv1.ConfigMap)
	return a.client.CoreV1().ConfigMaps(configmap.Namespace).Update(configmap)
}

func (a *ConfigMapAdapter) FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return a.client.CoreV1().ConfigMaps(namespace).Watch(options)
}

func (a *ConfigMapAdapter) ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	configmap := obj.(*apiv1.ConfigMap)
	return client.CoreV1().ConfigMaps(configmap.Namespace).Create(configmap)
}

func (a *ConfigMapAdapter) ClusterDelete(client kubeclientset.Interface, qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return client.CoreV1().ConfigMaps(qualifiedName.Namespace).Delete(qualifiedName.Name, options)
}

func (a *ConfigMapAdapter) ClusterGet(client kubeclientset.Interface, qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return client.CoreV1().ConfigMaps(qualifiedName.Namespace).Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *ConfigMapAdapter) ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return client.CoreV1().ConfigMaps(namespace).List(options)
}

func (a *ConfigMapAdapter) ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	configmap := obj.(*apiv1.ConfigMap)
	return client.CoreV1().ConfigMaps(configmap.Namespace).Update(configmap)
}

func (a *ConfigMapAdapter) ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return client.CoreV1().ConfigMaps(namespace).Watch(options)
}

func (a *ConfigMapAdapter) IsSchedulingAdapter() bool {
	return false
}

func (a *ConfigMapAdapter) NewTestObject(namespace string) pkgruntime.Object {
	return &apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-configmap-",
			Namespace:    namespace,
		},
		Data: map[string]string{
			"A": "ala ma kota",
		},
	}
}
