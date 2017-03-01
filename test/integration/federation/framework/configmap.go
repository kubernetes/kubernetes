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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	configmapcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/configmap"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

type ConfigMapAdapter struct {
	client federationclientset.Interface
}

func (a *ConfigMapAdapter) Kind() string {
	return "configmap"
}

func (a *ConfigMapAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	configMap1 := obj1.(*apiv1.ConfigMap)
	configMap2 := obj2.(*apiv1.ConfigMap)
	return util.ConfigMapEquivalent(configMap1, configMap2)
}

func (a *ConfigMapAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*apiv1.ConfigMap).ObjectMeta
}

func (a *ConfigMapAdapter) NamespacedName(obj pkgruntime.Object) types.NamespacedName {
	configMap := obj.(*apiv1.ConfigMap)
	return types.NamespacedName{Namespace: configMap.Namespace, Name: configMap.Name}
}

func (a *ConfigMapAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	configMap := obj.(*apiv1.ConfigMap)
	return a.client.CoreV1().ConfigMaps(configMap.Namespace).Create(configMap)
}

func (a *ConfigMapAdapter) FedGet(namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return a.client.CoreV1().ConfigMaps(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (a *ConfigMapAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	configMap := obj.(*apiv1.ConfigMap)
	return a.client.CoreV1().ConfigMaps(configMap.Namespace).Update(configMap)
}

func (a *ConfigMapAdapter) FedDelete(namespacedName types.NamespacedName, options *metav1.DeleteOptions) error {
	return a.client.CoreV1().ConfigMaps(namespacedName.Namespace).Delete(namespacedName.Name, options)
}

func (a *ConfigMapAdapter) Get(client clientset.Interface, namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return client.CoreV1().ConfigMaps(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

type ConfigMapFixture struct {
	client   federationclientset.Interface
	adapter  *ConfigMapAdapter
	stopChan chan struct{}
}

func (f *ConfigMapFixture) SetUp(t *testing.T, testClient federationclientset.Interface, config *restclient.Config) {
	f.adapter = &ConfigMapAdapter{client: testClient}
	f.stopChan = make(chan struct{})
	configmapcontroller.StartConfigMapController(config, f.stopChan, true)
}

func (f *ConfigMapFixture) TearDown(t *testing.T) {
	close(f.stopChan)
}

func (f *ConfigMapFixture) Kind() string {
	adapter := &ConfigMapAdapter{}
	return adapter.Kind()
}

func (f *ConfigMapFixture) Adapter() ResourceAdapter {
	return f.adapter
}

func (f *ConfigMapFixture) NewObject(namespace string) pkgruntime.Object {
	return NewTestConfigMap(namespace)
}

func NewTestConfigMap(namespace string) pkgruntime.Object {
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
