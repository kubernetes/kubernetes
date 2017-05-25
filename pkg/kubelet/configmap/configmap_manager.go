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

package configmap

import (
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type Manager interface {
	// Get configmap by configmap namespace and name.
	GetConfigMap(namespace, name string) (*v1.ConfigMap, error)

	// WARNING: Register/UnregisterPod functions should be efficient,
	// i.e. should not block on network operations.

	// RegisterPod registers all configmaps from a given pod.
	RegisterPod(pod *v1.Pod)

	// UnregisterPod unregisters configmaps from a given pod that are not
	// used by any other registered pod.
	UnregisterPod(pod *v1.Pod)
}

// simpleConfigMapManager implements ConfigMap Manager interface with
// simple operations to apiserver.
type simpleConfigMapManager struct {
	kubeClient clientset.Interface
}

func NewSimpleConfigMapManager(kubeClient clientset.Interface) Manager {
	return &simpleConfigMapManager{kubeClient: kubeClient}
}

func (s *simpleConfigMapManager) GetConfigMap(namespace, name string) (*v1.ConfigMap, error) {
	return s.kubeClient.Core().ConfigMaps(namespace).Get(name, metav1.GetOptions{})
}

func (s *simpleConfigMapManager) RegisterPod(pod *v1.Pod) {
}

func (s *simpleConfigMapManager) UnregisterPod(pod *v1.Pod) {
}
