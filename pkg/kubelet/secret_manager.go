/*
Copyright 2016 The Kubernetes Authors.

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

package kubelet

import (
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

type secretManager interface {
	// Get secret by secret namespace and name.
	GetSecret(namespace, name string) (*api.Secret, error)
}

// simpleSecretManager implements SecretManager interfaces with
// simple operations to apiserver.
type simpleSecretManager struct {
	kubeClient clientset.Interface
}

func newSimpleSecretManager(kubeClient clientset.Interface) (secretManager, error) {
	return &simpleSecretManager{kubeClient: kubeClient}, nil
}

func (s *simpleSecretManager) GetSecret(namespace, name string) (*api.Secret, error) {
	return s.kubeClient.Core().Secrets(namespace).Get(name)
}
