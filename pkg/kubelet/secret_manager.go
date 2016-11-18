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
	"fmt"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
)

type secretManager interface {
	// Get secret by secret namespace and name.
	GetSecret(namespace, name string) (*api.Secret, error)

	// RegisterPod registers all secrets from a given pod.
	RegisterPod(pod *api.Pod)

	// UnregisterPod unregisters secrets from a given pod that are not
	// registered still by any other registered pod.
	UnregisterPod(pod *api.Pod)
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

func (s *simpleSecretManager) RegisterPod(pod *api.Pod) {
}

func (s *simpleSecretManager) UnregisterPod(pod *api.Pod) {
}

// secretStoreItems is a single item stored in secretStore.
type secretStoreItem struct {
	secret   *api.Secret
	refCount int
}

// secretStore is a local cache of secrets.
type secretStore struct {
	kubeClient clientset.Interface

	sync.Mutex
	items map[string]*secretStoreItem
}

func newSecretStore(kubeClient clientset.Interface) *secretStore {
	ss := &secretStore{
		kubeClient: kubeClient,
		items:      make(map[string]*secretStoreItem),
	}
	go wait.NonSlidingUntil(func() { ss.Refresh() }, time.Minute, wait.NeverStop)
	return ss
}

func keyFunc(namespace, name string) string {
	return namespace + "/" + name
}

func namespaceNameFunc(key string) (string, string, error) {
	items := strings.Split(key, "/")
	if len(items) != 2 {
		return "", "", fmt.Errorf("invalid key: %s", key)
	}
	return items[0], items[1], nil
}

func (s *secretStore) Add(namespace, name string) {
	key := keyFunc(namespace, name)
	secret, err := s.kubeClient.Core().Secrets(namespace).Get(name)
	if err != nil {
		glog.Errorf("Unable to retrieve a secret %s/%s: %v", namespace, name, err)
	}

	s.Lock()
	defer s.Unlock()
	if item, ok := s.items[key]; ok {
		item.secret = secret
		item.refCount++
	} else {
		s.items[key] = &secretStoreItem{secret: secret, refCount: 1}
	}
}

func (s *secretStore) Delete(namespace, name string) {
	key := keyFunc(namespace, name)

	s.Lock()
	defer s.Unlock()
	if item, ok := s.items[key]; ok {
		item.refCount--
		if item.refCount == 0 {
			delete(s.items, key)
		}
	}
}

func (s *secretStore) Get(namespace, name string) (*api.Secret, error) {
	key := keyFunc(namespace, name)

	s.Lock()
	defer s.Unlock()
	if item, ok := s.items[key]; ok {
		return item.secret, nil
	}
	return nil, fmt.Errorf("secret no present")
}

func (s *secretStore) Refresh() {
	s.Lock()
	keys := make([]string, 0, len(s.items))
	for key := range s.items {
		keys = append(keys, key)
	}
	s.Unlock()

	secrets := make([]*api.Secret, 0, len(keys))
	for _, key := range keys {
		namespace, name, err := namespaceNameFunc(key)
		if err != nil {
			glog.Errorf("Unexpected key transform error: %v", err)
			secrets = append(secrets, nil)
			continue
		}
		secret, err := s.kubeClient.Core().Secrets(namespace).Get(name)
		if err != nil {
			glog.Errorf("Unable to retrieve a secret %s/%s: %v", namespace, name, err)
			secret = nil
		}
		secrets = append(secrets, secret)
	}

	s.Lock()
	defer s.Unlock()
	for i, key := range keys {
		if secrets[i] == nil {
			// Ignore updating to nil.
		}
		if item, ok := s.items[key]; ok {
			item.secret = secrets[i]
		}
	}
}

// cachingSecretManager keeps a cache of all secrets necessary for registered pods.
// It implements the following logic:
// - whenever a pod is created or updated, the current versions of all its secrets
//   are grabbed from apiserver and stored in local cache
// - every GetSecret call is served from local cache
// - every X seconds we are refreshing the local cache by grabbing current version
//   of all registered secrets from apiserver
type cachingSecretManager struct {
	secretStore *secretStore

	sync.Mutex
	registeredPods map[string]*api.Pod
}

func newCachingSecretManager(kubeClient clientset.Interface) (secretManager, error) {
	return &cachingSecretManager{
		secretStore:    newSecretStore(kubeClient),
		registeredPods: make(map[string]*api.Pod),
	}, nil
}

func (c *cachingSecretManager) GetSecret(namespace, name string) (*api.Secret, error) {
	return c.secretStore.Get(namespace, name)
}

func (c *cachingSecretManager) RegisterPod(pod *api.Pod) {
	for _, reference := range pod.Spec.ImagePullSecrets {
		c.secretStore.Add(pod.Namespace, reference.Name)
	}
	var prev *api.Pod
	func() {
		key := keyFunc(pod.Namespace, pod.Name)
		c.Lock()
		defer c.Unlock()
		prev = c.registeredPods[key]
		c.registeredPods[key] = pod
	}()
	if prev != nil {
		for _, reference := range prev.Spec.ImagePullSecrets {
			c.secretStore.Delete(prev.Namespace, reference.Name)
		}
	}
}

func (c *cachingSecretManager) UnregisterPod(pod *api.Pod) {
	var prev *api.Pod
	func() {
		key := keyFunc(pod.Namespace, pod.Name)
		c.Lock()
		defer c.Unlock()
		prev = c.registeredPods[key]
		delete(c.registeredPods, key)
	}()
	if prev != nil {
		for _, reference := range prev.Spec.ImagePullSecrets {
			c.secretStore.Delete(prev.Namespace, reference.Name)
		}
	}
}
