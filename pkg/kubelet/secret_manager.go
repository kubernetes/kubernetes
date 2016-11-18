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
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
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

type objectKey struct {
	namespace string
	name      string
}

// secretStoreItems is a single item stored in secretStore.
type secretStoreItem struct {
	secret   *api.Secret
	err      error
	refCount int
}

// secretStore is a local cache of secrets.
type secretStore struct {
	kubeClient clientset.Interface

	lock  sync.Mutex
	items map[objectKey]*secretStoreItem
}

func newSecretStore(kubeClient clientset.Interface) *secretStore {
	ss := &secretStore{
		kubeClient: kubeClient,
		items:      make(map[objectKey]*secretStoreItem),
	}
	go wait.NonSlidingUntil(func() { ss.Refresh() }, time.Minute, wait.NeverStop)
	return ss
}

func (s *secretStore) Add(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}
	secret, err := s.kubeClient.Core().Secrets(namespace).Get(name)

	s.lock.Lock()
	defer s.lock.Unlock()
	if item, ok := s.items[key]; ok {
		item.secret = secret
		item.err = err
		item.refCount++
	} else {
		s.items[key] = &secretStoreItem{secret: secret, err: err, refCount: 1}
	}
}

func (s *secretStore) Delete(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	s.lock.Lock()
	defer s.lock.Unlock()
	if item, ok := s.items[key]; ok {
		item.refCount--
		if item.refCount == 0 {
			delete(s.items, key)
		}
	}
}

func (s *secretStore) Get(namespace, name string) (*api.Secret, error) {
	key := objectKey{namespace: namespace, name: name}

	s.lock.Lock()
	defer s.lock.Unlock()
	if item, ok := s.items[key]; ok {
		return item.secret, item.err
	}
	return nil, fmt.Errorf("secret not registered")
}

func (s *secretStore) Refresh() {
	s.lock.Lock()
	keys := make([]objectKey, 0, len(s.items))
	for key := range s.items {
		keys = append(keys, key)
	}
	s.lock.Unlock()

	type result struct {
		secret *api.Secret
		err    error
	}
	results := make([]result, 0, len(keys))
	for _, key := range keys {
		secret, err := s.kubeClient.Core().Secrets(key.namespace).Get(key.name)
		if err != nil {
			glog.Warningf("Unable to retrieve a secret %s/%s: %v", key.namespace, key.name, err)
		}
		results = append(results, result{secret: secret, err: err})
	}

	s.lock.Lock()
	defer s.lock.Unlock()
	for i, key := range keys {
		secret := results[i].secret
		err := results[i].err
		if err != nil && !apierrors.IsNotFound(err) {
			// If we couldn't retrieve a secret and it wasn't 404 error, skip updating.
			continue
		}
		if item, ok := s.items[key]; ok {
			if secret != nil && item.secret != nil {
				// If the fetched version is older than the current one (such races are
				// possible), then skip update.
				if secret.ResourceVersion < item.secret.ResourceVersion {
					continue
				}
			}
			item.secret = secret
			item.err = err
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

	lock           sync.Mutex
	registeredPods map[objectKey]*api.Pod
}

func newCachingSecretManager(kubeClient clientset.Interface) (secretManager, error) {
	return &cachingSecretManager{
		secretStore:    newSecretStore(kubeClient),
		registeredPods: make(map[objectKey]*api.Pod),
	}, nil
}

func (c *cachingSecretManager) GetSecret(namespace, name string) (*api.Secret, error) {
	return c.secretStore.Get(namespace, name)
}

// TODO: Before we will use secretManager in other places (e.g. for secret volumes)
// we should update this function to also get secrets from those places.
func getSecretKeys(pod *api.Pod) []objectKey {
	var secretKeys []objectKey
	for _, reference := range pod.Spec.ImagePullSecrets {
		secretKeys = append(secretKeys, objectKey{namespace: pod.Namespace, name: reference.Name})
	}
	for i := range pod.Spec.Containers {
		for _, envVar := range pod.Spec.Containers[i].Env {
			if envVar.ValueFrom != nil && envVar.ValueFrom.SecretKeyRef != nil {
				secretKeys = append(secretKeys, objectKey{namespace: pod.Namespace, name: envVar.ValueFrom.SecretKeyRef.Name})
			}
		}
	}
	return secretKeys
}

func (c *cachingSecretManager) RegisterPod(pod *api.Pod) {
	for _, key := range getSecretKeys(pod) {
		c.secretStore.Add(key.namespace, key.name)
	}
	var prev *api.Pod
	func() {
		key := objectKey{namespace: pod.Namespace, name: pod.Name}
		c.lock.Lock()
		defer c.lock.Unlock()
		prev = c.registeredPods[key]
		c.registeredPods[key] = pod
	}()
	if prev != nil {
		for _, key := range getSecretKeys(pod) {
			c.secretStore.Delete(key.namespace, key.name)
		}
	}
}

func (c *cachingSecretManager) UnregisterPod(pod *api.Pod) {
	var prev *api.Pod
	func() {
		key := objectKey{namespace: pod.Namespace, name: pod.Name}
		c.lock.Lock()
		defer c.lock.Unlock()
		prev = c.registeredPods[key]
		delete(c.registeredPods, key)
	}()
	if prev != nil {
		for _, key := range getSecretKeys(pod) {
			c.secretStore.Delete(key.namespace, key.name)
		}
	}
}
