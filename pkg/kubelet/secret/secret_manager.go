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

package secret

import (
	"sync"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

type Manager interface {
	// Get secret by secret namespace and name.
	GetSecret(namespace, name string) (*v1.Secret, error)

	// WARNING: Register/UnregisterPod functions should be efficient,
	// i.e. should not block on network operations.

	// RegisterPod registers all secrets from a given pod.
	RegisterPod(pod *v1.Pod)

	// UnregisterPod unregisters secrets from a given pod that are not
	// used by any other registered pod.
	UnregisterPod(pod *v1.Pod)
}

type objectKey struct {
	namespace string
	name      string
}

// simpleSecretManager implements SecretManager interfaces with
// simple operations to apiserver.
type simpleSecretManager struct {
	kubeClient clientset.Interface
}

func NewSimpleSecretManager(kubeClient clientset.Interface) Manager {
	return &simpleSecretManager{kubeClient: kubeClient}
}

func (s *simpleSecretManager) GetSecret(namespace, name string) (*v1.Secret, error) {
	return s.kubeClient.CoreV1().Secrets(namespace).Get(name, metav1.GetOptions{})
}

func (s *simpleSecretManager) RegisterPod(pod *v1.Pod) {
}

func (s *simpleSecretManager) UnregisterPod(pod *v1.Pod) {
}

// store is the interface for a secrets cache that
// can be used by cacheBasedSecretManager.
type store interface {
	// Add a secret to the store.
	// Note that multiple additions to the store has to be allowed
	// in the implementations and effectively treated as refcounted.
	Add(namespace, name string)
	// Delete a secret from the store.
	// Note that secret should be deleted only when there was a
	// corresponding Delete call for each of Add calls (effectively
	// when refcount was reduced to zero).
	Delete(namespace, name string)
	// Get a secret from a store.
	Get(namespace, name string) (*v1.Secret, error)
}

// cachingBasedSecretManager keeps a store with secrets necessary
// for registered pods. Different implementations of the store
// may result in different semantics for freshness of secrets
// (e.g. ttl-based implementation vs watch-based implementation).
type cacheBasedSecretManager struct {
	secretStore store

	lock           sync.Mutex
	registeredPods map[objectKey]*v1.Pod
}

func newCacheBasedSecretManager(secretStore store) Manager {
	return &cacheBasedSecretManager{
		secretStore:    secretStore,
		registeredPods: make(map[objectKey]*v1.Pod),
	}
}

func (c *cacheBasedSecretManager) GetSecret(namespace, name string) (*v1.Secret, error) {
	return c.secretStore.Get(namespace, name)
}

func getSecretNames(pod *v1.Pod) sets.String {
	result := sets.NewString()
	podutil.VisitPodSecretNames(pod, func(name string) bool {
		result.Insert(name)
		return true
	})
	return result
}

func (c *cacheBasedSecretManager) RegisterPod(pod *v1.Pod) {
	names := getSecretNames(pod)
	c.lock.Lock()
	defer c.lock.Unlock()
	for name := range names {
		c.secretStore.Add(pod.Namespace, name)
	}
	var prev *v1.Pod
	key := objectKey{namespace: pod.Namespace, name: pod.Name}
	prev = c.registeredPods[key]
	c.registeredPods[key] = pod
	if prev != nil {
		for name := range getSecretNames(prev) {
			// On an update, the .Add() call above will have re-incremented the
			// ref count of any existing secrets, so any secrets that are in both
			// names and prev need to have their ref counts decremented. Any that
			// are only in prev need to be completely removed. This unconditional
			// call takes care of both cases.
			c.secretStore.Delete(prev.Namespace, name)
		}
	}
}

func (c *cacheBasedSecretManager) UnregisterPod(pod *v1.Pod) {
	var prev *v1.Pod
	key := objectKey{namespace: pod.Namespace, name: pod.Name}
	c.lock.Lock()
	defer c.lock.Unlock()
	prev = c.registeredPods[key]
	delete(c.registeredPods, key)
	if prev != nil {
		for name := range getSecretNames(prev) {
			c.secretStore.Delete(prev.Namespace, name)
		}
	}
}
