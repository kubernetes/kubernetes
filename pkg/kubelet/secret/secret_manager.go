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
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	storageetcd "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/util/sets"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

// simpleSecretManager implements SecretManager interfaces with
// simple operations to apiserver.
type simpleSecretManager struct {
	kubeClient clientset.Interface
}

func NewSimpleSecretManager(kubeClient clientset.Interface) (Manager, error) {
	return &simpleSecretManager{kubeClient: kubeClient}, nil
}

func (s *simpleSecretManager) GetSecret(namespace, name string) (*v1.Secret, error) {
	return s.kubeClient.Core().Secrets(namespace).Get(name, metav1.GetOptions{})
}

func (s *simpleSecretManager) RegisterPod(pod *v1.Pod) {
}

func (s *simpleSecretManager) UnregisterPod(pod *v1.Pod) {
}

type objectKey struct {
	namespace string
	name      string
}

// secretStoreItems is a single item stored in secretStore.
type secretStoreItem struct {
	refCount int
	secret   *secretData
}

type secretData struct {
	sync.Mutex
	secret         *v1.Secret
	err            error
	lastUpdateTime time.Time
}

/*func (s *secretData) initialized() bool {
	s.Lock()
	defer s.Unlock()
	return s.secret != nil || s.err != nil
}*/

// update assumes that the secret and error passed to this function
// we really want to store (e.g. it is not a transient error).
// Thus, the only case when we want to ignore it is when the passed
// secret is older then the one already stored.
func (s *secretData) update(secret *v1.Secret, err error, timestamp time.Time) {
	s.Lock()
	defer s.Unlock()
	if s.secret != nil && secret != nil && isSecretOlder(secret, s.secret) {
		return
	}
	s.secret = secret
	s.err = err
	s.lastUpdateTime = timestamp
}

// secretStore is a local cache of secrets.
type secretStore struct {
	kubeClient clientset.Interface

	lock  sync.Mutex
	items map[objectKey]*secretStoreItem
	ttl   time.Duration
}

func newSecretStore(kubeClient clientset.Interface, ttl time.Duration) *secretStore {
	return &secretStore{
		kubeClient: kubeClient,
		items:      make(map[objectKey]*secretStoreItem),
		ttl:        ttl,
	}
}

func isSecretOlder(newSecret, oldSecret *v1.Secret) bool {
	if newSecret != nil && oldSecret != nil {
		newVersion, _ := storageetcd.Versioner.ObjectResourceVersion(newSecret)
		oldVersion, _ := storageetcd.Versioner.ObjectResourceVersion(oldSecret)
		if newVersion >= oldVersion {
			return false
		}
	}
	return true
}

func (s *secretStore) Add(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	// Add is called from RegisterPod, thus it needs to be efficient.
	// Thus Add() is only increasing refCount and generation of a given secret.
	// Then Get() is responsible for fetching if needed.
	s.lock.Lock()
	defer s.lock.Unlock()
	item, exists := s.items[key]
	if !exists {
		item = &secretStoreItem{
			refCount: 0,
			secret:   &secretData{},
		}
		s.items[key] = item
	}

	item.refCount++
	// This should trigger fetch on the next Get() operation.
	item.secret.update(nil, nil, time.Now().Add(-s.ttl))
}

func (s *secretStore) Delete(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	s.lock.Lock()
	defer s.lock.Unlock()
	if item, ok := s.items[key]; ok {
		item.refCount--
		// TODO: Do we need to trigger fetch on all deletions?
		if item.refCount == 0 {
			delete(s.items, key)
		}
	}
}

func (s *secretStore) Get(namespace, name string) (*v1.Secret, error) {
	key := objectKey{namespace: namespace, name: name}

	item, shouldFetch := func() (*secretData, bool) {
		s.lock.Lock()
		defer s.lock.Unlock()
		if item := s.items[key]; item != nil {
			data := item.secret
			data.Lock()
			defer data.Unlock()
			return data, data.err != nil || time.Now().After(data.lastUpdateTime.Add(s.ttl))
		}
		return nil, false
	}()
	if item == nil {
		return nil, fmt.Errorf("secret %q/%q not registered", namespace, name)
	}

	if shouldFetch {
		secret, err := s.kubeClient.Core().Secrets(namespace).Get(name, metav1.GetOptions{})
		// Update state, unless we got error different than "not-found".
		if err == nil || apierrors.IsNotFound(err) {
			item.update(secret, err, time.Now())
		}
	}

	item.Lock()
	defer item.Unlock()
	return item.secret, item.err
}

// cachingSecretManager keeps a cache of all secrets necessary for registered pods.
// It implements the following logic:
// - whenever a pod is created or updated, the cached versions of all its secrets
//   are invalidated
// - every GetSecret() call tries to fetch the value from local cache; if it is
//   not there, invalidated or too old, we fetch it from apiserver and refresh the
//   value in cache; otherwise it is just fetched from cache
type cachingSecretManager struct {
	secretStore *secretStore

	lock           sync.Mutex
	registeredPods map[objectKey]*v1.Pod
}

func NewCachingSecretManager(kubeClient clientset.Interface) (Manager, error) {
	csm := &cachingSecretManager{
		secretStore:    newSecretStore(kubeClient, time.Minute),
		registeredPods: make(map[objectKey]*v1.Pod),
	}
	return csm, nil
}

func (c *cachingSecretManager) GetSecret(namespace, name string) (*v1.Secret, error) {
	return c.secretStore.Get(namespace, name)
}

// TODO: Before we will use secretManager in other places (e.g. for secret volumes)
// we should update this function to also get secrets from those places.
func getSecretNames(pod *v1.Pod) sets.String {
	result := sets.NewString()
	for _, reference := range pod.Spec.ImagePullSecrets {
		result.Insert(reference.Name)
	}
	for i := range pod.Spec.Containers {
		for _, envVar := range pod.Spec.Containers[i].Env {
			if envVar.ValueFrom != nil && envVar.ValueFrom.SecretKeyRef != nil {
				result.Insert(envVar.ValueFrom.SecretKeyRef.Name)
			}
		}
	}
	return result
}

func (c *cachingSecretManager) RegisterPod(pod *v1.Pod) {
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
			c.secretStore.Delete(prev.Namespace, name)
		}
	}
}

func (c *cachingSecretManager) UnregisterPod(pod *v1.Pod) {
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
