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
	"strconv"
	"sync"
	"time"

	storageetcd "k8s.io/apiserver/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/api/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/kubelet/util"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	defaultTTL = time.Minute
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

func NewSimpleSecretManager(kubeClient clientset.Interface) Manager {
	return &simpleSecretManager{kubeClient: kubeClient}
}

func (s *simpleSecretManager) GetSecret(namespace, name string) (*v1.Secret, error) {
	return s.kubeClient.Core().Secrets(namespace).Get(name, metav1.GetOptions{})
}

func (s *simpleSecretManager) RegisterPod(pod *v1.Pod) {
}

func (s *simpleSecretManager) UnregisterPod(pod *v1.Pod) {
}

type GetObjectTTLFunc func() (time.Duration, bool)

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

// secretStore is a local cache of secrets.
type secretStore struct {
	kubeClient clientset.Interface
	clock      clock.Clock

	lock  sync.Mutex
	items map[objectKey]*secretStoreItem

	defaultTTL time.Duration
	getTTL     GetObjectTTLFunc
}

func newSecretStore(kubeClient clientset.Interface, clock clock.Clock, getTTL GetObjectTTLFunc, ttl time.Duration) *secretStore {
	return &secretStore{
		kubeClient: kubeClient,
		clock:      clock,
		items:      make(map[objectKey]*secretStoreItem),
		defaultTTL: ttl,
		getTTL:     getTTL,
	}
}

func isSecretOlder(newSecret, oldSecret *v1.Secret) bool {
	if newSecret == nil || oldSecret == nil {
		return false
	}
	newVersion, _ := storageetcd.Versioner.ObjectResourceVersion(newSecret)
	oldVersion, _ := storageetcd.Versioner.ObjectResourceVersion(oldSecret)
	return newVersion < oldVersion
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
	// This will trigger fetch on the next Get() operation.
	item.secret = nil
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

func GetObjectTTLFromNodeFunc(getNode func() (*v1.Node, error)) GetObjectTTLFunc {
	return func() (time.Duration, bool) {
		node, err := getNode()
		if err != nil {
			return time.Duration(0), false
		}
		if node != nil && node.Annotations != nil {
			if value, ok := node.Annotations[v1.ObjectTTLAnnotationKey]; ok {
				if intValue, err := strconv.Atoi(value); err == nil {
					return time.Duration(intValue) * time.Second, true
				}
			}
		}
		return time.Duration(0), false
	}
}

func (s *secretStore) isSecretFresh(data *secretData) bool {
	secretTTL := s.defaultTTL
	if ttl, ok := s.getTTL(); ok {
		secretTTL = ttl
	}
	return s.clock.Now().Before(data.lastUpdateTime.Add(secretTTL))
}

func (s *secretStore) Get(namespace, name string) (*v1.Secret, error) {
	key := objectKey{namespace: namespace, name: name}

	data := func() *secretData {
		s.lock.Lock()
		defer s.lock.Unlock()
		item, exists := s.items[key]
		if !exists {
			return nil
		}
		if item.secret == nil {
			item.secret = &secretData{}
		}
		return item.secret
	}()
	if data == nil {
		return nil, fmt.Errorf("secret %q/%q not registered", namespace, name)
	}

	// After updating data in secretStore, lock the data, fetch secret if
	// needed and return data.
	data.Lock()
	defer data.Unlock()
	if data.err != nil || !s.isSecretFresh(data) {
		opts := metav1.GetOptions{}
		if data.secret != nil && data.err == nil {
			// This is just a periodic refresh of a secret we successfully fetched previously.
			// In this case, server data from apiserver cache to reduce the load on both
			// etcd and apiserver (the cache is eventually consistent).
			util.FromApiserverCache(&opts)
		}
		secret, err := s.kubeClient.Core().Secrets(namespace).Get(name, opts)
		if err != nil && !apierrors.IsNotFound(err) && data.secret == nil && data.err == nil {
			// Couldn't fetch the latest secret, but there is no cached data to return.
			// Return the fetch result instead.
			return secret, err
		}
		if (err == nil && !isSecretOlder(secret, data.secret)) || apierrors.IsNotFound(err) {
			// If the fetch succeeded with a newer version of the secret, or if the
			// secret could not be found in the apiserver, update the cached data to
			// reflect the current status.
			data.secret = secret
			data.err = err
			data.lastUpdateTime = s.clock.Now()
		}
	}
	return data.secret, data.err
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

func NewCachingSecretManager(kubeClient clientset.Interface, getTTL GetObjectTTLFunc) Manager {
	csm := &cachingSecretManager{
		secretStore:    newSecretStore(kubeClient, clock.RealClock{}, getTTL, defaultTTL),
		registeredPods: make(map[objectKey]*v1.Pod),
	}
	return csm
}

func (c *cachingSecretManager) GetSecret(namespace, name string) (*v1.Secret, error) {
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
