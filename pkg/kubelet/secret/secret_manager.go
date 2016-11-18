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

	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	storageetcd "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
)

type Manager interface {
	// Get secret by secret namespace and name.
	GetSecret(namespace, name string) (*v1.Secret, error)

	// RegisterPod registers all secrets from a given pod.
	RegisterPod(pod *v1.Pod)

	// UnregisterPod unregisters secrets from a given pod that are not
	// registered still by any other registered pod.
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
	secret   *v1.Secret
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
	return &secretStore{
		kubeClient: kubeClient,
		items:      make(map[objectKey]*secretStoreItem),
	}
}

func (s *secretStore) Add(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}
	secret, err := s.kubeClient.Core().Secrets(namespace).Get(name, metav1.GetOptions{})

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

func (s *secretStore) Get(namespace, name string) (*v1.Secret, error) {
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
		secret *v1.Secret
		err    error
	}
	results := make([]result, 0, len(keys))
	for _, key := range keys {
		secret, err := s.kubeClient.Core().Secrets(key.namespace).Get(key.name, metav1.GetOptions{})
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
				// If the fetched version is not newer than the current one (such races are
				// possible), then skip update.
				newVersion, _ := storageetcd.Versioner.ObjectResourceVersion(secret)
				oldVersion, _ := storageetcd.Versioner.ObjectResourceVersion(item.secret)
				if newVersion <= oldVersion {
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
	registeredPods map[objectKey]*v1.Pod
}

func NewCachingSecretManager(kubeClient clientset.Interface) (Manager, error) {
	csm := &cachingSecretManager{
		secretStore:    newSecretStore(kubeClient),
		registeredPods: make(map[objectKey]*v1.Pod),
	}
	go wait.NonSlidingUntil(func() { csm.secretStore.Refresh() }, time.Minute, wait.NeverStop)
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
	for key := range getSecretNames(pod) {
		c.secretStore.Add(pod.Namespace, key)
	}
	var prev *v1.Pod
	func() {
		key := objectKey{namespace: pod.Namespace, name: pod.Name}
		c.lock.Lock()
		defer c.lock.Unlock()
		prev = c.registeredPods[key]
		c.registeredPods[key] = pod
	}()
	if prev != nil {
		for key := range getSecretNames(prev) {
			c.secretStore.Delete(prev.Namespace, key)
		}
	}
}

func (c *cachingSecretManager) UnregisterPod(pod *v1.Pod) {
	var prev *v1.Pod
	func() {
		key := objectKey{namespace: pod.Namespace, name: pod.Name}
		c.lock.Lock()
		defer c.lock.Unlock()
		prev = c.registeredPods[key]
		delete(c.registeredPods, key)
	}()
	if prev != nil {
		for key := range getSecretNames(prev) {
			c.secretStore.Delete(prev.Namespace, key)
		}
	}
}
