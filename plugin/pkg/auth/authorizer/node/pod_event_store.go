/*
Copyright 2026 The Kubernetes Authors.

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

package node

import (
	"fmt"
	"sync"
	"sync/atomic"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
)

// podEventStore feeds Pod watch events into a graphPopulator without
// retaining full Pod objects. Per pod it keeps only a fingerprint: the
// handful of fields graphPopulator.updatePod compares to decide whether a
// change is relevant to the node authorization graph, reused unmodified so
// this doesn't drift from that tested logic.
type podEventStore struct {
	populator *graphPopulator

	mu    sync.Mutex
	known map[string]*corev1.Pod

	hasSynced atomic.Bool
}

func newPodEventStore(populator *graphPopulator) *podEventStore {
	return &podEventStore{
		populator: populator,
		known:     map[string]*corev1.Pod{},
	}
}

func podFingerprint(pod *corev1.Pod) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			UID:       pod.UID,
		},
		Spec: corev1.PodSpec{
			NodeName:            pod.Spec.NodeName,
			EphemeralContainers: make([]corev1.EphemeralContainer, len(pod.Spec.EphemeralContainers)),
		},
		Status: corev1.PodStatus{
			ResourceClaimStatuses:       pod.Status.ResourceClaimStatuses,
			ExtendedResourceClaimStatus: pod.Status.ExtendedResourceClaimStatus,
		},
	}
}

func (s *podEventStore) HasSynced() bool {
	return s.hasSynced.Load()
}

func (s *podEventStore) Add(obj interface{}) error {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return fmt.Errorf("podEventStore.Add: unexpected type %T", obj)
	}
	key, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.known[key] = podFingerprint(pod)
	s.mu.Unlock()
	s.populator.addPod(pod)
	return nil
}

func (s *podEventStore) Update(obj interface{}) error {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return fmt.Errorf("podEventStore.Update: unexpected type %T", obj)
	}
	key, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		return err
	}
	s.mu.Lock()
	oldFingerprint := s.known[key]
	s.known[key] = podFingerprint(pod)
	s.mu.Unlock()
	if oldFingerprint != nil {
		s.populator.updatePod(oldFingerprint, pod)
	} else {
		s.populator.addPod(pod)
	}
	return nil
}

func (s *podEventStore) Delete(obj interface{}) error {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return fmt.Errorf("podEventStore.Delete: unexpected type %T", obj)
	}
	key, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		return err
	}
	s.mu.Lock()
	delete(s.known, key)
	s.mu.Unlock()
	s.populator.deletePod(pod)
	return nil
}

// Replace handles both the reflector's initial List and any relist after a
// watch reconnects. Deletions that happened while disconnected are detected
// by diffing against the fingerprints kept from before the relist, the same
// way a full object store would, just against fingerprints instead of full
// pods.
func (s *podEventStore) Replace(list []interface{}, _ string) error {
	s.mu.Lock()
	oldKnown := s.known
	s.mu.Unlock()

	newKnown := make(map[string]*corev1.Pod, len(list))
	for _, item := range list {
		pod, ok := item.(*corev1.Pod)
		if !ok {
			continue
		}
		key, err := cache.MetaNamespaceKeyFunc(pod)
		if err != nil {
			continue
		}
		if oldFingerprint, existed := oldKnown[key]; existed {
			s.populator.updatePod(oldFingerprint, pod)
		} else {
			s.populator.addPod(pod)
		}
		newKnown[key] = podFingerprint(pod)
	}

	for key, oldFingerprint := range oldKnown {
		if _, stillPresent := newKnown[key]; stillPresent {
			continue
		}
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			continue
		}
		s.populator.deletePod(&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
			Spec:       corev1.PodSpec{NodeName: oldFingerprint.Spec.NodeName},
		})
	}

	s.mu.Lock()
	s.known = newKnown
	s.mu.Unlock()
	s.hasSynced.Store(true)
	return nil
}

// The methods below aren't used on Reflector's write path but are required
// to satisfy cache.Store.

func (s *podEventStore) List() []interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	items := make([]interface{}, 0, len(s.known))
	for _, fp := range s.known {
		items = append(items, fp)
	}
	return items
}

func (s *podEventStore) ListKeys() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	keys := make([]string, 0, len(s.known))
	for k := range s.known {
		keys = append(keys, k)
	}
	return keys
}

func (s *podEventStore) Get(obj interface{}) (item interface{}, exists bool, err error) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return nil, false, fmt.Errorf("podEventStore.Get: unexpected type %T", obj)
	}
	key, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		return nil, false, err
	}
	return s.GetByKey(key)
}

func (s *podEventStore) GetByKey(key string) (item interface{}, exists bool, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fp, ok := s.known[key]
	return fp, ok, nil
}

func (s *podEventStore) Resync() error {
	return nil
}

func (s *podEventStore) LastStoreSyncResourceVersion() string {
	return ""
}

func (s *podEventStore) Bookmark(_ string) {}

var _ cache.Store = &podEventStore{}
