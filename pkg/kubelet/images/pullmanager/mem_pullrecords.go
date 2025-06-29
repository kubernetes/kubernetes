/*
Copyright 2025 The Kubernetes Authors.

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

package pullmanager

import (
	"strings"
	"sync"

	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/utils/lru"
)

type LRUCache[K comparable, V any] struct {
	cache *lru.Cache
}

func NewLRUCache[K comparable, V any](size int) *LRUCache[K, V] {
	return &LRUCache[K, V]{
		cache: lru.New(size),
	}
}

func (c *LRUCache[K, V]) Get(key K) (*V, bool) {
	value, found := c.cache.Get(key)
	if !found {
		return nil, false
	}
	if value == nil {
		return nil, true
	}
	return value.(*V), true
}

func (c *LRUCache[K, V]) Set(key K, value *V) { c.cache.Add(key, value) }
func (c *LRUCache[K, V]) Delete(key K)        { c.cache.Remove(key) }
func (c *LRUCache[K, V]) Len() int            { return c.cache.Len() }

// cachedPullRecordsAccessor implements a write-through cache layer on top
// of another PullRecordsAccessor
type cachedPullRecordsAccessor struct {
	delegate PullRecordsAccessor

	intentsMutex       *sync.RWMutex
	intents            *LRUCache[string, kubeletconfiginternal.ImagePullIntent]
	pulledRecordsMutex *sync.RWMutex
	pulledRecords      *LRUCache[string, kubeletconfiginternal.ImagePulledRecord]
}

func NewCachedPullRecordsAccessor(delegate PullRecordsAccessor) *cachedPullRecordsAccessor {
	// TODO: handle errors from records listing here and below
	coldIntents, _ := delegate.ListImagePullIntents()
	warmIntents := NewLRUCache[string, kubeletconfiginternal.ImagePullIntent](50)
	for _, intent := range coldIntents[:min(50, len(coldIntents))] {
		warmIntents.Set(intent.Image, intent)
	}

	coldPulledRecords, _ := delegate.ListImagePulledRecords()
	warmPulledRecords := NewLRUCache[string, kubeletconfiginternal.ImagePulledRecord](100)
	for _, pulledRecord := range coldPulledRecords[:min(100, len(coldPulledRecords))] {
		warmPulledRecords.Set(pulledRecord.ImageRef, pulledRecord)
	}

	return &cachedPullRecordsAccessor{
		delegate: delegate,

		intentsMutex:       &sync.RWMutex{},
		intents:            warmIntents,
		pulledRecordsMutex: &sync.RWMutex{},
		pulledRecords:      warmPulledRecords,
	}
}

func (c *cachedPullRecordsAccessor) ListImagePullIntents() ([]*kubeletconfiginternal.ImagePullIntent, error) {
	//TODO: maybe this should be protected by a read lock
	c.intentsMutex.RLock()
	defer c.intentsMutex.RUnlock()
	return c.delegate.ListImagePullIntents()
}

func (c *cachedPullRecordsAccessor) ImagePullIntentExists(image string) (bool, error) {
	if _, exists := c.intents.Get(image); exists {
		return true, nil
	}

	c.intentsMutex.RLock()
	defer c.intentsMutex.RUnlock()

	exists, err := c.delegate.ImagePullIntentExists(image)
	if err == nil && exists {
		c.intents.Set(image, &kubeletconfiginternal.ImagePullIntent{
			Image: image,
		})
	}
	return exists, err
}

func (c *cachedPullRecordsAccessor) WriteImagePullIntent(image string) error {
	c.intentsMutex.Lock()
	defer c.intentsMutex.Unlock()

	if err := c.delegate.WriteImagePullIntent(image); err != nil {
		return err
	}

	c.intents.Set(image, &kubeletconfiginternal.ImagePullIntent{
		Image: image,
	})
	return nil
}

func (c *cachedPullRecordsAccessor) DeleteImagePullIntent(image string) error {
	c.intentsMutex.Lock()
	defer c.intentsMutex.Unlock()

	if err := c.delegate.DeleteImagePullIntent(image); err != nil {
		return err
	}
	c.intents.Delete(image)
	return nil
}

func (c *cachedPullRecordsAccessor) ListImagePulledRecords() ([]*kubeletconfiginternal.ImagePulledRecord, error) {
	//TODO: maybe this should be protected by a read lock
	c.pulledRecordsMutex.RLock()
	defer c.pulledRecordsMutex.RUnlock()
	return c.delegate.ListImagePulledRecords()
}

func (c *cachedPullRecordsAccessor) GetImagePulledRecord(imageRef string) (*kubeletconfiginternal.ImagePulledRecord, bool, error) {
	pulledRecord, exists := c.pulledRecords.Get(imageRef)
	if exists {
		return pulledRecord, true, nil
	}

	c.pulledRecordsMutex.RLock()
	defer c.pulledRecordsMutex.RUnlock()

	pulledRecord, exists, err := c.delegate.GetImagePulledRecord(imageRef)
	if err == nil && exists && pulledRecord != nil {
		c.pulledRecords.Set(imageRef, pulledRecord)
	}
	return pulledRecord, exists, err
}

func (c *cachedPullRecordsAccessor) WriteImagePulledRecord(record *kubeletconfiginternal.ImagePulledRecord) error {
	c.pulledRecordsMutex.Lock()
	defer c.pulledRecordsMutex.Unlock()

	if err := c.delegate.WriteImagePulledRecord(record); err != nil {
		return err
	}
	c.pulledRecords.Set(record.ImageRef, record)
	return nil
}

func (c *cachedPullRecordsAccessor) DeleteImagePulledRecord(imageRef string) error {
	c.pulledRecordsMutex.Lock()
	defer c.pulledRecordsMutex.Unlock()

	if err := c.delegate.DeleteImagePulledRecord(imageRef); err != nil {
		return err
	}
	c.pulledRecords.Delete(imageRef)
	return nil
}

func pullIntentsCmp(a, b *kubeletconfiginternal.ImagePullIntent) int {
	return strings.Compare(a.Image, b.Image)
}
func pulledRecordsCmp(a, b *kubeletconfiginternal.ImagePulledRecord) int {
	return strings.Compare(a.ImageRef, b.ImageRef)
}
