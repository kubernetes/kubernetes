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
	"slices"
	"strings"
	"sync"

	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

// cachedPullRecordsAccessor implements a write-through cache layer on top
// of another PullRecordsAccessor
type cachedPullRecordsAccessor struct {
	delegate PullRecordsAccessor

	intentsMutex    *sync.RWMutex
	intents         map[string]*kubeletconfiginternal.ImagePullIntent
	intentListError error

	pulledRecordsMutex     *sync.RWMutex
	pulledRecords          map[string]*kubeletconfiginternal.ImagePulledRecord
	pulledRecordsListError error
}

func NewCachedPullRecordsAccessor(delegate PullRecordsAccessor) *cachedPullRecordsAccessor {
	coldIntents, intentListError := delegate.ListImagePullIntents()
	cachedIntents := make(map[string]*kubeletconfiginternal.ImagePullIntent, len(coldIntents))
	for _, intent := range coldIntents {
		cachedIntents[intent.Image] = intent
	}

	coldPulledRecords, pulledRecordsListError := delegate.ListImagePulledRecords()
	cachedPulledRecords := make(map[string]*kubeletconfiginternal.ImagePulledRecord, len(coldPulledRecords))
	for _, pulledRecord := range coldPulledRecords {
		cachedPulledRecords[pulledRecord.ImageRef] = pulledRecord
	}

	return &cachedPullRecordsAccessor{
		delegate: delegate,

		intentsMutex:    &sync.RWMutex{},
		intents:         cachedIntents,
		intentListError: intentListError,

		pulledRecordsMutex:     &sync.RWMutex{},
		pulledRecords:          cachedPulledRecords,
		pulledRecordsListError: pulledRecordsListError,
	}
}

func (c *cachedPullRecordsAccessor) ListImagePullIntents() ([]*kubeletconfiginternal.ImagePullIntent, error) {
	return genericList(
		c.intentsMutex,
		c.delegate.ListImagePullIntents,
		pullIntentToMapKey,
		pullIntentsCmp,
		&c.intentListError,
		&c.intents,
	)
}

func (c *cachedPullRecordsAccessor) ImagePullIntentExists(image string) (bool, error) {
	var exists bool
	func() {
		c.intentsMutex.RLock()
		defer c.intentsMutex.RUnlock()

		_, exists = c.intents[image]
	}()
	if exists {
		return true, nil
	}

	c.intentsMutex.Lock()
	defer c.intentsMutex.Unlock()

	exists, err := c.delegate.ImagePullIntentExists(image)
	if err == nil && exists {
		c.intents[image] = &kubeletconfiginternal.ImagePullIntent{
			Image: image,
		}
	}
	return exists, err
}

func (c *cachedPullRecordsAccessor) WriteImagePullIntent(image string) error {
	c.intentsMutex.Lock()
	defer c.intentsMutex.Unlock()

	if err := c.delegate.WriteImagePullIntent(image); err != nil {
		return err
	}
	c.intents[image] = &kubeletconfiginternal.ImagePullIntent{
		Image: image,
	}
	return nil
}

func (c *cachedPullRecordsAccessor) DeleteImagePullIntent(image string) error {
	c.intentsMutex.Lock()
	defer c.intentsMutex.Unlock()

	if err := c.delegate.DeleteImagePullIntent(image); err != nil {
		return err
	}
	delete(c.intents, image)
	return nil
}

func (c *cachedPullRecordsAccessor) ListImagePulledRecords() ([]*kubeletconfiginternal.ImagePulledRecord, error) {
	return genericList(
		c.pulledRecordsMutex,
		c.delegate.ListImagePulledRecords,
		pulledRecordToMapKey,
		pulledRecordsCmp,
		&c.pulledRecordsListError,
		&c.pulledRecords,
	)
}

func (c *cachedPullRecordsAccessor) GetImagePulledRecord(imageRef string) (*kubeletconfiginternal.ImagePulledRecord, bool, error) {
	var pulledRecord *kubeletconfiginternal.ImagePulledRecord
	var exists bool

	func() {
		c.pulledRecordsMutex.RLock()
		defer c.pulledRecordsMutex.RUnlock()

		pulledRecord, exists = c.pulledRecords[imageRef]
	}()
	if exists {
		return pulledRecord, true, nil
	}

	c.pulledRecordsMutex.Lock()
	defer c.pulledRecordsMutex.Unlock()

	pulledRecord, exists, err := c.delegate.GetImagePulledRecord(imageRef)
	if err == nil && exists && pulledRecord != nil {
		c.pulledRecords[imageRef] = pulledRecord
	}
	return pulledRecord, exists, err
}

func (c *cachedPullRecordsAccessor) WriteImagePulledRecord(record *kubeletconfiginternal.ImagePulledRecord) error {
	c.pulledRecordsMutex.Lock()
	defer c.pulledRecordsMutex.Unlock()

	if err := c.delegate.WriteImagePulledRecord(record); err != nil {
		return err
	}
	c.pulledRecords[record.ImageRef] = record
	return nil
}

func (c *cachedPullRecordsAccessor) DeleteImagePulledRecord(imageRef string) error {
	c.pulledRecordsMutex.Lock()
	defer c.pulledRecordsMutex.Unlock()

	if err := c.delegate.DeleteImagePulledRecord(imageRef); err != nil {
		return err
	}
	delete(c.pulledRecords, imageRef)
	return nil
}

func pullIntentToMapKey(i *kubeletconfiginternal.ImagePullIntent) string     { return i.Image }
func pulledRecordToMapKey(i *kubeletconfiginternal.ImagePulledRecord) string { return i.ImageRef }
func pullIntentsCmp(a, b *kubeletconfiginternal.ImagePullIntent) int {
	return strings.Compare(a.Image, b.Image)
}
func pulledRecordsCmp(a, b *kubeletconfiginternal.ImagePulledRecord) int {
	return strings.Compare(a.ImageRef, b.ImageRef)
}

func genericList[K comparable, V any](
	lock *sync.RWMutex,
	delegateList func() ([]V, error),
	toKey func(V) K,
	cmpFunc func(a, b V) int,
	previousError *error,
	cache *map[K]V,
) ([]V, error) {

	var ret []V = nil
	func() {
		lock.RLock()
		defer lock.RUnlock()

		// if there was an error during previous listing, we should retry delegateList() again
		if *previousError != nil {
			return
		}

		ret = make([]V, 0, len(*cache))
		for _, intent := range *cache {
			intent := intent
			ret = append(ret, intent)
		}

		// we're using an unstable sort since no two records should ever be the same
		slices.SortFunc(ret, cmpFunc)
	}()

	if ret != nil {
		return ret, nil
	}

	lock.Lock()
	defer lock.Unlock()

	objList, err := delegateList()
	newCache := make(map[K]V, len(objList))
	for _, item := range objList {
		newCache[toKey(item)] = item
	}
	*cache = newCache
	*previousError = err
	return objList, err
}
