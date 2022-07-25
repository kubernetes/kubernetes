//go:build !providerless
// +build !providerless

/*
Copyright 2019 The Kubernetes Authors.

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

package azure

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/klog/v2"
)

const (
	tagsDelimiter        = ","
	tagKeyValueDelimiter = "="
)

// lockMap used to lock on entries
type lockMap struct {
	sync.Mutex
	mutexMap map[string]*sync.Mutex
}

// NewLockMap returns a new lock map
func newLockMap() *lockMap {
	return &lockMap{
		mutexMap: make(map[string]*sync.Mutex),
	}
}

// LockEntry acquires a lock associated with the specific entry
func (lm *lockMap) LockEntry(entry string) {
	lm.Lock()
	// check if entry does not exists, then add entry
	if _, exists := lm.mutexMap[entry]; !exists {
		lm.addEntry(entry)
	}

	lm.Unlock()
	lm.lockEntry(entry)
}

// UnlockEntry release the lock associated with the specific entry
func (lm *lockMap) UnlockEntry(entry string) {
	lm.Lock()
	defer lm.Unlock()

	if _, exists := lm.mutexMap[entry]; !exists {
		return
	}
	lm.unlockEntry(entry)
}

func (lm *lockMap) addEntry(entry string) {
	lm.mutexMap[entry] = &sync.Mutex{}
}

func (lm *lockMap) lockEntry(entry string) {
	lm.mutexMap[entry].Lock()
}

func (lm *lockMap) unlockEntry(entry string) {
	lm.mutexMap[entry].Unlock()
}

func getContextWithCancel() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

// ConvertTagsToMap convert the tags from string to map
// the valid tags format is "key1=value1,key2=value2", which could be converted to
// {"key1": "value1", "key2": "value2"}
func ConvertTagsToMap(tags string) (map[string]string, error) {
	m := make(map[string]string)
	if tags == "" {
		return m, nil
	}
	s := strings.Split(tags, tagsDelimiter)
	for _, tag := range s {
		kv := strings.Split(tag, tagKeyValueDelimiter)
		if len(kv) != 2 {
			return nil, fmt.Errorf("Tags '%s' are invalid, the format should like: 'key1=value1,key2=value2'", tags)
		}
		key := strings.TrimSpace(kv[0])
		if key == "" {
			return nil, fmt.Errorf("Tags '%s' are invalid, the format should like: 'key1=value1,key2=value2'", tags)
		}
		value := strings.TrimSpace(kv[1])
		m[key] = value
	}

	return m, nil
}

func convertMapToMapPointer(origin map[string]string) map[string]*string {
	newly := make(map[string]*string)
	for k, v := range origin {
		value := v
		newly[k] = &value
	}
	return newly
}

func parseTags(tags string) map[string]*string {
	kvs := strings.Split(tags, ",")
	formatted := make(map[string]*string)
	for _, kv := range kvs {
		res := strings.Split(kv, "=")
		if len(res) != 2 {
			klog.Warningf("parseTags: error when parsing key-value pair %s, would ignore this one", kv)
			continue
		}
		k, v := strings.TrimSpace(res[0]), strings.TrimSpace(res[1])
		if k == "" || v == "" {
			klog.Warningf("parseTags: error when parsing key-value pair %s-%s, would ignore this one", k, v)
			continue
		}
		formatted[strings.ToLower(k)] = to.StringPtr(v)
	}
	return formatted
}

func findKeyInMapCaseInsensitive(targetMap map[string]*string, key string) (bool, string) {
	for k := range targetMap {
		if strings.EqualFold(k, key) {
			return true, k
		}
	}

	return false, ""
}

func reconcileTags(currentTagsOnResource, newTags map[string]*string) (reconciledTags map[string]*string, changed bool) {
	for k, v := range newTags {
		found, key := findKeyInMapCaseInsensitive(currentTagsOnResource, k)
		if !found {
			currentTagsOnResource[k] = v
			changed = true
		} else if !strings.EqualFold(to.String(v), to.String(currentTagsOnResource[key])) {
			currentTagsOnResource[key] = v
			changed = true
		}
	}

	return currentTagsOnResource, changed
}
