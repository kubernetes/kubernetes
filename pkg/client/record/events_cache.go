/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package record

import (
	"sync"

	"github.com/golang/groupcache/lru"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

type history struct {
	// The number of times the event has occurred since first occurrence.
	Count int

	// The time at which the event was first recorded.
	FirstTimestamp unversioned.Time

	// The unique name of the first occurrence of this event
	Name string

	// Resource version returned from previous interaction with server
	ResourceVersion string
}

const (
	maxLruCacheEntries = 4096
)

type historyCache struct {
	sync.RWMutex
	cache *lru.Cache
}

func NewEventCache() *historyCache {
	return &historyCache{cache: lru.New(maxLruCacheEntries)}
}

// addOrUpdateEvent creates a new entry for the given event in the previous events hash table if the event
// doesn't already exist, otherwise it updates the existing entry.
func (eventCache *historyCache) addOrUpdateEvent(newEvent *api.Event) history {
	key := getEventKey(newEvent)
	eventCache.Lock()
	defer eventCache.Unlock()
	eventCache.cache.Add(
		key,
		history{
			Count:           newEvent.Count,
			FirstTimestamp:  newEvent.FirstTimestamp,
			Name:            newEvent.Name,
			ResourceVersion: newEvent.ResourceVersion,
		})
	return eventCache.getEventFromCache(key)
}

// getEvent returns the entry corresponding to the given event, if one exists, otherwise a history object
// with a count of 0 is returned.
func (eventCache *historyCache) getEvent(event *api.Event) history {
	key := getEventKey(event)
	eventCache.RLock()
	defer eventCache.RUnlock()
	return eventCache.getEventFromCache(key)
}

func (eventCache *historyCache) getEventFromCache(key string) history {
	value, ok := eventCache.cache.Get(key)
	if ok {
		historyValue, ok := value.(history)
		if ok {
			return historyValue
		}
	}
	return history{}
}

func getEventKey(event *api.Event) string {
	return event.Source.Component +
		event.Source.Host +
		event.InvolvedObject.Kind +
		event.InvolvedObject.Namespace +
		event.InvolvedObject.Name +
		string(event.InvolvedObject.UID) +
		event.InvolvedObject.APIVersion +
		event.Reason +
		event.Message
}
