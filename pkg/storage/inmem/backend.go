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

package inmem

import (
	"bytes"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/watch"
	"strings"
	"sync"
	"time"
)

type itemData struct {
	uid    types.UID
	data   []byte
	expiry uint64
	lsn    LSN
}

type Backend struct {
	log           *changeLog
	expiryManager *expiryManager

	mutex   sync.RWMutex
	root    *bucket
	lastLSN LSN
}

func NewBackend() *Backend {
	b := &Backend{
		root: newBucket(nil, ""),
	}

	b.log = newChangeLog()

	{
		init := &logEntry{lsn: 1}
		b.log.append(init)
		b.lastLSN = 1
	}
	b.expiryManager = newExpiryManager(b)
	go b.expiryManager.Run()

	return b
}

type bucket struct {
	parent   *bucket
	path     string
	children map[string]*bucket

	items map[string]itemData
}

func newBucket(parent *bucket, key string) *bucket {
	b := &bucket{
		parent:   parent,
		children: make(map[string]*bucket),
		items:    make(map[string]itemData),
	}
	if parent != nil {
		b.path = parent.path + key + "/"
	} else {
		b.path = key + "/"
	}
	return b

}

// We assume lock is held!
func (s *bucket) resolveBucket(path string, create bool) *bucket {
	// TODO: Easy to optimize (a lot!)
	path = strings.Trim(path, "/")
	if path == "" {
		return s
	}

	tokens := strings.Split(path, "/")
	b := s
	for _, t := range tokens {
		if t == "" {
			continue
		}
		child := b.children[t]
		if child == nil {
			if create {
				child = newBucket(b, t)
				b.children[t] = child
			} else {
				return nil
			}
		}
		b = child
	}
	return b
}

func splitPath(path string) (string, string) {
	// TODO: Easy to optimize (a lot!)
	path = strings.Trim(path, "/")

	lastSlash := strings.LastIndexByte(path, '/')
	if lastSlash == -1 {
		return "", path
	}
	item := path[lastSlash+1:]
	bucket := path[:lastSlash]
	return bucket, item
}

func normalizePath(path string) string {
	// TODO: Easy to optimize (a lot)
	path = strings.Trim(path, "/")

	var b bytes.Buffer

	for _, t := range strings.Split(path, "/") {
		if t == "" {
			continue
		}
		if b.Len() != 0 {
			b.WriteString("/")
		}
		b.WriteString(t)
	}
	return b.String()
}

func (s *Backend) Create(bucket string, k string, item itemData) (itemData, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	b := s.root.resolveBucket(bucket, true)
	existing, found := b.items[k]
	if found {
		return existing, errorAlreadyExists
	}

	s.lastLSN++
	lsn := s.lastLSN

	item.lsn = lsn

	log := &logEntry{lsn: lsn}
	log.items = []logItem{{b.path + k, watch.Added, item.data}}

	b.items[k] = item

	// Note we send the notifications before returning ... should be interesting :-)
	s.log.append(log)

	if item.expiry != 0 {
		s.expiryManager.add(b, k, item.expiry)
	}

	return item, nil
}

func (s *Backend) Delete(bucket string, k string, preconditions *storage.Preconditions) (itemData, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	b := s.root.resolveBucket(bucket, false)
	if b == nil {
		return itemData{}, errorItemNotFound
	}
	item, found := b.items[k]
	if !found {
		return itemData{}, errorItemNotFound
	}

	if preconditions != nil {
		if err := checkPreconditions(bucket+"/"+k, preconditions, item.uid); err != nil {
			return itemData{}, err
		}
	}

	oldItem := item

	s.lastLSN++
	lsn := s.lastLSN
	log := &logEntry{lsn: lsn}
	log.items = []logItem{{b.path + k, watch.Deleted, oldItem.data}}

	// Note we send the notifications before returning ... should be interesting :-)
	s.log.append(log)

	delete(b.items, k)

	return oldItem, nil
}

func (s *Backend) List(bucket string) ([]itemData, LSN, error) {
	var items []itemData

	s.mutex.Lock()
	defer s.mutex.Unlock()

	b := s.root.resolveBucket(bucket, false)
	if b == nil {
		// We return an empty list
		return nil, s.lastLSN, nil
	}

	if len(b.items) != 0 {
		items = make([]itemData, 0, len(b.items))
		for _, v := range b.items {
			items = append(items, v)
		}
	}

	return items, s.lastLSN, nil
}

// response will be the new item if we swapped,
// or the existing item if err==errorLSNMismatch
func (s *Backend) Update(bucket string, k string, oldLSN LSN, newItem itemData) (itemData, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	b := s.root.resolveBucket(bucket, true)
	item, found := b.items[k]
	if !found {
		return itemData{}, errorItemNotFound
	}
	if item.lsn != oldLSN {
		return item, errorLSNMismatch
	}

	s.lastLSN++
	lsn := s.lastLSN
	log := &logEntry{lsn: lsn}
	log.items = []logItem{{b.path + k, watch.Modified, newItem.data}}

	newItem.lsn = lsn
	b.items[k] = newItem

	// Note we send the notifications before returning ... should be interesting :-)
	s.log.append(log)

	if newItem.expiry != 0 {
		s.expiryManager.add(b, k, newItem.expiry)
	}

	return newItem, nil
}

func (s *Backend) checkExpiration(candidates []expiringItem) error {
	if len(candidates) == 0 {
		return nil
	}

	type notification struct {
		w     *watcher
		event *watch.Event
	}

	// Because channels have limited capacity, we buffer everything and send the notifications outside of the mutex
	// This will also let us be more atomic when we have things that can fail
	var notifications []notification

	err := func() error {
		s.mutex.Lock()
		defer s.mutex.Unlock()

		var log *logEntry

		now := uint64(time.Now().Unix())

		for i := range candidates {
			candidate := &candidates[i]
			k := candidate.key
			b := candidate.bucket

			item, found := b.items[k]
			if !found {
				continue
			}
			if item.expiry > now {
				continue
			}

			if log == nil {
				s.lastLSN++
				log = &logEntry{lsn: s.lastLSN}
			}
			log.items = []logItem{{b.path + k, watch.Deleted, item.data}}

			delete(b.items, candidate.key)

		}

		if log != nil {
			s.log.append(log)
		}

		return nil
	}()
	if err != nil {
		return err
	}

	for i := range notifications {
		notifications[i].w.resultChan <- *notifications[i].event
	}

	return nil
}

func (s *Backend) Get(bucket, k string) (itemData, LSN, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	b := s.root.resolveBucket(bucket, false)
	if b == nil {
		return itemData{}, s.lastLSN, errorItemNotFound
	}
	item, found := b.items[k]
	if !found {
		return itemData{}, s.lastLSN, errorItemNotFound
	}

	return item, s.lastLSN, nil
}
