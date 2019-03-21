/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpc

import (
	"container/list"
)

type linkedMapKVPair struct {
	key   string
	value *stickyStoreEntry
}

// linkedMap is an implementation of a map that supports removing the oldest
// entry.
//
// linkedMap is NOT thread safe.
//
// It's for use of stickiness only!
type linkedMap struct {
	m map[string]*list.Element
	l *list.List // Head of the list is the oldest element.
}

// newLinkedMap returns a new LinkedMap.
func newLinkedMap() *linkedMap {
	return &linkedMap{
		m: make(map[string]*list.Element),
		l: list.New(),
	}
}

// put adds entry (key, value) to the map. Existing key will be overridden.
func (m *linkedMap) put(key string, value *stickyStoreEntry) {
	if oldE, ok := m.m[key]; ok {
		// Remove existing entry.
		m.l.Remove(oldE)
	}
	e := m.l.PushBack(&linkedMapKVPair{key: key, value: value})
	m.m[key] = e
}

// get returns the value of the given key.
func (m *linkedMap) get(key string) (*stickyStoreEntry, bool) {
	e, ok := m.m[key]
	if !ok {
		return nil, false
	}
	m.l.MoveToBack(e)
	return e.Value.(*linkedMapKVPair).value, true
}

// remove removes key from the map, and returns the value. The map is not
// modified if key is not in the map.
func (m *linkedMap) remove(key string) (*stickyStoreEntry, bool) {
	e, ok := m.m[key]
	if !ok {
		return nil, false
	}
	delete(m.m, key)
	m.l.Remove(e)
	return e.Value.(*linkedMapKVPair).value, true
}

// len returns the len of the map.
func (m *linkedMap) len() int {
	return len(m.m)
}

// clear removes all elements from the map.
func (m *linkedMap) clear() {
	m.m = make(map[string]*list.Element)
	m.l = list.New()
}

// removeOldest removes the oldest key from the map.
func (m *linkedMap) removeOldest() {
	e := m.l.Front()
	m.l.Remove(e)
	delete(m.m, e.Value.(*linkedMapKVPair).key)
}
