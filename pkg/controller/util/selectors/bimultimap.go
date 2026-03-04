/*
Copyright 2022 The Kubernetes Authors.

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

package selectors

import (
	"fmt"
	"strings"
	"sync"

	pkglabels "k8s.io/apimachinery/pkg/labels"
)

// BiMultimap is an efficient, bi-directional mapping of object
// keys. Associations are created by putting keys with a selector.
type BiMultimap struct {
	mux sync.RWMutex

	// Objects.
	labeledObjects   map[Key]*labeledObject
	selectingObjects map[Key]*selectingObject

	// Associations.
	labeledBySelecting map[selectorKey]*labeledObjects
	selectingByLabeled map[labelsKey]*selectingObjects
}

// NewBiMultimap creates a map.
func NewBiMultimap() *BiMultimap {
	return &BiMultimap{
		labeledObjects:     make(map[Key]*labeledObject),
		selectingObjects:   make(map[Key]*selectingObject),
		labeledBySelecting: make(map[selectorKey]*labeledObjects),
		selectingByLabeled: make(map[labelsKey]*selectingObjects),
	}
}

// Key is a tuple of name and namespace.
type Key struct {
	Name      string
	Namespace string
}

// Parse turns a string in the format namespace/name into a Key.
func Parse(s string) (key Key) {
	ns := strings.SplitN(s, "/", 2)
	if len(ns) == 2 {
		key.Namespace = ns[0]
		key.Name = ns[1]
	} else {
		key.Name = ns[0]
	}
	return key
}

func (k Key) String() string {
	return fmt.Sprintf("%v/%v", k.Namespace, k.Name)
}

type selectorKey struct {
	key       string
	namespace string
}

type selectingObject struct {
	key      Key
	selector pkglabels.Selector
	// selectorKey is a stable serialization of selector for
	// association caching.
	selectorKey selectorKey
}

type selectingObjects struct {
	objects  map[Key]*selectingObject
	refCount int
}

type labelsKey struct {
	key       string
	namespace string
}

type labeledObject struct {
	key    Key
	labels map[string]string
	// labelsKey is a stable serialization of labels for association
	// caching.
	labelsKey labelsKey
}

type labeledObjects struct {
	objects  map[Key]*labeledObject
	refCount int
}

// Put inserts or updates an object and the incoming associations
// based on the object labels.
func (m *BiMultimap) Put(key Key, labels map[string]string) {
	m.mux.Lock()
	defer m.mux.Unlock()

	labelsKey := labelsKey{
		key:       pkglabels.Set(labels).String(),
		namespace: key.Namespace,
	}
	if l, ok := m.labeledObjects[key]; ok {
		// Update labeled object.
		if labelsKey == l.labelsKey {
			// No change to labels.
			return
		}
		// Delete before readding.
		m.delete(key)
	}
	// Add labeled object.
	labels = copyLabels(labels)
	labeledObject := &labeledObject{
		key:       key,
		labels:    labels,
		labelsKey: labelsKey,
	}
	m.labeledObjects[key] = labeledObject
	// Add associations.
	if _, ok := m.selectingByLabeled[labelsKey]; !ok {
		// Cache miss. Scan selecting objects.
		selecting := &selectingObjects{
			objects: make(map[Key]*selectingObject),
		}
		set := pkglabels.Set(labels)
		for _, s := range m.selectingObjects {
			if s.key.Namespace != key.Namespace {
				continue
			}
			if s.selector.Matches(set) {
				selecting.objects[s.key] = s
			}
		}
		// Associate selecting with labeled.
		m.selectingByLabeled[labelsKey] = selecting
	}
	selecting := m.selectingByLabeled[labelsKey]
	selecting.refCount++
	for _, sObject := range selecting.objects {
		// Associate labeled with selecting.
		labeled := m.labeledBySelecting[sObject.selectorKey]
		labeled.objects[labeledObject.key] = labeledObject
	}
}

// Delete removes a labeled object and incoming associations.
func (m *BiMultimap) Delete(key Key) {
	m.mux.Lock()
	defer m.mux.Unlock()
	m.delete(key)
}

func (m *BiMultimap) delete(key Key) {
	if _, ok := m.labeledObjects[key]; !ok {
		// Does not exist.
		return
	}
	labeledObject := m.labeledObjects[key]
	labelsKey := labeledObject.labelsKey
	defer delete(m.labeledObjects, key)
	if _, ok := m.selectingByLabeled[labelsKey]; !ok {
		// No associations.
		return
	}
	// Remove associations.
	for _, selectingObject := range m.selectingByLabeled[labelsKey].objects {
		selectorKey := selectingObject.selectorKey
		// Delete selectingObject to labeledObject association.
		delete(m.labeledBySelecting[selectorKey].objects, key)
	}
	m.selectingByLabeled[labelsKey].refCount--
	// Garbage collect labeledObject to selectingObject associations.
	if m.selectingByLabeled[labelsKey].refCount == 0 {
		delete(m.selectingByLabeled, labelsKey)
	}
}

// Exists returns true if the labeled object is present in the map.
func (m *BiMultimap) Exists(key Key) bool {
	m.mux.RLock()
	defer m.mux.RUnlock()

	_, exists := m.labeledObjects[key]
	return exists
}

// PutSelector inserts or updates an object with a selector. Associations
// are created or updated based on the selector.
func (m *BiMultimap) PutSelector(key Key, selector pkglabels.Selector) {
	m.mux.Lock()
	defer m.mux.Unlock()

	selectorKey := selectorKey{
		key:       selector.String(),
		namespace: key.Namespace,
	}
	if s, ok := m.selectingObjects[key]; ok {
		// Update selecting object.
		if selectorKey == s.selectorKey {
			// No change to selector.
			return
		}
		// Delete before readding.
		m.deleteSelector(key)
	}
	// Add selecting object.
	selectingObject := &selectingObject{
		key:         key,
		selector:    selector,
		selectorKey: selectorKey,
	}
	m.selectingObjects[key] = selectingObject
	// Add associations.
	if _, ok := m.labeledBySelecting[selectorKey]; !ok {
		// Cache miss. Scan labeled objects.
		labeled := &labeledObjects{
			objects: make(map[Key]*labeledObject),
		}
		for _, l := range m.labeledObjects {
			if l.key.Namespace != key.Namespace {
				continue
			}
			set := pkglabels.Set(l.labels)
			if selector.Matches(set) {
				labeled.objects[l.key] = l
			}
		}
		// Associate labeled with selecting.
		m.labeledBySelecting[selectorKey] = labeled
	}
	labeled := m.labeledBySelecting[selectorKey]
	labeled.refCount++
	for _, labeledObject := range labeled.objects {
		// Associate selecting with labeled.
		selecting := m.selectingByLabeled[labeledObject.labelsKey]
		selecting.objects[selectingObject.key] = selectingObject
	}
}

// DeleteSelector deletes a selecting object and associations created by its
// selector.
func (m *BiMultimap) DeleteSelector(key Key) {
	m.mux.Lock()
	defer m.mux.Unlock()
	m.deleteSelector(key)
}

func (m *BiMultimap) deleteSelector(key Key) {
	if _, ok := m.selectingObjects[key]; !ok {
		// Does not exist.
		return
	}
	selectingObject := m.selectingObjects[key]
	selectorKey := selectingObject.selectorKey
	defer delete(m.selectingObjects, key)
	if _, ok := m.labeledBySelecting[selectorKey]; !ok {
		// No associations.
		return
	}
	// Remove associations.
	for _, labeledObject := range m.labeledBySelecting[selectorKey].objects {
		labelsKey := labeledObject.labelsKey
		// Delete labeledObject to selectingObject association.
		delete(m.selectingByLabeled[labelsKey].objects, key)
	}
	m.labeledBySelecting[selectorKey].refCount--
	// Garbage collect selectingObjects to labeledObject associations.
	if m.labeledBySelecting[selectorKey].refCount == 0 {
		delete(m.labeledBySelecting, selectorKey)
	}
}

// SelectorExists returns true if the selecting object is present in the map.
func (m *BiMultimap) SelectorExists(key Key) bool {
	m.mux.RLock()
	defer m.mux.RUnlock()

	_, exists := m.selectingObjects[key]
	return exists
}

// KeepOnly retains only the specified labeled objects and deletes the
// rest. Like calling Delete for all keys not specified.
func (m *BiMultimap) KeepOnly(keys []Key) {
	m.mux.Lock()
	defer m.mux.Unlock()

	keyMap := make(map[Key]bool)
	for _, k := range keys {
		keyMap[k] = true
	}
	for k := range m.labeledObjects {
		if !keyMap[k] {
			m.delete(k)
		}
	}
}

// KeepOnlySelectors retains only the specified selecting objects and
// deletes the rest. Like calling DeleteSelector for all keys not
// specified.
func (m *BiMultimap) KeepOnlySelectors(keys []Key) {
	m.mux.Lock()
	defer m.mux.Unlock()

	keyMap := make(map[Key]bool)
	for _, k := range keys {
		keyMap[k] = true
	}
	for k := range m.selectingObjects {
		if !keyMap[k] {
			m.deleteSelector(k)
		}
	}
}

// Select finds objects associated with a selecting object. If the
// given key was found in the map `ok` will be true. Otherwise false.
func (m *BiMultimap) Select(key Key) (keys []Key, ok bool) {
	m.mux.RLock()
	defer m.mux.RUnlock()

	selectingObject, ok := m.selectingObjects[key]
	if !ok {
		// Does not exist.
		return nil, false
	}
	keys = make([]Key, 0)
	if labeled, ok := m.labeledBySelecting[selectingObject.selectorKey]; ok {
		for _, labeledObject := range labeled.objects {
			keys = append(keys, labeledObject.key)
		}
	}
	return keys, true
}

// ReverseSelect finds objects selecting the given object. If the
// given key was found in the map `ok` will be true. Otherwise false.
func (m *BiMultimap) ReverseSelect(key Key) (keys []Key, ok bool) {
	m.mux.RLock()
	defer m.mux.RUnlock()

	labeledObject, ok := m.labeledObjects[key]
	if !ok {
		// Does not exist.
		return []Key{}, false
	}
	keys = make([]Key, 0)
	if selecting, ok := m.selectingByLabeled[labeledObject.labelsKey]; ok {
		for _, selectingObject := range selecting.objects {
			keys = append(keys, selectingObject.key)
		}
	}
	return keys, true
}

func copyLabels(labels map[string]string) map[string]string {
	l := make(map[string]string)
	for k, v := range labels {
		l[k] = v
	}
	return l
}
