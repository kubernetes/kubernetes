// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package trace // import "go.opentelemetry.io/otel/sdk/trace"

import (
	"container/list"

	"go.opentelemetry.io/otel/attribute"
)

// attributesMap is a capped map of attributes, holding the most recent attributes.
// Eviction is done via a LRU method, the oldest entry is removed to create room for a new entry.
// Updates are allowed and they refresh the usage of the key.
//
// This is based from https://github.com/hashicorp/golang-lru/blob/master/simplelru/lru.go
// With a subset of the its operations and specific for holding attribute.KeyValue
type attributesMap struct {
	attributes   map[attribute.Key]*list.Element
	evictList    *list.List
	droppedCount int
	capacity     int
}

func newAttributesMap(capacity int) *attributesMap {
	lm := &attributesMap{
		attributes: make(map[attribute.Key]*list.Element),
		evictList:  list.New(),
		capacity:   capacity,
	}
	return lm
}

func (am *attributesMap) add(kv attribute.KeyValue) {
	// Check for existing item
	if ent, ok := am.attributes[kv.Key]; ok {
		am.evictList.MoveToFront(ent)
		ent.Value = &kv
		return
	}

	// Add new item
	entry := am.evictList.PushFront(&kv)
	am.attributes[kv.Key] = entry

	// Verify size not exceeded
	if am.evictList.Len() > am.capacity {
		am.removeOldest()
		am.droppedCount++
	}
}

// toKeyValue copies the attributesMap into a slice of attribute.KeyValue and
// returns it. If the map is empty, a nil is returned.
// TODO: Is it more efficient to return a pointer to the slice?
func (am *attributesMap) toKeyValue() []attribute.KeyValue {
	len := am.evictList.Len()
	if len == 0 {
		return nil
	}

	attributes := make([]attribute.KeyValue, 0, len)
	for ent := am.evictList.Back(); ent != nil; ent = ent.Prev() {
		if value, ok := ent.Value.(*attribute.KeyValue); ok {
			attributes = append(attributes, *value)
		}
	}

	return attributes
}

// removeOldest removes the oldest item from the cache.
func (am *attributesMap) removeOldest() {
	ent := am.evictList.Back()
	if ent != nil {
		am.evictList.Remove(ent)
		kv := ent.Value.(*attribute.KeyValue)
		delete(am.attributes, kv.Key)
	}
}
