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

package correlation

import (
	"go.opentelemetry.io/otel/api/core"
)

type rawMap map[core.Key]core.Value
type keySet map[core.Key]struct{}

// Map is an immutable storage for correlations.
type Map struct {
	m rawMap
}

// MapUpdate contains information about correlation changes to be
// made.
type MapUpdate struct {
	// DropSingleK contains a single key to be dropped from
	// correlations. Use this to avoid an overhead of a slice
	// allocation if there is only one key to drop.
	DropSingleK core.Key
	// DropMultiK contains all the keys to be dropped from
	// correlations.
	DropMultiK []core.Key

	// SingleKV contains a single key-value pair to be added to
	// correlations. Use this to avoid an overhead of a slice
	// allocation if there is only one key-value pair to add.
	SingleKV core.KeyValue
	// MultiKV contains all the key-value pairs to be added to
	// correlations.
	MultiKV []core.KeyValue
}

func newMap(raw rawMap) Map {
	return Map{
		m: raw,
	}
}

// NewEmptyMap creates an empty correlations map.
func NewEmptyMap() Map {
	return newMap(nil)
}

// NewMap creates a map with the contents of the update applied. In
// this function, having an update with DropSingleK or DropMultiK
// makes no sense - those fields are effectively ignored.
func NewMap(update MapUpdate) Map {
	return NewEmptyMap().Apply(update)
}

// Apply creates a copy of the map with the contents of the update
// applied. Apply will first drop the keys from DropSingleK and
// DropMultiK, then add key-value pairs from SingleKV and MultiKV.
func (m Map) Apply(update MapUpdate) Map {
	delSet, addSet := getModificationSets(update)
	mapSize := getNewMapSize(m.m, delSet, addSet)

	r := make(rawMap, mapSize)
	for k, v := range m.m {
		// do not copy items we want to drop
		if _, ok := delSet[k]; ok {
			continue
		}
		// do not copy items we would overwrite
		if _, ok := addSet[k]; ok {
			continue
		}
		r[k] = v
	}
	if update.SingleKV.Key.Defined() {
		r[update.SingleKV.Key] = update.SingleKV.Value
	}
	for _, kv := range update.MultiKV {
		r[kv.Key] = kv.Value
	}
	if len(r) == 0 {
		r = nil
	}
	return newMap(r)
}

func getModificationSets(update MapUpdate) (delSet, addSet keySet) {
	deletionsCount := len(update.DropMultiK)
	if update.DropSingleK.Defined() {
		deletionsCount++
	}
	if deletionsCount > 0 {
		delSet = make(map[core.Key]struct{}, deletionsCount)
		for _, k := range update.DropMultiK {
			delSet[k] = struct{}{}
		}
		if update.DropSingleK.Defined() {
			delSet[update.DropSingleK] = struct{}{}
		}
	}

	additionsCount := len(update.MultiKV)
	if update.SingleKV.Key.Defined() {
		additionsCount++
	}
	if additionsCount > 0 {
		addSet = make(map[core.Key]struct{}, additionsCount)
		for _, k := range update.MultiKV {
			addSet[k.Key] = struct{}{}
		}
		if update.SingleKV.Key.Defined() {
			addSet[update.SingleKV.Key] = struct{}{}
		}
	}

	return
}

func getNewMapSize(m rawMap, delSet, addSet keySet) int {
	mapSizeDiff := 0
	for k := range addSet {
		if _, ok := m[k]; !ok {
			mapSizeDiff++
		}
	}
	for k := range delSet {
		if _, ok := m[k]; ok {
			if _, inAddSet := addSet[k]; !inAddSet {
				mapSizeDiff--
			}
		}
	}
	return len(m) + mapSizeDiff
}

// Value gets a value from correlations map and returns a boolean
// value indicating whether the key exist in the map.
func (m Map) Value(k core.Key) (core.Value, bool) {
	value, ok := m.m[k]
	return value, ok
}

// HasValue returns a boolean value indicating whether the key exist
// in the map.
func (m Map) HasValue(k core.Key) bool {
	_, has := m.Value(k)
	return has
}

// Len returns a length of the map.
func (m Map) Len() int {
	return len(m.m)
}

// Foreach calls a passed callback once on each key-value pair until
// all the key-value pairs of the map were iterated or the callback
// returns false, whichever happens first.
func (m Map) Foreach(f func(kv core.KeyValue) bool) {
	for k, v := range m.m {
		if !f(core.KeyValue{
			Key:   k,
			Value: v,
		}) {
			return
		}
	}
}
