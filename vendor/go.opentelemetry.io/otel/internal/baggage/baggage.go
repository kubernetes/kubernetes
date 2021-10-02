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

// Package baggage provides types and functions to manage W3C Baggage.
package baggage

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
)

type rawMap map[attribute.Key]attribute.Value
type keySet map[attribute.Key]struct{}

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
	DropSingleK attribute.Key
	// DropMultiK contains all the keys to be dropped from
	// correlations.
	DropMultiK []attribute.Key

	// SingleKV contains a single key-value pair to be added to
	// correlations. Use this to avoid an overhead of a slice
	// allocation if there is only one key-value pair to add.
	SingleKV attribute.KeyValue
	// MultiKV contains all the key-value pairs to be added to
	// correlations.
	MultiKV []attribute.KeyValue
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
		delSet = make(map[attribute.Key]struct{}, deletionsCount)
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
		addSet = make(map[attribute.Key]struct{}, additionsCount)
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
func (m Map) Value(k attribute.Key) (attribute.Value, bool) {
	value, ok := m.m[k]
	return value, ok
}

// HasValue returns a boolean value indicating whether the key exist
// in the map.
func (m Map) HasValue(k attribute.Key) bool {
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
func (m Map) Foreach(f func(attribute.KeyValue) bool) {
	for k, v := range m.m {
		if !f(attribute.KeyValue{
			Key:   k,
			Value: v,
		}) {
			return
		}
	}
}

type correlationsType struct{}

// SetHookFunc describes a type of a callback that is called when
// storing baggage in the context.
type SetHookFunc func(context.Context) context.Context

// GetHookFunc describes a type of a callback that is called when
// getting baggage from the context.
type GetHookFunc func(context.Context, Map) Map

// value under this key is either of type Map or correlationsData
var correlationsKey = &correlationsType{}

type correlationsData struct {
	m       Map
	setHook SetHookFunc
	getHook GetHookFunc
}

func (d correlationsData) isHookless() bool {
	return d.setHook == nil && d.getHook == nil
}

type hookKind int

const (
	hookKindSet hookKind = iota
	hookKindGet
)

func (d *correlationsData) overrideHook(kind hookKind, setHook SetHookFunc, getHook GetHookFunc) {
	switch kind {
	case hookKindSet:
		d.setHook = setHook
	case hookKindGet:
		d.getHook = getHook
	}
}

// ContextWithSetHook installs a hook function that will be invoked
// every time ContextWithMap is called. To avoid unnecessary callback
// invocations (recursive or not), the callback can temporarily clear
// the hooks from the context with the ContextWithNoHooks function.
//
// Note that NewContext also calls ContextWithMap, so the hook will be
// invoked.
//
// Passing nil SetHookFunc creates a context with no set hook to call.
//
// This function should not be used by applications or libraries. It
// is mostly for interoperation with other observability APIs.
func ContextWithSetHook(ctx context.Context, hook SetHookFunc) context.Context {
	return contextWithHook(ctx, hookKindSet, hook, nil)
}

// ContextWithGetHook installs a hook function that will be invoked
// every time MapFromContext is called. To avoid unnecessary callback
// invocations (recursive or not), the callback can temporarily clear
// the hooks from the context with the ContextWithNoHooks function.
//
// Note that NewContext also calls MapFromContext, so the hook will be
// invoked.
//
// Passing nil GetHookFunc creates a context with no get hook to call.
//
// This function should not be used by applications or libraries. It
// is mostly for interoperation with other observability APIs.
func ContextWithGetHook(ctx context.Context, hook GetHookFunc) context.Context {
	return contextWithHook(ctx, hookKindGet, nil, hook)
}

func contextWithHook(ctx context.Context, kind hookKind, setHook SetHookFunc, getHook GetHookFunc) context.Context {
	switch v := ctx.Value(correlationsKey).(type) {
	case correlationsData:
		v.overrideHook(kind, setHook, getHook)
		if v.isHookless() {
			return context.WithValue(ctx, correlationsKey, v.m)
		}
		return context.WithValue(ctx, correlationsKey, v)
	case Map:
		return contextWithOneHookAndMap(ctx, kind, setHook, getHook, v)
	default:
		m := NewEmptyMap()
		return contextWithOneHookAndMap(ctx, kind, setHook, getHook, m)
	}
}

func contextWithOneHookAndMap(ctx context.Context, kind hookKind, setHook SetHookFunc, getHook GetHookFunc, m Map) context.Context {
	d := correlationsData{m: m}
	d.overrideHook(kind, setHook, getHook)
	if d.isHookless() {
		return ctx
	}
	return context.WithValue(ctx, correlationsKey, d)
}

// ContextWithNoHooks creates a context with all the hooks
// disabled. Also returns old set and get hooks. This function can be
// used to temporarily clear the context from hooks and then reinstate
// them by calling ContextWithSetHook and ContextWithGetHook functions
// passing the hooks returned by this function.
//
// This function should not be used by applications or libraries. It
// is mostly for interoperation with other observability APIs.
func ContextWithNoHooks(ctx context.Context) (context.Context, SetHookFunc, GetHookFunc) {
	switch v := ctx.Value(correlationsKey).(type) {
	case correlationsData:
		return context.WithValue(ctx, correlationsKey, v.m), v.setHook, v.getHook
	default:
		return ctx, nil, nil
	}
}

// ContextWithMap returns a context with the Map entered into it.
func ContextWithMap(ctx context.Context, m Map) context.Context {
	switch v := ctx.Value(correlationsKey).(type) {
	case correlationsData:
		v.m = m
		ctx = context.WithValue(ctx, correlationsKey, v)
		if v.setHook != nil {
			ctx = v.setHook(ctx)
		}
		return ctx
	default:
		return context.WithValue(ctx, correlationsKey, m)
	}
}

// ContextWithNoCorrelationData returns a context stripped of correlation
// data.
func ContextWithNoCorrelationData(ctx context.Context) context.Context {
	return context.WithValue(ctx, correlationsKey, nil)
}

// NewContext returns a context with the map from passed context
// updated with the passed key-value pairs.
func NewContext(ctx context.Context, keyvalues ...attribute.KeyValue) context.Context {
	return ContextWithMap(ctx, MapFromContext(ctx).Apply(MapUpdate{
		MultiKV: keyvalues,
	}))
}

// MapFromContext gets the current Map from a Context.
func MapFromContext(ctx context.Context) Map {
	switch v := ctx.Value(correlationsKey).(type) {
	case correlationsData:
		if v.getHook != nil {
			return v.getHook(ctx, v.m)
		}
		return v.m
	case Map:
		return v
	default:
		return NewEmptyMap()
	}
}
