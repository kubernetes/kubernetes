/*
Copyright 2020 The Kubernetes Authors.

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

package value

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"sync"
	"sync/atomic"
)

// UnstructuredConverter defines how a type can be converted directly to unstructured.
// Types that implement json.Marshaler may also optionally implement this interface to provide a more
// direct and more efficient conversion. All types that choose to implement this interface must still
// implement this same conversion via json.Marshaler.
type UnstructuredConverter interface {
	json.Marshaler // require that json.Marshaler is implemented

	// ToUnstructured returns the unstructured representation.
	ToUnstructured() interface{}
}

// TypeReflectCacheEntry keeps data gathered using reflection about how a type is converted to/from unstructured.
type TypeReflectCacheEntry struct {
	isJsonMarshaler        bool
	ptrIsJsonMarshaler     bool
	isJsonUnmarshaler      bool
	ptrIsJsonUnmarshaler   bool
	isStringConvertable    bool
	ptrIsStringConvertable bool

	structFields        map[string]*FieldCacheEntry
	orderedStructFields []*FieldCacheEntry
}

// FieldCacheEntry keeps data gathered using reflection about how the field of a struct is converted to/from
// unstructured.
type FieldCacheEntry struct {
	// JsonName returns the name of the field according to the json tags on the struct field.
	JsonName string
	// isOmitEmpty is true if the field has the json 'omitempty' tag.
	isOmitEmpty bool
	// omitzero is set if the field has the json 'omitzero' tag.
	omitzero func(reflect.Value) bool
	// fieldPath is a list of field indices (see FieldByIndex) to lookup the value of
	// a field in a reflect.Value struct. The field indices in the list form a path used
	// to traverse through intermediary 'inline' fields.
	fieldPath [][]int

	fieldType reflect.Type
	TypeEntry *TypeReflectCacheEntry
}

func (f *FieldCacheEntry) CanOmit(fieldVal reflect.Value) bool {
	if f.isOmitEmpty && (safeIsNil(fieldVal) || isEmpty(fieldVal)) {
		return true
	}
	if f.omitzero != nil && f.omitzero(fieldVal) {
		return true
	}
	return false
}

// GetFrom returns the field identified by this FieldCacheEntry from the provided struct.
func (f *FieldCacheEntry) GetFrom(structVal reflect.Value) reflect.Value {
	// field might be nested within 'inline' structs
	for _, elem := range f.fieldPath {
		structVal = dereference(structVal).FieldByIndex(elem)
	}
	return structVal
}

var marshalerType = reflect.TypeOf(new(json.Marshaler)).Elem()
var unmarshalerType = reflect.TypeOf(new(json.Unmarshaler)).Elem()
var unstructuredConvertableType = reflect.TypeOf(new(UnstructuredConverter)).Elem()
var defaultReflectCache = newReflectCache()

// TypeReflectEntryOf returns the TypeReflectCacheEntry of the provided reflect.Type.
func TypeReflectEntryOf(t reflect.Type) *TypeReflectCacheEntry {
	cm := defaultReflectCache.get()
	if record, ok := cm[t]; ok {
		return record
	}
	updates := reflectCacheMap{}
	result := typeReflectEntryOf(cm, t, updates)
	if len(updates) > 0 {
		defaultReflectCache.update(updates)
	}
	return result
}

// TypeReflectEntryOf returns all updates needed to add provided reflect.Type, and the types its fields transitively
// depend on, to the cache.
func typeReflectEntryOf(cm reflectCacheMap, t reflect.Type, updates reflectCacheMap) *TypeReflectCacheEntry {
	if record, ok := cm[t]; ok {
		return record
	}
	if record, ok := updates[t]; ok {
		return record
	}
	typeEntry := &TypeReflectCacheEntry{
		isJsonMarshaler:        t.Implements(marshalerType),
		ptrIsJsonMarshaler:     reflect.PtrTo(t).Implements(marshalerType),
		isJsonUnmarshaler:      reflect.PtrTo(t).Implements(unmarshalerType),
		isStringConvertable:    t.Implements(unstructuredConvertableType),
		ptrIsStringConvertable: reflect.PtrTo(t).Implements(unstructuredConvertableType),
	}
	if t.Kind() == reflect.Struct {
		fieldEntries := map[string]*FieldCacheEntry{}
		buildStructCacheEntry(t, fieldEntries, nil)
		typeEntry.structFields = fieldEntries
		sortedByJsonName := make([]*FieldCacheEntry, len(fieldEntries))
		i := 0
		for _, entry := range fieldEntries {
			sortedByJsonName[i] = entry
			i++
		}
		sort.Slice(sortedByJsonName, func(i, j int) bool {
			return sortedByJsonName[i].JsonName < sortedByJsonName[j].JsonName
		})
		typeEntry.orderedStructFields = sortedByJsonName
	}

	// cyclic type references are allowed, so we must add the typeEntry to the updates map before resolving
	// the field.typeEntry references, or creating them if they are not already in the cache
	updates[t] = typeEntry

	for _, field := range typeEntry.structFields {
		if field.TypeEntry == nil {
			field.TypeEntry = typeReflectEntryOf(cm, field.fieldType, updates)
		}
	}
	return typeEntry
}

func buildStructCacheEntry(t reflect.Type, infos map[string]*FieldCacheEntry, fieldPath [][]int) {
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		jsonName, omit, isInline, isOmitempty, omitzero := lookupJsonTags(field)
		if omit {
			continue
		}
		if isInline {
			e := field.Type
			if field.Type.Kind() == reflect.Ptr {
				e = field.Type.Elem()
			}
			if e.Kind() == reflect.Struct {
				buildStructCacheEntry(e, infos, append(fieldPath, field.Index))
			}
			continue
		}
		info := &FieldCacheEntry{JsonName: jsonName, isOmitEmpty: isOmitempty, omitzero: omitzero, fieldPath: append(fieldPath, field.Index), fieldType: field.Type}
		infos[jsonName] = info
	}
}

// Fields returns a map of JSON field name to FieldCacheEntry for structs, or nil for non-structs.
func (e TypeReflectCacheEntry) Fields() map[string]*FieldCacheEntry {
	return e.structFields
}

// Fields returns a map of JSON field name to FieldCacheEntry for structs, or nil for non-structs.
func (e TypeReflectCacheEntry) OrderedFields() []*FieldCacheEntry {
	return e.orderedStructFields
}

// CanConvertToUnstructured returns true if this TypeReflectCacheEntry can convert values of its type to unstructured.
func (e TypeReflectCacheEntry) CanConvertToUnstructured() bool {
	return e.isJsonMarshaler || e.ptrIsJsonMarshaler || e.isStringConvertable || e.ptrIsStringConvertable
}

// ToUnstructured converts the provided value to unstructured and returns it.
func (e TypeReflectCacheEntry) ToUnstructured(sv reflect.Value) (interface{}, error) {
	// This is based on https://github.com/kubernetes/kubernetes/blob/82c9e5c814eb7acc6cc0a090c057294d0667ad66/staging/src/k8s.io/apimachinery/pkg/runtime/converter.go#L505
	// and is intended to replace it.

	// Check if the object is a nil pointer.
	if sv.Kind() == reflect.Ptr && sv.IsNil() {
		// We're done - we don't need to store anything.
		return nil, nil
	}
	// Check if the object has a custom string converter and use it if available, since it is much more efficient
	// than round tripping through json.
	if converter, ok := e.getUnstructuredConverter(sv); ok {
		return converter.ToUnstructured(), nil
	}
	// Check if the object has a custom JSON marshaller/unmarshaller.
	if marshaler, ok := e.getJsonMarshaler(sv); ok {
		data, err := marshaler.MarshalJSON()
		if err != nil {
			return nil, err
		}
		switch {
		case len(data) == 0:
			return nil, fmt.Errorf("error decoding from json: empty value")

		case bytes.Equal(data, nullBytes):
			// We're done - we don't need to store anything.
			return nil, nil

		case bytes.Equal(data, trueBytes):
			return true, nil

		case bytes.Equal(data, falseBytes):
			return false, nil

		case data[0] == '"':
			var result string
			err := unmarshal(data, &result)
			if err != nil {
				return nil, fmt.Errorf("error decoding string from json: %v", err)
			}
			return result, nil

		case data[0] == '{':
			result := make(map[string]interface{})
			err := unmarshal(data, &result)
			if err != nil {
				return nil, fmt.Errorf("error decoding object from json: %v", err)
			}
			return result, nil

		case data[0] == '[':
			result := make([]interface{}, 0)
			err := unmarshal(data, &result)
			if err != nil {
				return nil, fmt.Errorf("error decoding array from json: %v", err)
			}
			return result, nil

		default:
			var (
				resultInt   int64
				resultFloat float64
				err         error
			)
			if err = unmarshal(data, &resultInt); err == nil {
				return resultInt, nil
			} else if err = unmarshal(data, &resultFloat); err == nil {
				return resultFloat, nil
			} else {
				return nil, fmt.Errorf("error decoding number from json: %v", err)
			}
		}
	}

	return nil, fmt.Errorf("provided type cannot be converted: %v", sv.Type())
}

// CanConvertFromUnstructured returns true if this TypeReflectCacheEntry can convert objects of the type from unstructured.
func (e TypeReflectCacheEntry) CanConvertFromUnstructured() bool {
	return e.isJsonUnmarshaler
}

// FromUnstructured converts the provided source value from unstructured into the provided destination value.
func (e TypeReflectCacheEntry) FromUnstructured(sv, dv reflect.Value) error {
	// TODO: this could be made much more efficient using direct conversions like
	// UnstructuredConverter.ToUnstructured provides.
	st := dv.Type()
	data, err := json.Marshal(sv.Interface())
	if err != nil {
		return fmt.Errorf("error encoding %s to json: %v", st.String(), err)
	}
	if unmarshaler, ok := e.getJsonUnmarshaler(dv); ok {
		return unmarshaler.UnmarshalJSON(data)
	}
	return fmt.Errorf("unable to unmarshal %v into %v", sv.Type(), dv.Type())
}

var (
	nullBytes  = []byte("null")
	trueBytes  = []byte("true")
	falseBytes = []byte("false")
)

func (e TypeReflectCacheEntry) getJsonMarshaler(v reflect.Value) (json.Marshaler, bool) {
	if e.isJsonMarshaler {
		return v.Interface().(json.Marshaler), true
	}
	if e.ptrIsJsonMarshaler {
		// Check pointer receivers if v is not a pointer
		if v.Kind() != reflect.Ptr && v.CanAddr() {
			v = v.Addr()
			return v.Interface().(json.Marshaler), true
		}
	}
	return nil, false
}

func (e TypeReflectCacheEntry) getJsonUnmarshaler(v reflect.Value) (json.Unmarshaler, bool) {
	if !e.isJsonUnmarshaler {
		return nil, false
	}
	return v.Addr().Interface().(json.Unmarshaler), true
}

func (e TypeReflectCacheEntry) getUnstructuredConverter(v reflect.Value) (UnstructuredConverter, bool) {
	if e.isStringConvertable {
		return v.Interface().(UnstructuredConverter), true
	}
	if e.ptrIsStringConvertable {
		// Check pointer receivers if v is not a pointer
		if v.CanAddr() {
			v = v.Addr()
			return v.Interface().(UnstructuredConverter), true
		}
	}
	return nil, false
}

type typeReflectCache struct {
	// use an atomic and copy-on-write since there are a fixed (typically very small) number of structs compiled into any
	// go program using this cache
	value atomic.Value
	// mu is held by writers when performing load/modify/store operations on the cache, readers do not need to hold a
	// read-lock since the atomic value is always read-only
	mu sync.Mutex
}

func newReflectCache() *typeReflectCache {
	cache := &typeReflectCache{}
	cache.value.Store(make(reflectCacheMap))
	return cache
}

type reflectCacheMap map[reflect.Type]*TypeReflectCacheEntry

// get returns the reflectCacheMap.
func (c *typeReflectCache) get() reflectCacheMap {
	return c.value.Load().(reflectCacheMap)
}

// update merges the provided updates into the cache.
func (c *typeReflectCache) update(updates reflectCacheMap) {
	c.mu.Lock()
	defer c.mu.Unlock()

	currentCacheMap := c.value.Load().(reflectCacheMap)

	hasNewEntries := false
	for t := range updates {
		if _, ok := currentCacheMap[t]; !ok {
			hasNewEntries = true
			break
		}
	}
	if !hasNewEntries {
		// Bail if the updates have been set while waiting for lock acquisition.
		// This is safe since setting entries is idempotent.
		return
	}

	newCacheMap := make(reflectCacheMap, len(currentCacheMap)+len(updates))
	for k, v := range currentCacheMap {
		newCacheMap[k] = v
	}
	for t, update := range updates {
		newCacheMap[t] = update
	}
	c.value.Store(newCacheMap)
}

// Below json Unmarshal is fromk8s.io/apimachinery/pkg/util/json
// to handle number conversions as expected by Kubernetes

// limit recursive depth to prevent stack overflow errors
const maxDepth = 10000

// unmarshal unmarshals the given data
// If v is a *map[string]interface{}, numbers are converted to int64 or float64
func unmarshal(data []byte, v interface{}) error {
	// Build a decoder from the given data
	decoder := json.NewDecoder(bytes.NewBuffer(data))
	// Preserve numbers, rather than casting to float64 automatically
	decoder.UseNumber()
	// Run the decode
	if err := decoder.Decode(v); err != nil {
		return err
	}
	next := decoder.InputOffset()
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		tail := bytes.TrimLeft(data[next:], " \t\r\n")
		return fmt.Errorf("unexpected trailing data at offset %d", len(data)-len(tail))
	}

	// If the decode succeeds, post-process the object to convert json.Number objects to int64 or float64
	switch v := v.(type) {
	case *map[string]interface{}:
		return convertMapNumbers(*v, 0)

	case *[]interface{}:
		return convertSliceNumbers(*v, 0)

	case *interface{}:
		return convertInterfaceNumbers(v, 0)

	default:
		return nil
	}
}

func convertInterfaceNumbers(v *interface{}, depth int) error {
	var err error
	switch v2 := (*v).(type) {
	case json.Number:
		*v, err = convertNumber(v2)
	case map[string]interface{}:
		err = convertMapNumbers(v2, depth+1)
	case []interface{}:
		err = convertSliceNumbers(v2, depth+1)
	}
	return err
}

// convertMapNumbers traverses the map, converting any json.Number values to int64 or float64.
// values which are map[string]interface{} or []interface{} are recursively visited
func convertMapNumbers(m map[string]interface{}, depth int) error {
	if depth > maxDepth {
		return fmt.Errorf("exceeded max depth of %d", maxDepth)
	}

	var err error
	for k, v := range m {
		switch v := v.(type) {
		case json.Number:
			m[k], err = convertNumber(v)
		case map[string]interface{}:
			err = convertMapNumbers(v, depth+1)
		case []interface{}:
			err = convertSliceNumbers(v, depth+1)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// convertSliceNumbers traverses the slice, converting any json.Number values to int64 or float64.
// values which are map[string]interface{} or []interface{} are recursively visited
func convertSliceNumbers(s []interface{}, depth int) error {
	if depth > maxDepth {
		return fmt.Errorf("exceeded max depth of %d", maxDepth)
	}

	var err error
	for i, v := range s {
		switch v := v.(type) {
		case json.Number:
			s[i], err = convertNumber(v)
		case map[string]interface{}:
			err = convertMapNumbers(v, depth+1)
		case []interface{}:
			err = convertSliceNumbers(v, depth+1)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// convertNumber converts a json.Number to an int64 or float64, or returns an error
func convertNumber(n json.Number) (interface{}, error) {
	// Attempt to convert to an int64 first
	if i, err := n.Int64(); err == nil {
		return i, nil
	}
	// Return a float64 (default json.Decode() behavior)
	// An overflow will return an error
	return n.Float64()
}
