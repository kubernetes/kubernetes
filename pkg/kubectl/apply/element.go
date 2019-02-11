/*
Copyright 2017 The Kubernetes Authors.

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

package apply

import (
	"fmt"
)

// Element contains the record, local, and remote value for a field in an object
// and metadata about the field read from openapi.
// Calling Merge on an element will apply the passed in strategy to Element -
// e.g. either replacing the whole element with the local copy or merging each
// of the recorded, local and remote fields of the element.
type Element interface {
	// FieldMeta specifies which merge strategy to use for this element
	FieldMeta

	// Merge merges the recorded, local and remote values in the element using the Strategy
	// provided as an argument.  Calls the type specific method on the Strategy - following the
	// "Accept" method from the "Visitor" pattern.
	// e.g. Merge on a ListElement will call Strategy.MergeList(self)
	// Returns the Result of the merged elements
	Merge(Strategy) (Result, error)

	// HasRecorded returns true if the field was explicitly
	// present in the recorded source.  This is to differentiate between
	// undefined and set to null
	HasRecorded() bool

	// GetRecorded returns the field value from the recorded source of the object
	GetRecorded() interface{}

	// HasLocal returns true if the field was explicitly
	// present in the local source.  This is to differentiate between
	// undefined and set to null
	HasLocal() bool

	// GetLocal returns the field value from the local source of the object
	GetLocal() interface{}

	// HasRemote returns true if the field was explicitly
	// present in the remote source.  This is to differentiate between
	// undefined and set to null
	HasRemote() bool

	// GetRemote returns the field value from the remote source of the object
	GetRemote() interface{}
}

// FieldMeta defines the strategy used to apply a Patch for an element
type FieldMeta interface {
	// GetFieldMergeType returns the type of merge strategy to use for this field
	// maybe "merge", "replace" or "retainkeys"
	// TODO: There maybe multiple strategies, so this may need to be a slice, map, or struct
	// Address this in a follow up in the PR to introduce retainkeys strategy
	GetFieldMergeType() string

	// GetFieldMergeKeys returns the merge key to use when the MergeType is "merge" and underlying type is a list
	GetFieldMergeKeys() MergeKeys

	// GetFieldType returns the openapi field type - e.g. primitive, array, map, type, reference
	GetFieldType() string
}

// FieldMetaImpl implements FieldMeta
type FieldMetaImpl struct {
	// MergeType is the type of merge strategy to use for this field
	// maybe "merge", "replace" or "retainkeys"
	MergeType string

	// MergeKeys are the merge keys to use when the MergeType is "merge" and underlying type is a list
	MergeKeys MergeKeys

	// Type is the openapi type of the field - "list", "primitive", "map"
	Type string

	// Name contains name of the field
	Name string
}

// GetFieldMergeType implements FieldMeta.GetFieldMergeType
func (s FieldMetaImpl) GetFieldMergeType() string {
	return s.MergeType
}

// GetFieldMergeKeys implements FieldMeta.GetFieldMergeKeys
func (s FieldMetaImpl) GetFieldMergeKeys() MergeKeys {
	return s.MergeKeys
}

// GetFieldType implements FieldMeta.GetFieldType
func (s FieldMetaImpl) GetFieldType() string {
	return s.Type
}

// MergeKeyValue records the value of the mergekey for an item in a list
type MergeKeyValue map[string]string

// Equal returns true if the MergeKeyValues share the same value,
// representing the same item in a list
func (v MergeKeyValue) Equal(o MergeKeyValue) bool {
	if len(v) != len(o) {
		return false
	}

	for key, v1 := range v {
		if v2, found := o[key]; !found || v1 != v2 {
			return false
		}
	}

	return true
}

// MergeKeys is the set of fields on an object that uniquely identify
// and is used when merging lists to identify the "same" object
// independent of the ordering of the objects
type MergeKeys []string

// GetMergeKeyValue parses the MergeKeyValue from an item in a list
func (mk MergeKeys) GetMergeKeyValue(i interface{}) (MergeKeyValue, error) {
	result := MergeKeyValue{}
	if len(mk) <= 0 {
		return result, fmt.Errorf("merge key must have at least 1 value to merge")
	}
	m, ok := i.(map[string]interface{})
	if !ok {
		return result, fmt.Errorf("cannot use mergekey %v for primitive item in list %v", mk, i)
	}
	for _, field := range mk {
		if value, found := m[field]; !found {
			result[field] = ""
		} else {
			result[field] = fmt.Sprintf("%v", value)
		}
	}
	return result, nil
}

type source int

// CombinedPrimitiveSlice implements a slice of primitives
type CombinedPrimitiveSlice struct {
	Items []*PrimitiveListItem
}

// PrimitiveListItem represents a single value in a slice of primitives
type PrimitiveListItem struct {
	// Value is the value of the primitive, should match recorded, local and remote
	Value interface{}

	RawElementData
}

// Contains returns true if the slice contains the l
func (s *CombinedPrimitiveSlice) lookup(l interface{}) *PrimitiveListItem {
	val := fmt.Sprintf("%v", l)
	for _, i := range s.Items {
		if fmt.Sprintf("%v", i.Value) == val {
			return i
		}
	}
	return nil
}

func (s *CombinedPrimitiveSlice) upsert(l interface{}) *PrimitiveListItem {
	// Return the item if it exists
	if item := s.lookup(l); item != nil {
		return item
	}

	// Otherwise create a new item and append to the list
	item := &PrimitiveListItem{
		Value: l,
	}
	s.Items = append(s.Items, item)
	return item
}

// UpsertRecorded adds l to the slice.  If there is already a value of l in the
// slice for either the local or remote, set on that value as the recorded value
// Otherwise append a new item to the list with the recorded value.
func (s *CombinedPrimitiveSlice) UpsertRecorded(l interface{}) {
	v := s.upsert(l)
	v.recorded = l
	v.recordedSet = true
}

// UpsertLocal adds l to the slice.  If there is already a value of l in the
// slice for either the recorded or remote, set on that value as the local value
// Otherwise append a new item to the list with the local value.
func (s *CombinedPrimitiveSlice) UpsertLocal(l interface{}) {
	v := s.upsert(l)
	v.local = l
	v.localSet = true
}

// UpsertRemote adds l to the slice.  If there is already a value of l in the
// slice for either the local or recorded, set on that value as the remote value
// Otherwise append a new item to the list with the remote value.
func (s *CombinedPrimitiveSlice) UpsertRemote(l interface{}) {
	v := s.upsert(l)
	v.remote = l
	v.remoteSet = true
}

// ListItem represents a single value in a slice of maps or types
type ListItem struct {
	// KeyValue is the merge key value of the item
	KeyValue MergeKeyValue

	// RawElementData contains the field values
	RawElementData
}

// CombinedMapSlice is a slice of maps or types with merge keys
type CombinedMapSlice struct {
	Items []*ListItem
}

// Lookup returns the ListItem matching the merge key, or nil if not found.
func (s *CombinedMapSlice) lookup(v MergeKeyValue) *ListItem {
	for _, i := range s.Items {
		if i.KeyValue.Equal(v) {
			return i
		}
	}
	return nil
}

func (s *CombinedMapSlice) upsert(key MergeKeys, l interface{}) (*ListItem, error) {
	// Get the identity of the item
	val, err := key.GetMergeKeyValue(l)
	if err != nil {
		return nil, err
	}

	// Return the item if it exists
	if item := s.lookup(val); item != nil {
		return item, nil
	}

	// Otherwise create a new item and append to the list
	item := &ListItem{
		KeyValue: val,
	}
	s.Items = append(s.Items, item)
	return item, nil
}

// UpsertRecorded adds l to the slice.  If there is already a value of l sharing
// l's merge key in the slice for either the local or remote, set l the recorded value
// Otherwise append a new item to the list with the recorded value.
func (s *CombinedMapSlice) UpsertRecorded(key MergeKeys, l interface{}) error {
	item, err := s.upsert(key, l)
	if err != nil {
		return err
	}
	item.SetRecorded(l)
	return nil
}

// UpsertLocal adds l to the slice.  If there is already a value of l sharing
// l's merge key in the slice for either the recorded or remote, set l the local value
// Otherwise append a new item to the list with the local value.
func (s *CombinedMapSlice) UpsertLocal(key MergeKeys, l interface{}) error {
	item, err := s.upsert(key, l)
	if err != nil {
		return err
	}
	item.SetLocal(l)
	return nil
}

// UpsertRemote adds l to the slice.  If there is already a value of l sharing
// l's merge key in the slice for either the recorded or local, set l the remote value
// Otherwise append a new item to the list with the remote value.
func (s *CombinedMapSlice) UpsertRemote(key MergeKeys, l interface{}) error {
	item, err := s.upsert(key, l)
	if err != nil {
		return err
	}
	item.SetRemote(l)
	return nil
}

// IsDrop returns true if the field represented by e should be dropped from the merged object
func IsDrop(e Element) bool {
	// Specified in the last value recorded value and since deleted from the local
	removed := e.HasRecorded() && !e.HasLocal()

	// Specified locally and explicitly set to null
	setToNil := e.HasLocal() && e.GetLocal() == nil

	return removed || setToNil
}

// IsAdd returns true if the field represented by e should have the local value directly
// added to the merged object instead of merging the recorded, local and remote values
func IsAdd(e Element) bool {
	// If it isn't already present in the remote value and is present in the local value
	return e.HasLocal() && !e.HasRemote()
}

// NewRawElementData returns a new RawElementData, setting IsSet to true for
// non-nil values, and leaving IsSet false for nil values.
// Note: use this only when you want a nil-value to be considered "unspecified"
// (ignore) and not "unset" (deleted).
func NewRawElementData(recorded, local, remote interface{}) RawElementData {
	data := RawElementData{}
	if recorded != nil {
		data.SetRecorded(recorded)
	}
	if local != nil {
		data.SetLocal(local)
	}
	if remote != nil {
		data.SetRemote(remote)
	}
	return data
}

// RawElementData contains the raw recorded, local and remote data
// and metadata about whethere or not each was set
type RawElementData struct {
	HasElementData

	recorded interface{}
	local    interface{}
	remote   interface{}
}

// SetRecorded sets the recorded value
func (b *RawElementData) SetRecorded(value interface{}) {
	b.recorded = value
	b.recordedSet = true
}

// SetLocal sets the local value
func (b *RawElementData) SetLocal(value interface{}) {
	b.local = value
	b.localSet = true
}

// SetRemote sets the remote value
func (b *RawElementData) SetRemote(value interface{}) {
	b.remote = value
	b.remoteSet = true
}

// GetRecorded implements Element.GetRecorded
func (b RawElementData) GetRecorded() interface{} {
	// https://golang.org/doc/faq#nil_error
	if b.recorded == nil {
		return nil
	}
	return b.recorded
}

// GetLocal implements Element.GetLocal
func (b RawElementData) GetLocal() interface{} {
	// https://golang.org/doc/faq#nil_error
	if b.local == nil {
		return nil
	}
	return b.local
}

// GetRemote implements Element.GetRemote
func (b RawElementData) GetRemote() interface{} {
	// https://golang.org/doc/faq#nil_error
	if b.remote == nil {
		return nil
	}
	return b.remote
}

// HasElementData contains whether a field was set in the recorded, local and remote sources
type HasElementData struct {
	recordedSet bool
	localSet    bool
	remoteSet   bool
}

// HasRecorded implements Element.HasRecorded
func (e HasElementData) HasRecorded() bool {
	return e.recordedSet
}

// HasLocal implements Element.HasLocal
func (e HasElementData) HasLocal() bool {
	return e.localSet
}

// HasRemote implements Element.HasRemote
func (e HasElementData) HasRemote() bool {
	return e.remoteSet
}

// ConflictDetector defines the capability to detect conflict. An element can examine remote/recorded value to detect conflict.
type ConflictDetector interface {
	HasConflict() error
}
