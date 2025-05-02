// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
)

// Tag represents CBOR tag data, including tag number and unmarshaled tag content. Marshaling and
// unmarshaling of tag content is subject to any encode and decode options that would apply to
// enclosed data item if it were to appear outside of a tag.
type Tag struct {
	Number  uint64
	Content any
}

// RawTag represents CBOR tag data, including tag number and raw tag content.
// RawTag implements Unmarshaler and Marshaler interfaces.
type RawTag struct {
	Number  uint64
	Content RawMessage
}

// UnmarshalCBOR sets *t with tag number and raw tag content copied from data.
//
// Deprecated: No longer used by this codec; kept for compatibility
// with user apps that directly call this function.
func (t *RawTag) UnmarshalCBOR(data []byte) error {
	if t == nil {
		return errors.New("cbor.RawTag: UnmarshalCBOR on nil pointer")
	}

	d := decoder{data: data, dm: defaultDecMode}

	// Check if data is a well-formed CBOR data item.
	// RawTag.UnmarshalCBOR() is exported, so
	// the codec needs to support same behavior for:
	// - Unmarshal(data, *RawTag)
	// - RawTag.UnmarshalCBOR(data)
	err := d.wellformed(false, false)
	if err != nil {
		return err
	}

	return t.unmarshalCBOR(data)
}

// unmarshalCBOR sets *t with tag number and raw tag content copied from data.
// This function assumes data is well-formed, and does not perform bounds checking.
// This function is called by Unmarshal().
func (t *RawTag) unmarshalCBOR(data []byte) error {
	if t == nil {
		return errors.New("cbor.RawTag: UnmarshalCBOR on nil pointer")
	}

	// Decoding CBOR null and undefined to cbor.RawTag is no-op.
	if len(data) == 1 && (data[0] == 0xf6 || data[0] == 0xf7) {
		return nil
	}

	d := decoder{data: data, dm: defaultDecMode}

	// Unmarshal tag number.
	typ, _, num := d.getHead()
	if typ != cborTypeTag {
		return &UnmarshalTypeError{CBORType: typ.String(), GoType: typeRawTag.String()}
	}
	t.Number = num

	// Unmarshal tag content.
	c := d.data[d.off:]
	t.Content = make([]byte, len(c))
	copy(t.Content, c)
	return nil
}

// MarshalCBOR returns CBOR encoding of t.
func (t RawTag) MarshalCBOR() ([]byte, error) {
	if t.Number == 0 && len(t.Content) == 0 {
		// Marshal uninitialized cbor.RawTag
		b := make([]byte, len(cborNil))
		copy(b, cborNil)
		return b, nil
	}

	e := getEncodeBuffer()

	encodeHead(e, byte(cborTypeTag), t.Number)

	content := t.Content
	if len(content) == 0 {
		content = cborNil
	}

	buf := make([]byte, len(e.Bytes())+len(content))
	n := copy(buf, e.Bytes())
	copy(buf[n:], content)

	putEncodeBuffer(e)
	return buf, nil
}

// DecTagMode specifies how decoder handles tag number.
type DecTagMode int

const (
	// DecTagIgnored makes decoder ignore tag number (skips if present).
	DecTagIgnored DecTagMode = iota

	// DecTagOptional makes decoder verify tag number if it's present.
	DecTagOptional

	// DecTagRequired makes decoder verify tag number and tag number must be present.
	DecTagRequired

	maxDecTagMode
)

func (dtm DecTagMode) valid() bool {
	return dtm >= 0 && dtm < maxDecTagMode
}

// EncTagMode specifies how encoder handles tag number.
type EncTagMode int

const (
	// EncTagNone makes encoder not encode tag number.
	EncTagNone EncTagMode = iota

	// EncTagRequired makes encoder encode tag number.
	EncTagRequired

	maxEncTagMode
)

func (etm EncTagMode) valid() bool {
	return etm >= 0 && etm < maxEncTagMode
}

// TagOptions specifies how encoder and decoder handle tag number.
type TagOptions struct {
	DecTag DecTagMode
	EncTag EncTagMode
}

// TagSet is an interface to add and remove tag info.  It is used by EncMode and DecMode
// to provide CBOR tag support.
type TagSet interface {
	// Add adds given tag number(s), content type, and tag options to TagSet.
	Add(opts TagOptions, contentType reflect.Type, num uint64, nestedNum ...uint64) error

	// Remove removes given tag content type from TagSet.
	Remove(contentType reflect.Type)

	tagProvider
}

type tagProvider interface {
	getTagItemFromType(t reflect.Type) *tagItem
	getTypeFromTagNum(num []uint64) reflect.Type
}

type tagItem struct {
	num         []uint64
	cborTagNum  []byte
	contentType reflect.Type
	opts        TagOptions
}

func (t *tagItem) equalTagNum(num []uint64) bool {
	// Fast path to compare 1 tag number
	if len(t.num) == 1 && len(num) == 1 && t.num[0] == num[0] {
		return true
	}

	if len(t.num) != len(num) {
		return false
	}

	for i := 0; i < len(t.num); i++ {
		if t.num[i] != num[i] {
			return false
		}
	}

	return true
}

type (
	tagSet map[reflect.Type]*tagItem

	syncTagSet struct {
		sync.RWMutex
		t tagSet
	}
)

func (t tagSet) getTagItemFromType(typ reflect.Type) *tagItem {
	return t[typ]
}

func (t tagSet) getTypeFromTagNum(num []uint64) reflect.Type {
	for typ, tag := range t {
		if tag.equalTagNum(num) {
			return typ
		}
	}
	return nil
}

// NewTagSet returns TagSet (safe for concurrency).
func NewTagSet() TagSet {
	return &syncTagSet{t: make(map[reflect.Type]*tagItem)}
}

// Add adds given tag number(s), content type, and tag options to TagSet.
func (t *syncTagSet) Add(opts TagOptions, contentType reflect.Type, num uint64, nestedNum ...uint64) error {
	if contentType == nil {
		return errors.New("cbor: cannot add nil content type to TagSet")
	}
	for contentType.Kind() == reflect.Pointer {
		contentType = contentType.Elem()
	}
	tag, err := newTagItem(opts, contentType, num, nestedNum...)
	if err != nil {
		return err
	}
	t.Lock()
	defer t.Unlock()
	for typ, ti := range t.t {
		if typ == contentType {
			return errors.New("cbor: content type " + contentType.String() + " already exists in TagSet")
		}
		if ti.equalTagNum(tag.num) {
			return fmt.Errorf("cbor: tag number %v already exists in TagSet", tag.num)
		}
	}
	t.t[contentType] = tag
	return nil
}

// Remove removes given tag content type from TagSet.
func (t *syncTagSet) Remove(contentType reflect.Type) {
	for contentType.Kind() == reflect.Pointer {
		contentType = contentType.Elem()
	}
	t.Lock()
	delete(t.t, contentType)
	t.Unlock()
}

func (t *syncTagSet) getTagItemFromType(typ reflect.Type) *tagItem {
	t.RLock()
	ti := t.t[typ]
	t.RUnlock()
	return ti
}

func (t *syncTagSet) getTypeFromTagNum(num []uint64) reflect.Type {
	t.RLock()
	rt := t.t.getTypeFromTagNum(num)
	t.RUnlock()
	return rt
}

func newTagItem(opts TagOptions, contentType reflect.Type, num uint64, nestedNum ...uint64) (*tagItem, error) {
	if opts.DecTag == DecTagIgnored && opts.EncTag == EncTagNone {
		return nil, errors.New("cbor: cannot add tag with DecTagIgnored and EncTagNone options to TagSet")
	}
	if contentType.PkgPath() == "" || contentType.Kind() == reflect.Interface {
		return nil, errors.New("cbor: can only add named types to TagSet, got " + contentType.String())
	}
	if contentType == typeTime {
		return nil, errors.New("cbor: cannot add time.Time to TagSet, use EncOptions.TimeTag and DecOptions.TimeTag instead")
	}
	if contentType == typeBigInt {
		return nil, errors.New("cbor: cannot add big.Int to TagSet, it's built-in and supported automatically")
	}
	if contentType == typeTag {
		return nil, errors.New("cbor: cannot add cbor.Tag to TagSet")
	}
	if contentType == typeRawTag {
		return nil, errors.New("cbor: cannot add cbor.RawTag to TagSet")
	}
	if num == 0 || num == 1 {
		return nil, errors.New("cbor: cannot add tag number 0 or 1 to TagSet, use EncOptions.TimeTag and DecOptions.TimeTag instead")
	}
	if num == 2 || num == 3 {
		return nil, errors.New("cbor: cannot add tag number 2 or 3 to TagSet, it's built-in and supported automatically")
	}
	if num == tagNumSelfDescribedCBOR {
		return nil, errors.New("cbor: cannot add tag number 55799 to TagSet, it's built-in and ignored automatically")
	}

	te := tagItem{num: []uint64{num}, opts: opts, contentType: contentType}
	te.num = append(te.num, nestedNum...)

	// Cache encoded tag numbers
	e := getEncodeBuffer()
	for _, n := range te.num {
		encodeHead(e, byte(cborTypeTag), n)
	}
	te.cborTagNum = make([]byte, e.Len())
	copy(te.cborTagNum, e.Bytes())
	putEncodeBuffer(e)

	return &te, nil
}

var (
	typeTag    = reflect.TypeOf(Tag{})
	typeRawTag = reflect.TypeOf(RawTag{})
)

// WrongTagError describes mismatch between CBOR tag and registered tag.
type WrongTagError struct {
	RegisteredType   reflect.Type
	RegisteredTagNum []uint64
	TagNum           []uint64
}

func (e *WrongTagError) Error() string {
	return fmt.Sprintf("cbor: wrong tag number for %s, got %v, expected %v", e.RegisteredType.String(), e.TagNum, e.RegisteredTagNum)
}
