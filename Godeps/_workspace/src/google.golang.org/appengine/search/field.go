// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package search

import (
	"fmt"
	"reflect"
)

// Field is a name/value pair. A search index's document can be loaded and
// saved as a sequence of Fields.
type Field struct {
	// Name is the field name.
	Name string
	// Value is the field value. The valid types are:
	//  - string,
	//  - search.Atom,
	//  - search.HTML,
	//  - time.Time (stored with millisecond precision),
	//  - float64,
	//  - GeoPoint.
	Value interface{}
	// Language is a two-letter ISO 693-1 code for the field's language,
	// defaulting to "en" if nothing is specified. It may only be specified for
	// fields of type string and search.HTML.
	Language string
	// Derived marks fields that were calculated as a result of a
	// FieldExpression provided to Search. This field is ignored when saving a
	// document.
	Derived bool
}

// DocumentMetadata is a struct containing information describing a given document.
type DocumentMetadata struct {
	// Rank is an integer specifying the order the document will be returned in
	// search results. If zero, the rank will be set to the number of seconds since
	// 2011-01-01 00:00:00 UTC when being Put into an index.
	Rank int
}

// FieldLoadSaver can be converted from and to a slice of Fields
// with additional document metadata.
type FieldLoadSaver interface {
	Load([]Field, *DocumentMetadata) error
	Save() ([]Field, *DocumentMetadata, error)
}

// FieldList converts a []Field to implement FieldLoadSaver.
type FieldList []Field

// Load loads all of the provided fields into l.
// It does not first reset *l to an empty slice.
func (l *FieldList) Load(f []Field, _ *DocumentMetadata) error {
	*l = append(*l, f...)
	return nil
}

// Save returns all of l's fields as a slice of Fields.
func (l *FieldList) Save() ([]Field, *DocumentMetadata, error) {
	return *l, nil, nil
}

var _ FieldLoadSaver = (*FieldList)(nil)

// structFLS adapts a struct to be a FieldLoadSaver.
type structFLS struct {
	reflect.Value
}

func (s structFLS) Load(fields []Field, _ *DocumentMetadata) (err error) {
	for _, field := range fields {
		f := s.FieldByName(field.Name)
		if !f.IsValid() {
			err = &ErrFieldMismatch{
				FieldName: field.Name,
				Reason:    "no such struct field",
			}
			continue
		}
		if !f.CanSet() {
			err = &ErrFieldMismatch{
				FieldName: field.Name,
				Reason:    "cannot set struct field",
			}
			continue
		}
		v := reflect.ValueOf(field.Value)
		if ft, vt := f.Type(), v.Type(); ft != vt {
			err = &ErrFieldMismatch{
				FieldName: field.Name,
				Reason:    fmt.Sprintf("type mismatch: %v for %v data", ft, vt),
			}
			continue
		}
		f.Set(v)
	}
	return err
}

func (s structFLS) Save() ([]Field, *DocumentMetadata, error) {
	fields := make([]Field, 0, s.NumField())
	for i := 0; i < s.NumField(); i++ {
		f := s.Field(i)
		if !f.CanSet() {
			continue
		}
		fields = append(fields, Field{
			Name:  s.Type().Field(i).Name,
			Value: f.Interface(),
		})
	}
	return fields, nil, nil
}

// newStructFLS returns a FieldLoadSaver for the struct pointer p.
func newStructFLS(p interface{}) (FieldLoadSaver, error) {
	v := reflect.ValueOf(p)
	if v.Kind() != reflect.Ptr || v.IsNil() || v.Elem().Kind() != reflect.Struct {
		return nil, ErrInvalidDocumentType
	}
	return structFLS{v.Elem()}, nil
}

// LoadStruct loads the fields from f to dst. dst must be a struct pointer.
func LoadStruct(dst interface{}, f []Field) error {
	x, err := newStructFLS(dst)
	if err != nil {
		return err
	}
	return x.Load(f, nil)
}

// SaveStruct returns the fields from src as a slice of Field.
// src must be a struct pointer.
func SaveStruct(src interface{}) ([]Field, error) {
	x, err := newStructFLS(src)
	if err != nil {
		return nil, err
	}
	fs, _, err := x.Save()
	return fs, err
}
