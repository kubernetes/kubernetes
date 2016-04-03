// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package search

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
)

// ErrFieldMismatch is returned when a field is to be loaded into a different
// than the one it was stored from, or when a field is missing or unexported in
// the destination struct.
type ErrFieldMismatch struct {
	FieldName string
	Reason    string
}

func (e *ErrFieldMismatch) Error() string {
	return fmt.Sprintf("search: cannot load field %q: %s", e.FieldName, e.Reason)
}

// ErrFacetMismatch is returned when a facet is to be loaded into a different
// type than the one it was stored from, or when a field is missing or
// unexported in the destination struct. StructType is the type of the struct
// pointed to by the destination argument passed to Iterator.Next.
type ErrFacetMismatch struct {
	StructType reflect.Type
	FacetName  string
	Reason     string
}

func (e *ErrFacetMismatch) Error() string {
	return fmt.Sprintf("search: cannot load facet %q into a %q: %s", e.FacetName, e.StructType, e.Reason)
}

// structCodec defines how to convert a given struct to/from a search document.
type structCodec struct {
	// byIndex returns the struct tag for the i'th struct field.
	byIndex []structTag

	// fieldByName returns the index of the struct field for the given field name.
	fieldByName map[string]int

	// facetByName returns the index of the struct field for the given facet name,
	facetByName map[string]int
}

// structTag holds a structured version of each struct field's parsed tag.
type structTag struct {
	name   string
	facet  bool
	ignore bool
}

var (
	codecsMu sync.RWMutex
	codecs   = map[reflect.Type]*structCodec{}
)

func loadCodec(t reflect.Type) (*structCodec, error) {
	codecsMu.RLock()
	codec, ok := codecs[t]
	codecsMu.RUnlock()
	if ok {
		return codec, nil
	}

	codecsMu.Lock()
	defer codecsMu.Unlock()
	if codec, ok := codecs[t]; ok {
		return codec, nil
	}

	codec = &structCodec{
		fieldByName: make(map[string]int),
		facetByName: make(map[string]int),
	}

	for i, I := 0, t.NumField(); i < I; i++ {
		f := t.Field(i)
		name, opts := f.Tag.Get("search"), ""
		if i := strings.Index(name, ","); i != -1 {
			name, opts = name[:i], name[i+1:]
		}
		ignore := false
		if name == "-" {
			ignore = true
		} else if name == "" {
			name = f.Name
		} else if !validFieldName(name) {
			return nil, fmt.Errorf("search: struct tag has invalid field name: %q", name)
		}
		facet := opts == "facet"
		codec.byIndex = append(codec.byIndex, structTag{name: name, facet: facet, ignore: ignore})
		if facet {
			codec.facetByName[name] = i
		} else {
			codec.fieldByName[name] = i
		}
	}

	codecs[t] = codec
	return codec, nil
}

// structFLS adapts a struct to be a FieldLoadSaver.
type structFLS struct {
	v     reflect.Value
	codec *structCodec
}

func (s structFLS) Load(fields []Field, meta *DocumentMetadata) error {
	var err error
	for _, field := range fields {
		i, ok := s.codec.fieldByName[field.Name]
		if !ok {
			// Note the error, but keep going.
			err = &ErrFieldMismatch{
				FieldName: field.Name,
				Reason:    "no such struct field",
			}
			continue

		}
		f := s.v.Field(i)
		if !f.CanSet() {
			// Note the error, but keep going.
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
	if meta == nil {
		return err
	}
	for _, facet := range meta.Facets {
		i, ok := s.codec.facetByName[facet.Name]
		if !ok {
			// Note the error, but keep going.
			if err == nil {
				err = &ErrFacetMismatch{
					StructType: s.v.Type(),
					FacetName:  facet.Name,
					Reason:     "no matching field found",
				}
			}
			continue
		}
		f := s.v.Field(i)
		if !f.CanSet() {
			// Note the error, but keep going.
			if err == nil {
				err = &ErrFacetMismatch{
					StructType: s.v.Type(),
					FacetName:  facet.Name,
					Reason:     "unable to set unexported field of struct",
				}
			}
			continue
		}
		v := reflect.ValueOf(facet.Value)
		if ft, vt := f.Type(), v.Type(); ft != vt {
			if err == nil {
				err = &ErrFacetMismatch{
					StructType: s.v.Type(),
					FacetName:  facet.Name,
					Reason:     fmt.Sprintf("type mismatch: %v for %d data", ft, vt),
				}
				continue
			}
		}
		f.Set(v)
	}
	return err
}

func (s structFLS) Save() ([]Field, *DocumentMetadata, error) {
	fields := make([]Field, 0, len(s.codec.fieldByName))
	var facets []Facet
	for i, tag := range s.codec.byIndex {
		if tag.ignore {
			continue
		}
		f := s.v.Field(i)
		if !f.CanSet() {
			continue
		}
		if tag.facet {
			facets = append(facets, Facet{Name: tag.name, Value: f.Interface()})
		} else {
			fields = append(fields, Field{Name: tag.name, Value: f.Interface()})
		}
	}
	return fields, &DocumentMetadata{Facets: facets}, nil
}

// newStructFLS returns a FieldLoadSaver for the struct pointer p.
func newStructFLS(p interface{}) (FieldLoadSaver, error) {
	v := reflect.ValueOf(p)
	if v.Kind() != reflect.Ptr || v.IsNil() || v.Elem().Kind() != reflect.Struct {
		return nil, ErrInvalidDocumentType
	}
	codec, err := loadCodec(v.Elem().Type())
	if err != nil {
		return nil, err
	}
	return structFLS{v.Elem(), codec}, nil
}

func loadStructWithMeta(dst interface{}, f []Field, meta *DocumentMetadata) error {
	x, err := newStructFLS(dst)
	if err != nil {
		return err
	}
	return x.Load(f, meta)
}

func saveStructWithMeta(src interface{}) ([]Field, *DocumentMetadata, error) {
	x, err := newStructFLS(src)
	if err != nil {
		return nil, nil, err
	}
	return x.Save()
}

// LoadStruct loads the fields from f to dst. dst must be a struct pointer.
func LoadStruct(dst interface{}, f []Field) error {
	return loadStructWithMeta(dst, f, nil)
}

// SaveStruct returns the fields from src as a slice of Field.
// src must be a struct pointer.
func SaveStruct(src interface{}) ([]Field, error) {
	f, _, err := saveStructWithMeta(src)
	return f, err
}
