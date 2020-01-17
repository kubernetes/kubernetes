// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsoncodec

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"

	"go.mongodb.org/mongo-driver/bson/bsonrw"
	"go.mongodb.org/mongo-driver/bson/bsontype"
)

var defaultStructCodec = &StructCodec{
	cache:  make(map[reflect.Type]*structDescription),
	parser: DefaultStructTagParser,
}

// Zeroer allows custom struct types to implement a report of zero
// state. All struct types that don't implement Zeroer or where IsZero
// returns false are considered to be not zero.
type Zeroer interface {
	IsZero() bool
}

// StructCodec is the Codec used for struct values.
type StructCodec struct {
	cache  map[reflect.Type]*structDescription
	l      sync.RWMutex
	parser StructTagParser
}

var _ ValueEncoder = &StructCodec{}
var _ ValueDecoder = &StructCodec{}

// NewStructCodec returns a StructCodec that uses p for struct tag parsing.
func NewStructCodec(p StructTagParser) (*StructCodec, error) {
	if p == nil {
		return nil, errors.New("a StructTagParser must be provided to NewStructCodec")
	}

	return &StructCodec{
		cache:  make(map[reflect.Type]*structDescription),
		parser: p,
	}, nil
}

// EncodeValue handles encoding generic struct types.
func (sc *StructCodec) EncodeValue(r EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Kind() != reflect.Struct {
		return ValueEncoderError{Name: "StructCodec.EncodeValue", Kinds: []reflect.Kind{reflect.Struct}, Received: val}
	}

	sd, err := sc.describeStruct(r.Registry, val.Type())
	if err != nil {
		return err
	}

	dw, err := vw.WriteDocument()
	if err != nil {
		return err
	}
	var rv reflect.Value
	for _, desc := range sd.fl {
		if desc.inline == nil {
			rv = val.Field(desc.idx)
		} else {
			rv = val.FieldByIndex(desc.inline)
		}

		if desc.encoder == nil {
			return ErrNoEncoder{Type: rv.Type()}
		}

		encoder := desc.encoder

		iszero := sc.isZero
		if iz, ok := encoder.(CodecZeroer); ok {
			iszero = iz.IsTypeZero
		}

		if desc.omitEmpty && iszero(rv.Interface()) {
			continue
		}

		vw2, err := dw.WriteDocumentElement(desc.name)
		if err != nil {
			return err
		}

		ectx := EncodeContext{Registry: r.Registry, MinSize: desc.minSize}
		err = encoder.EncodeValue(ectx, vw2, rv)
		if err != nil {
			return err
		}
	}

	if sd.inlineMap >= 0 {
		rv := val.Field(sd.inlineMap)
		collisionFn := func(key string) bool {
			_, exists := sd.fm[key]
			return exists
		}

		return defaultValueEncoders.mapEncodeValue(r, dw, rv, collisionFn)
	}

	return dw.WriteDocumentEnd()
}

// DecodeValue implements the Codec interface.
// By default, map types in val will not be cleared. If a map has existing key/value pairs, it will be extended with the new ones from vr.
// For slices, the decoder will set the length of the slice to zero and append all elements. The underlying array will not be cleared.
func (sc *StructCodec) DecodeValue(r DecodeContext, vr bsonrw.ValueReader, val reflect.Value) error {
	if !val.CanSet() || val.Kind() != reflect.Struct {
		return ValueDecoderError{Name: "StructCodec.DecodeValue", Kinds: []reflect.Kind{reflect.Struct}, Received: val}
	}

	switch vr.Type() {
	case bsontype.Type(0), bsontype.EmbeddedDocument:
	default:
		return fmt.Errorf("cannot decode %v into a %s", vr.Type(), val.Type())
	}

	sd, err := sc.describeStruct(r.Registry, val.Type())
	if err != nil {
		return err
	}

	var decoder ValueDecoder
	var inlineMap reflect.Value
	if sd.inlineMap >= 0 {
		inlineMap = val.Field(sd.inlineMap)
		if inlineMap.IsNil() {
			inlineMap.Set(reflect.MakeMap(inlineMap.Type()))
		}
		decoder, err = r.LookupDecoder(inlineMap.Type().Elem())
		if err != nil {
			return err
		}
	}

	dr, err := vr.ReadDocument()
	if err != nil {
		return err
	}

	for {
		name, vr, err := dr.ReadElement()
		if err == bsonrw.ErrEOD {
			break
		}
		if err != nil {
			return err
		}

		fd, exists := sd.fm[name]
		if !exists {
			// if the original name isn't found in the struct description, try again with the name in lowercase
			// this could match if a BSON tag isn't specified because by default, describeStruct lowercases all field
			// names
			fd, exists = sd.fm[strings.ToLower(name)]
		}

		if !exists {
			if sd.inlineMap < 0 {
				// The encoding/json package requires a flag to return on error for non-existent fields.
				// This functionality seems appropriate for the struct codec.
				err = vr.Skip()
				if err != nil {
					return err
				}
				continue
			}

			elem := reflect.New(inlineMap.Type().Elem()).Elem()
			err = decoder.DecodeValue(r, vr, elem)
			if err != nil {
				return err
			}
			inlineMap.SetMapIndex(reflect.ValueOf(name), elem)
			continue
		}

		var field reflect.Value
		if fd.inline == nil {
			field = val.Field(fd.idx)
		} else {
			field = val.FieldByIndex(fd.inline)
		}

		if !field.CanSet() { // Being settable is a super set of being addressable.
			return fmt.Errorf("cannot decode element '%s' into field %v; it is not settable", name, field)
		}
		if field.Kind() == reflect.Ptr && field.IsNil() {
			field.Set(reflect.New(field.Type().Elem()))
		}
		field = field.Addr()

		dctx := DecodeContext{Registry: r.Registry, Truncate: fd.truncate || r.Truncate}
		if fd.decoder == nil {
			return ErrNoDecoder{Type: field.Elem().Type()}
		}

		if decoder, ok := fd.decoder.(ValueDecoder); ok {
			err = decoder.DecodeValue(dctx, vr, field.Elem())
			if err != nil {
				return err
			}
			continue
		}
		err = fd.decoder.DecodeValue(dctx, vr, field)
		if err != nil {
			return err
		}
	}

	return nil
}

func (sc *StructCodec) isZero(i interface{}) bool {
	v := reflect.ValueOf(i)

	// check the value validity
	if !v.IsValid() {
		return true
	}

	if z, ok := v.Interface().(Zeroer); ok && (v.Kind() != reflect.Ptr || !v.IsNil()) {
		return z.IsZero()
	}

	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}

	return false
}

type structDescription struct {
	fm        map[string]fieldDescription
	fl        []fieldDescription
	inlineMap int
}

type fieldDescription struct {
	name      string
	idx       int
	omitEmpty bool
	minSize   bool
	truncate  bool
	inline    []int
	encoder   ValueEncoder
	decoder   ValueDecoder
}

func (sc *StructCodec) describeStruct(r *Registry, t reflect.Type) (*structDescription, error) {
	// We need to analyze the struct, including getting the tags, collecting
	// information about inlining, and create a map of the field name to the field.
	sc.l.RLock()
	ds, exists := sc.cache[t]
	sc.l.RUnlock()
	if exists {
		return ds, nil
	}

	numFields := t.NumField()
	sd := &structDescription{
		fm:        make(map[string]fieldDescription, numFields),
		fl:        make([]fieldDescription, 0, numFields),
		inlineMap: -1,
	}

	for i := 0; i < numFields; i++ {
		sf := t.Field(i)
		if sf.PkgPath != "" {
			// unexported, ignore
			continue
		}

		encoder, err := r.LookupEncoder(sf.Type)
		if err != nil {
			encoder = nil
		}
		decoder, err := r.LookupDecoder(sf.Type)
		if err != nil {
			decoder = nil
		}

		description := fieldDescription{idx: i, encoder: encoder, decoder: decoder}

		stags, err := sc.parser.ParseStructTags(sf)
		if err != nil {
			return nil, err
		}
		if stags.Skip {
			continue
		}
		description.name = stags.Name
		description.omitEmpty = stags.OmitEmpty
		description.minSize = stags.MinSize
		description.truncate = stags.Truncate

		if stags.Inline {
			switch sf.Type.Kind() {
			case reflect.Map:
				if sd.inlineMap >= 0 {
					return nil, errors.New("(struct " + t.String() + ") multiple inline maps")
				}
				if sf.Type.Key() != tString {
					return nil, errors.New("(struct " + t.String() + ") inline map must have a string keys")
				}
				sd.inlineMap = description.idx
			case reflect.Struct:
				inlinesf, err := sc.describeStruct(r, sf.Type)
				if err != nil {
					return nil, err
				}
				for _, fd := range inlinesf.fl {
					if _, exists := sd.fm[fd.name]; exists {
						return nil, fmt.Errorf("(struct %s) duplicated key %s", t.String(), fd.name)
					}
					if fd.inline == nil {
						fd.inline = []int{i, fd.idx}
					} else {
						fd.inline = append([]int{i}, fd.inline...)
					}
					sd.fm[fd.name] = fd
					sd.fl = append(sd.fl, fd)
				}
			default:
				return nil, fmt.Errorf("(struct %s) inline fields must be either a struct or a map", t.String())
			}
			continue
		}

		if _, exists := sd.fm[description.name]; exists {
			return nil, fmt.Errorf("struct %s) duplicated key %s", t.String(), description.name)
		}

		sd.fm[description.name] = description
		sd.fl = append(sd.fl, description)
	}

	sc.l.Lock()
	sc.cache[t] = sd
	sc.l.Unlock()

	return sd, nil
}
