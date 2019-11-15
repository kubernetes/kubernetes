// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bson

import (
	"errors"
	"reflect"

	"go.mongodb.org/mongo-driver/bson/bsoncodec"
	"go.mongodb.org/mongo-driver/bson/bsonrw"
)

var primitiveCodecs PrimitiveCodecs

// PrimitiveCodecs is a namespace for all of the default bsoncodec.Codecs for the primitive types
// defined in this package.
type PrimitiveCodecs struct{}

// RegisterPrimitiveCodecs will register the encode and decode methods attached to PrimitiveCodecs
// with the provided RegistryBuilder. if rb is nil, a new empty RegistryBuilder will be created.
func (pc PrimitiveCodecs) RegisterPrimitiveCodecs(rb *bsoncodec.RegistryBuilder) {
	if rb == nil {
		panic(errors.New("argument to RegisterPrimitiveCodecs must not be nil"))
	}

	rb.
		RegisterEncoder(tRawValue, bsoncodec.ValueEncoderFunc(pc.RawValueEncodeValue)).
		RegisterEncoder(tRaw, bsoncodec.ValueEncoderFunc(pc.RawEncodeValue)).
		RegisterDecoder(tRawValue, bsoncodec.ValueDecoderFunc(pc.RawValueDecodeValue)).
		RegisterDecoder(tRaw, bsoncodec.ValueDecoderFunc(pc.RawDecodeValue))
}

// RawValueEncodeValue is the ValueEncoderFunc for RawValue.
func (PrimitiveCodecs) RawValueEncodeValue(ec bsoncodec.EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tRawValue {
		return bsoncodec.ValueEncoderError{Name: "RawValueEncodeValue", Types: []reflect.Type{tRawValue}, Received: val}
	}

	rawvalue := val.Interface().(RawValue)

	return bsonrw.Copier{}.CopyValueFromBytes(vw, rawvalue.Type, rawvalue.Value)
}

// RawValueDecodeValue is the ValueDecoderFunc for RawValue.
func (PrimitiveCodecs) RawValueDecodeValue(dc bsoncodec.DecodeContext, vr bsonrw.ValueReader, val reflect.Value) error {
	if !val.CanSet() || val.Type() != tRawValue {
		return bsoncodec.ValueDecoderError{Name: "RawValueDecodeValue", Types: []reflect.Type{tRawValue}, Received: val}
	}

	t, value, err := bsonrw.Copier{}.CopyValueToBytes(vr)
	if err != nil {
		return err
	}

	val.Set(reflect.ValueOf(RawValue{Type: t, Value: value}))
	return nil
}

// RawEncodeValue is the ValueEncoderFunc for Reader.
func (PrimitiveCodecs) RawEncodeValue(ec bsoncodec.EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tRaw {
		return bsoncodec.ValueEncoderError{Name: "RawEncodeValue", Types: []reflect.Type{tRaw}, Received: val}
	}

	rdr := val.Interface().(Raw)

	return bsonrw.Copier{}.CopyDocumentFromBytes(vw, rdr)
}

// RawDecodeValue is the ValueDecoderFunc for Reader.
func (PrimitiveCodecs) RawDecodeValue(dc bsoncodec.DecodeContext, vr bsonrw.ValueReader, val reflect.Value) error {
	if !val.CanSet() || val.Type() != tRaw {
		return bsoncodec.ValueDecoderError{Name: "RawDecodeValue", Types: []reflect.Type{tRaw}, Received: val}
	}

	if val.IsNil() {
		val.Set(reflect.MakeSlice(val.Type(), 0, 0))
	}

	val.SetLen(0)

	rdr, err := bsonrw.Copier{}.AppendDocumentBytes(val.Interface().(Raw), vr)
	val.Set(reflect.ValueOf(rdr))
	return err
}

func (pc PrimitiveCodecs) encodeRaw(ec bsoncodec.EncodeContext, dw bsonrw.DocumentWriter, raw Raw) error {
	var copier bsonrw.Copier
	elems, err := raw.Elements()
	if err != nil {
		return err
	}
	for _, elem := range elems {
		dvw, err := dw.WriteDocumentElement(elem.Key())
		if err != nil {
			return err
		}

		val := elem.Value()
		err = copier.CopyValueFromBytes(dvw, val.Type, val.Value)
		if err != nil {
			return err
		}
	}

	return dw.WriteDocumentEnd()
}
