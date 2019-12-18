// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bson

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"time"

	"go.mongodb.org/mongo-driver/bson/bsoncodec"
	"go.mongodb.org/mongo-driver/bson/bsonrw"
	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/x/bsonx/bsoncore"
)

// ErrNilContext is returned when the provided DecodeContext is nil.
var ErrNilContext = errors.New("DecodeContext cannot be nil")

// ErrNilRegistry is returned when the provided registry is nil.
var ErrNilRegistry = errors.New("Registry cannot be nil")

// RawValue represents a BSON value in byte form. It can be used to hold unprocessed BSON or to
// defer processing of BSON. Type is the BSON type of the value and Value are the raw bytes that
// represent the element.
//
// This type wraps bsoncore.Value for most of it's functionality.
type RawValue struct {
	Type  bsontype.Type
	Value []byte

	r *bsoncodec.Registry
}

// Unmarshal deserializes BSON into the provided val. If RawValue cannot be unmarshaled into val, an
// error is returned. This method will use the registry used to create the RawValue, if the RawValue
// was created from partial BSON processing, or it will use the default registry. Users wishing to
// specify the registry to use should use UnmarshalWithRegistry.
func (rv RawValue) Unmarshal(val interface{}) error {
	reg := rv.r
	if reg == nil {
		reg = DefaultRegistry
	}
	return rv.UnmarshalWithRegistry(reg, val)
}

// Equal compares rv and rv2 and returns true if they are equal.
func (rv RawValue) Equal(rv2 RawValue) bool {
	if rv.Type != rv2.Type {
		return false
	}

	if !bytes.Equal(rv.Value, rv2.Value) {
		return false
	}

	return true
}

// UnmarshalWithRegistry performs the same unmarshalling as Unmarshal but uses the provided registry
// instead of the one attached or the default registry.
func (rv RawValue) UnmarshalWithRegistry(r *bsoncodec.Registry, val interface{}) error {
	if r == nil {
		return ErrNilRegistry
	}

	vr := bsonrw.NewBSONValueReader(rv.Type, rv.Value)
	rval := reflect.ValueOf(val)
	if rval.Kind() != reflect.Ptr {
		return fmt.Errorf("argument to Unmarshal* must be a pointer to a type, but got %v", rval)
	}
	rval = rval.Elem()
	dec, err := r.LookupDecoder(rval.Type())
	if err != nil {
		return err
	}
	return dec.DecodeValue(bsoncodec.DecodeContext{Registry: r}, vr, rval)
}

// UnmarshalWithContext performs the same unmarshalling as Unmarshal but uses the provided DecodeContext
// instead of the one attached or the default registry.
func (rv RawValue) UnmarshalWithContext(dc *bsoncodec.DecodeContext, val interface{}) error {
	if dc == nil {
		return ErrNilContext
	}

	vr := bsonrw.NewBSONValueReader(rv.Type, rv.Value)
	rval := reflect.ValueOf(val)
	if rval.Kind() != reflect.Ptr {
		return fmt.Errorf("argument to Unmarshal* must be a pointer to a type, but got %v", rval)
	}
	rval = rval.Elem()
	dec, err := dc.LookupDecoder(rval.Type())
	if err != nil {
		return err
	}
	return dec.DecodeValue(*dc, vr, rval)
}

func convertFromCoreValue(v bsoncore.Value) RawValue { return RawValue{Type: v.Type, Value: v.Data} }
func convertToCoreValue(v RawValue) bsoncore.Value   { return bsoncore.Value{Type: v.Type, Data: v.Value} }

// Validate ensures the value is a valid BSON value.
func (rv RawValue) Validate() error { return convertToCoreValue(rv).Validate() }

// IsNumber returns true if the type of v is a numeric BSON type.
func (rv RawValue) IsNumber() bool { return convertToCoreValue(rv).IsNumber() }

// String implements the fmt.String interface. This method will return values in extended JSON
// format. If the value is not valid, this returns an empty string
func (rv RawValue) String() string { return convertToCoreValue(rv).String() }

// DebugString outputs a human readable version of Document. It will attempt to stringify the
// valid components of the document even if the entire document is not valid.
func (rv RawValue) DebugString() string { return convertToCoreValue(rv).DebugString() }

// Double returns the float64 value for this element.
// It panics if e's BSON type is not bsontype.Double.
func (rv RawValue) Double() float64 { return convertToCoreValue(rv).Double() }

// DoubleOK is the same as Double, but returns a boolean instead of panicking.
func (rv RawValue) DoubleOK() (float64, bool) { return convertToCoreValue(rv).DoubleOK() }

// StringValue returns the string value for this element.
// It panics if e's BSON type is not bsontype.String.
//
// NOTE: This method is called StringValue to avoid a collision with the String method which
// implements the fmt.Stringer interface.
func (rv RawValue) StringValue() string { return convertToCoreValue(rv).StringValue() }

// StringValueOK is the same as StringValue, but returns a boolean instead of
// panicking.
func (rv RawValue) StringValueOK() (string, bool) { return convertToCoreValue(rv).StringValueOK() }

// Document returns the BSON document the Value represents as a Document. It panics if the
// value is a BSON type other than document.
func (rv RawValue) Document() Raw { return Raw(convertToCoreValue(rv).Document()) }

// DocumentOK is the same as Document, except it returns a boolean
// instead of panicking.
func (rv RawValue) DocumentOK() (Raw, bool) {
	doc, ok := convertToCoreValue(rv).DocumentOK()
	return Raw(doc), ok
}

// Array returns the BSON array the Value represents as an Array. It panics if the
// value is a BSON type other than array.
func (rv RawValue) Array() Raw { return Raw(convertToCoreValue(rv).Array()) }

// ArrayOK is the same as Array, except it returns a boolean instead
// of panicking.
func (rv RawValue) ArrayOK() (Raw, bool) {
	doc, ok := convertToCoreValue(rv).ArrayOK()
	return Raw(doc), ok
}

// Binary returns the BSON binary value the Value represents. It panics if the value is a BSON type
// other than binary.
func (rv RawValue) Binary() (subtype byte, data []byte) { return convertToCoreValue(rv).Binary() }

// BinaryOK is the same as Binary, except it returns a boolean instead of
// panicking.
func (rv RawValue) BinaryOK() (subtype byte, data []byte, ok bool) {
	return convertToCoreValue(rv).BinaryOK()
}

// ObjectID returns the BSON objectid value the Value represents. It panics if the value is a BSON
// type other than objectid.
func (rv RawValue) ObjectID() primitive.ObjectID { return convertToCoreValue(rv).ObjectID() }

// ObjectIDOK is the same as ObjectID, except it returns a boolean instead of
// panicking.
func (rv RawValue) ObjectIDOK() (primitive.ObjectID, bool) { return convertToCoreValue(rv).ObjectIDOK() }

// Boolean returns the boolean value the Value represents. It panics if the
// value is a BSON type other than boolean.
func (rv RawValue) Boolean() bool { return convertToCoreValue(rv).Boolean() }

// BooleanOK is the same as Boolean, except it returns a boolean instead of
// panicking.
func (rv RawValue) BooleanOK() (bool, bool) { return convertToCoreValue(rv).BooleanOK() }

// DateTime returns the BSON datetime value the Value represents as a
// unix timestamp. It panics if the value is a BSON type other than datetime.
func (rv RawValue) DateTime() int64 { return convertToCoreValue(rv).DateTime() }

// DateTimeOK is the same as DateTime, except it returns a boolean instead of
// panicking.
func (rv RawValue) DateTimeOK() (int64, bool) { return convertToCoreValue(rv).DateTimeOK() }

// Time returns the BSON datetime value the Value represents. It panics if the value is a BSON
// type other than datetime.
func (rv RawValue) Time() time.Time { return convertToCoreValue(rv).Time() }

// TimeOK is the same as Time, except it returns a boolean instead of
// panicking.
func (rv RawValue) TimeOK() (time.Time, bool) { return convertToCoreValue(rv).TimeOK() }

// Regex returns the BSON regex value the Value represents. It panics if the value is a BSON
// type other than regex.
func (rv RawValue) Regex() (pattern, options string) { return convertToCoreValue(rv).Regex() }

// RegexOK is the same as Regex, except it returns a boolean instead of
// panicking.
func (rv RawValue) RegexOK() (pattern, options string, ok bool) {
	return convertToCoreValue(rv).RegexOK()
}

// DBPointer returns the BSON dbpointer value the Value represents. It panics if the value is a BSON
// type other than DBPointer.
func (rv RawValue) DBPointer() (string, primitive.ObjectID) { return convertToCoreValue(rv).DBPointer() }

// DBPointerOK is the same as DBPoitner, except that it returns a boolean
// instead of panicking.
func (rv RawValue) DBPointerOK() (string, primitive.ObjectID, bool) {
	return convertToCoreValue(rv).DBPointerOK()
}

// JavaScript returns the BSON JavaScript code value the Value represents. It panics if the value is
// a BSON type other than JavaScript code.
func (rv RawValue) JavaScript() string { return convertToCoreValue(rv).JavaScript() }

// JavaScriptOK is the same as Javascript, excepti that it returns a boolean
// instead of panicking.
func (rv RawValue) JavaScriptOK() (string, bool) { return convertToCoreValue(rv).JavaScriptOK() }

// Symbol returns the BSON symbol value the Value represents. It panics if the value is a BSON
// type other than symbol.
func (rv RawValue) Symbol() string { return convertToCoreValue(rv).Symbol() }

// SymbolOK is the same as Symbol, excepti that it returns a boolean
// instead of panicking.
func (rv RawValue) SymbolOK() (string, bool) { return convertToCoreValue(rv).SymbolOK() }

// CodeWithScope returns the BSON JavaScript code with scope the Value represents.
// It panics if the value is a BSON type other than JavaScript code with scope.
func (rv RawValue) CodeWithScope() (string, Raw) {
	code, scope := convertToCoreValue(rv).CodeWithScope()
	return code, Raw(scope)
}

// CodeWithScopeOK is the same as CodeWithScope, except that it returns a boolean instead of
// panicking.
func (rv RawValue) CodeWithScopeOK() (string, Raw, bool) {
	code, scope, ok := convertToCoreValue(rv).CodeWithScopeOK()
	return code, Raw(scope), ok
}

// Int32 returns the int32 the Value represents. It panics if the value is a BSON type other than
// int32.
func (rv RawValue) Int32() int32 { return convertToCoreValue(rv).Int32() }

// Int32OK is the same as Int32, except that it returns a boolean instead of
// panicking.
func (rv RawValue) Int32OK() (int32, bool) { return convertToCoreValue(rv).Int32OK() }

// Timestamp returns the BSON timestamp value the Value represents. It panics if the value is a
// BSON type other than timestamp.
func (rv RawValue) Timestamp() (t, i uint32) { return convertToCoreValue(rv).Timestamp() }

// TimestampOK is the same as Timestamp, except that it returns a boolean
// instead of panicking.
func (rv RawValue) TimestampOK() (t, i uint32, ok bool) { return convertToCoreValue(rv).TimestampOK() }

// Int64 returns the int64 the Value represents. It panics if the value is a BSON type other than
// int64.
func (rv RawValue) Int64() int64 { return convertToCoreValue(rv).Int64() }

// Int64OK is the same as Int64, except that it returns a boolean instead of
// panicking.
func (rv RawValue) Int64OK() (int64, bool) { return convertToCoreValue(rv).Int64OK() }

// Decimal128 returns the decimal the Value represents. It panics if the value is a BSON type other than
// decimal.
func (rv RawValue) Decimal128() primitive.Decimal128 { return convertToCoreValue(rv).Decimal128() }

// Decimal128OK is the same as Decimal128, except that it returns a boolean
// instead of panicking.
func (rv RawValue) Decimal128OK() (primitive.Decimal128, bool) {
	return convertToCoreValue(rv).Decimal128OK()
}
