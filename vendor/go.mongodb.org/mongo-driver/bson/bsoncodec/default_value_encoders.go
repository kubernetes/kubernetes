// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsoncodec

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/url"
	"reflect"
	"sync"
	"time"

	"go.mongodb.org/mongo-driver/bson/bsonrw"
	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/x/bsonx/bsoncore"
)

var defaultValueEncoders DefaultValueEncoders

var bvwPool = bsonrw.NewBSONValueWriterPool()

var sliceWriterPool = sync.Pool{
	New: func() interface{} {
		sw := make(bsonrw.SliceWriter, 0, 0)
		return &sw
	},
}

func encodeElement(ec EncodeContext, dw bsonrw.DocumentWriter, e primitive.E) error {
	vw, err := dw.WriteDocumentElement(e.Key)
	if err != nil {
		return err
	}

	if e.Value == nil {
		return vw.WriteNull()
	}
	encoder, err := ec.LookupEncoder(reflect.TypeOf(e.Value))
	if err != nil {
		return err
	}

	err = encoder.EncodeValue(ec, vw, reflect.ValueOf(e.Value))
	if err != nil {
		return err
	}
	return nil
}

// DefaultValueEncoders is a namespace type for the default ValueEncoders used
// when creating a registry.
type DefaultValueEncoders struct{}

// RegisterDefaultEncoders will register the encoder methods attached to DefaultValueEncoders with
// the provided RegistryBuilder.
func (dve DefaultValueEncoders) RegisterDefaultEncoders(rb *RegistryBuilder) {
	if rb == nil {
		panic(errors.New("argument to RegisterDefaultEncoders must not be nil"))
	}
	rb.
		RegisterEncoder(tByteSlice, ValueEncoderFunc(dve.ByteSliceEncodeValue)).
		RegisterEncoder(tTime, ValueEncoderFunc(dve.TimeEncodeValue)).
		RegisterEncoder(tEmpty, ValueEncoderFunc(dve.EmptyInterfaceEncodeValue)).
		RegisterEncoder(tOID, ValueEncoderFunc(dve.ObjectIDEncodeValue)).
		RegisterEncoder(tDecimal, ValueEncoderFunc(dve.Decimal128EncodeValue)).
		RegisterEncoder(tJSONNumber, ValueEncoderFunc(dve.JSONNumberEncodeValue)).
		RegisterEncoder(tURL, ValueEncoderFunc(dve.URLEncodeValue)).
		RegisterEncoder(tValueMarshaler, ValueEncoderFunc(dve.ValueMarshalerEncodeValue)).
		RegisterEncoder(tMarshaler, ValueEncoderFunc(dve.MarshalerEncodeValue)).
		RegisterEncoder(tProxy, ValueEncoderFunc(dve.ProxyEncodeValue)).
		RegisterEncoder(tJavaScript, ValueEncoderFunc(dve.JavaScriptEncodeValue)).
		RegisterEncoder(tSymbol, ValueEncoderFunc(dve.SymbolEncodeValue)).
		RegisterEncoder(tBinary, ValueEncoderFunc(dve.BinaryEncodeValue)).
		RegisterEncoder(tUndefined, ValueEncoderFunc(dve.UndefinedEncodeValue)).
		RegisterEncoder(tDateTime, ValueEncoderFunc(dve.DateTimeEncodeValue)).
		RegisterEncoder(tNull, ValueEncoderFunc(dve.NullEncodeValue)).
		RegisterEncoder(tRegex, ValueEncoderFunc(dve.RegexEncodeValue)).
		RegisterEncoder(tDBPointer, ValueEncoderFunc(dve.DBPointerEncodeValue)).
		RegisterEncoder(tTimestamp, ValueEncoderFunc(dve.TimestampEncodeValue)).
		RegisterEncoder(tMinKey, ValueEncoderFunc(dve.MinKeyEncodeValue)).
		RegisterEncoder(tMaxKey, ValueEncoderFunc(dve.MaxKeyEncodeValue)).
		RegisterEncoder(tCoreDocument, ValueEncoderFunc(dve.CoreDocumentEncodeValue)).
		RegisterEncoder(tCodeWithScope, ValueEncoderFunc(dve.CodeWithScopeEncodeValue)).
		RegisterDefaultEncoder(reflect.Bool, ValueEncoderFunc(dve.BooleanEncodeValue)).
		RegisterDefaultEncoder(reflect.Int, ValueEncoderFunc(dve.IntEncodeValue)).
		RegisterDefaultEncoder(reflect.Int8, ValueEncoderFunc(dve.IntEncodeValue)).
		RegisterDefaultEncoder(reflect.Int16, ValueEncoderFunc(dve.IntEncodeValue)).
		RegisterDefaultEncoder(reflect.Int32, ValueEncoderFunc(dve.IntEncodeValue)).
		RegisterDefaultEncoder(reflect.Int64, ValueEncoderFunc(dve.IntEncodeValue)).
		RegisterDefaultEncoder(reflect.Uint, ValueEncoderFunc(dve.UintEncodeValue)).
		RegisterDefaultEncoder(reflect.Uint8, ValueEncoderFunc(dve.UintEncodeValue)).
		RegisterDefaultEncoder(reflect.Uint16, ValueEncoderFunc(dve.UintEncodeValue)).
		RegisterDefaultEncoder(reflect.Uint32, ValueEncoderFunc(dve.UintEncodeValue)).
		RegisterDefaultEncoder(reflect.Uint64, ValueEncoderFunc(dve.UintEncodeValue)).
		RegisterDefaultEncoder(reflect.Float32, ValueEncoderFunc(dve.FloatEncodeValue)).
		RegisterDefaultEncoder(reflect.Float64, ValueEncoderFunc(dve.FloatEncodeValue)).
		RegisterDefaultEncoder(reflect.Array, ValueEncoderFunc(dve.ArrayEncodeValue)).
		RegisterDefaultEncoder(reflect.Map, ValueEncoderFunc(dve.MapEncodeValue)).
		RegisterDefaultEncoder(reflect.Slice, ValueEncoderFunc(dve.SliceEncodeValue)).
		RegisterDefaultEncoder(reflect.String, ValueEncoderFunc(dve.StringEncodeValue)).
		RegisterDefaultEncoder(reflect.Struct, &StructCodec{cache: make(map[reflect.Type]*structDescription), parser: DefaultStructTagParser}).
		RegisterDefaultEncoder(reflect.Ptr, NewPointerCodec())
}

// BooleanEncodeValue is the ValueEncoderFunc for bool types.
func (dve DefaultValueEncoders) BooleanEncodeValue(ectx EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Kind() != reflect.Bool {
		return ValueEncoderError{Name: "BooleanEncodeValue", Kinds: []reflect.Kind{reflect.Bool}, Received: val}
	}
	return vw.WriteBoolean(val.Bool())
}

func fitsIn32Bits(i int64) bool {
	return math.MinInt32 <= i && i <= math.MaxInt32
}

// IntEncodeValue is the ValueEncoderFunc for int types.
func (dve DefaultValueEncoders) IntEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	switch val.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32:
		return vw.WriteInt32(int32(val.Int()))
	case reflect.Int:
		i64 := val.Int()
		if fitsIn32Bits(i64) {
			return vw.WriteInt32(int32(i64))
		}
		return vw.WriteInt64(i64)
	case reflect.Int64:
		i64 := val.Int()
		if ec.MinSize && fitsIn32Bits(i64) {
			return vw.WriteInt32(int32(i64))
		}
		return vw.WriteInt64(i64)
	}

	return ValueEncoderError{
		Name:     "IntEncodeValue",
		Kinds:    []reflect.Kind{reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Int},
		Received: val,
	}
}

// UintEncodeValue is the ValueEncoderFunc for uint types.
func (dve DefaultValueEncoders) UintEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	switch val.Kind() {
	case reflect.Uint8, reflect.Uint16:
		return vw.WriteInt32(int32(val.Uint()))
	case reflect.Uint, reflect.Uint32, reflect.Uint64:
		u64 := val.Uint()
		if ec.MinSize && u64 <= math.MaxInt32 {
			return vw.WriteInt32(int32(u64))
		}
		if u64 > math.MaxInt64 {
			return fmt.Errorf("%d overflows int64", u64)
		}
		return vw.WriteInt64(int64(u64))
	}

	return ValueEncoderError{
		Name:     "UintEncodeValue",
		Kinds:    []reflect.Kind{reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uint},
		Received: val,
	}
}

// FloatEncodeValue is the ValueEncoderFunc for float types.
func (dve DefaultValueEncoders) FloatEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	switch val.Kind() {
	case reflect.Float32, reflect.Float64:
		return vw.WriteDouble(val.Float())
	}

	return ValueEncoderError{Name: "FloatEncodeValue", Kinds: []reflect.Kind{reflect.Float32, reflect.Float64}, Received: val}
}

// StringEncodeValue is the ValueEncoderFunc for string types.
func (dve DefaultValueEncoders) StringEncodeValue(ectx EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if val.Kind() != reflect.String {
		return ValueEncoderError{
			Name:     "StringEncodeValue",
			Kinds:    []reflect.Kind{reflect.String},
			Received: val,
		}
	}

	return vw.WriteString(val.String())
}

// ObjectIDEncodeValue is the ValueEncoderFunc for primitive.ObjectID.
func (dve DefaultValueEncoders) ObjectIDEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tOID {
		return ValueEncoderError{Name: "ObjectIDEncodeValue", Types: []reflect.Type{tOID}, Received: val}
	}
	return vw.WriteObjectID(val.Interface().(primitive.ObjectID))
}

// Decimal128EncodeValue is the ValueEncoderFunc for primitive.Decimal128.
func (dve DefaultValueEncoders) Decimal128EncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tDecimal {
		return ValueEncoderError{Name: "Decimal128EncodeValue", Types: []reflect.Type{tDecimal}, Received: val}
	}
	return vw.WriteDecimal128(val.Interface().(primitive.Decimal128))
}

// JSONNumberEncodeValue is the ValueEncoderFunc for json.Number.
func (dve DefaultValueEncoders) JSONNumberEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tJSONNumber {
		return ValueEncoderError{Name: "JSONNumberEncodeValue", Types: []reflect.Type{tJSONNumber}, Received: val}
	}
	jsnum := val.Interface().(json.Number)

	// Attempt int first, then float64
	if i64, err := jsnum.Int64(); err == nil {
		return dve.IntEncodeValue(ec, vw, reflect.ValueOf(i64))
	}

	f64, err := jsnum.Float64()
	if err != nil {
		return err
	}

	return dve.FloatEncodeValue(ec, vw, reflect.ValueOf(f64))
}

// URLEncodeValue is the ValueEncoderFunc for url.URL.
func (dve DefaultValueEncoders) URLEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tURL {
		return ValueEncoderError{Name: "URLEncodeValue", Types: []reflect.Type{tURL}, Received: val}
	}
	u := val.Interface().(url.URL)
	return vw.WriteString(u.String())
}

// TimeEncodeValue is the ValueEncoderFunc for time.TIme.
func (dve DefaultValueEncoders) TimeEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tTime {
		return ValueEncoderError{Name: "TimeEncodeValue", Types: []reflect.Type{tTime}, Received: val}
	}
	tt := val.Interface().(time.Time)
	return vw.WriteDateTime(tt.Unix()*1000 + int64(tt.Nanosecond()/1e6))
}

// ByteSliceEncodeValue is the ValueEncoderFunc for []byte.
func (dve DefaultValueEncoders) ByteSliceEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tByteSlice {
		return ValueEncoderError{Name: "ByteSliceEncodeValue", Types: []reflect.Type{tByteSlice}, Received: val}
	}
	if val.IsNil() {
		return vw.WriteNull()
	}
	return vw.WriteBinary(val.Interface().([]byte))
}

// MapEncodeValue is the ValueEncoderFunc for map[string]* types.
func (dve DefaultValueEncoders) MapEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Kind() != reflect.Map || val.Type().Key().Kind() != reflect.String {
		return ValueEncoderError{Name: "MapEncodeValue", Kinds: []reflect.Kind{reflect.Map}, Received: val}
	}

	if val.IsNil() {
		// If we have a nill map but we can't WriteNull, that means we're probably trying to encode
		// to a TopLevel document. We can't currently tell if this is what actually happened, but if
		// there's a deeper underlying problem, the error will also be returned from WriteDocument,
		// so just continue. The operations on a map reflection value are valid, so we can call
		// MapKeys within mapEncodeValue without a problem.
		err := vw.WriteNull()
		if err == nil {
			return nil
		}
	}

	dw, err := vw.WriteDocument()
	if err != nil {
		return err
	}

	return dve.mapEncodeValue(ec, dw, val, nil)
}

// mapEncodeValue handles encoding of the values of a map. The collisionFn returns
// true if the provided key exists, this is mainly used for inline maps in the
// struct codec.
func (dve DefaultValueEncoders) mapEncodeValue(ec EncodeContext, dw bsonrw.DocumentWriter, val reflect.Value, collisionFn func(string) bool) error {

	encoder, err := ec.LookupEncoder(val.Type().Elem())
	if err != nil {
		return err
	}

	keys := val.MapKeys()
	for _, key := range keys {
		if collisionFn != nil && collisionFn(key.String()) {
			return fmt.Errorf("Key %s of inlined map conflicts with a struct field name", key)
		}
		vw, err := dw.WriteDocumentElement(key.String())
		if err != nil {
			return err
		}

		if enc, ok := encoder.(ValueEncoder); ok {
			err = enc.EncodeValue(ec, vw, val.MapIndex(key))
			if err != nil {
				return err
			}
			continue
		}
		err = encoder.EncodeValue(ec, vw, val.MapIndex(key))
		if err != nil {
			return err
		}
	}

	return dw.WriteDocumentEnd()
}

// ArrayEncodeValue is the ValueEncoderFunc for array types.
func (dve DefaultValueEncoders) ArrayEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Kind() != reflect.Array {
		return ValueEncoderError{Name: "ArrayEncodeValue", Kinds: []reflect.Kind{reflect.Array}, Received: val}
	}

	// If we have a []primitive.E we want to treat it as a document instead of as an array.
	if val.Type().Elem() == tE {
		dw, err := vw.WriteDocument()
		if err != nil {
			return err
		}

		for idx := 0; idx < val.Len(); idx++ {
			e := val.Index(idx).Interface().(primitive.E)
			err = encodeElement(ec, dw, e)
			if err != nil {
				return err
			}
		}

		return dw.WriteDocumentEnd()
	}

	aw, err := vw.WriteArray()
	if err != nil {
		return err
	}

	encoder, err := ec.LookupEncoder(val.Type().Elem())
	if err != nil {
		return err
	}

	for idx := 0; idx < val.Len(); idx++ {
		vw, err := aw.WriteArrayElement()
		if err != nil {
			return err
		}

		err = encoder.EncodeValue(ec, vw, val.Index(idx))
		if err != nil {
			return err
		}
	}
	return aw.WriteArrayEnd()
}

// SliceEncodeValue is the ValueEncoderFunc for slice types.
func (dve DefaultValueEncoders) SliceEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Kind() != reflect.Slice {
		return ValueEncoderError{Name: "SliceEncodeValue", Kinds: []reflect.Kind{reflect.Slice}, Received: val}
	}

	if val.IsNil() {
		return vw.WriteNull()
	}

	// If we have a []primitive.E we want to treat it as a document instead of as an array.
	if val.Type().ConvertibleTo(tD) {
		d := val.Convert(tD).Interface().(primitive.D)

		dw, err := vw.WriteDocument()
		if err != nil {
			return err
		}

		for _, e := range d {
			err = encodeElement(ec, dw, e)
			if err != nil {
				return err
			}
		}

		return dw.WriteDocumentEnd()
	}

	aw, err := vw.WriteArray()
	if err != nil {
		return err
	}

	encoder, err := ec.LookupEncoder(val.Type().Elem())
	if err != nil {
		return err
	}

	for idx := 0; idx < val.Len(); idx++ {
		vw, err := aw.WriteArrayElement()
		if err != nil {
			return err
		}

		err = encoder.EncodeValue(ec, vw, val.Index(idx))
		if err != nil {
			return err
		}
	}
	return aw.WriteArrayEnd()
}

// EmptyInterfaceEncodeValue is the ValueEncoderFunc for interface{}.
func (dve DefaultValueEncoders) EmptyInterfaceEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tEmpty {
		return ValueEncoderError{Name: "EmptyInterfaceEncodeValue", Types: []reflect.Type{tEmpty}, Received: val}
	}

	if val.IsNil() {
		return vw.WriteNull()
	}
	encoder, err := ec.LookupEncoder(val.Elem().Type())
	if err != nil {
		return err
	}

	return encoder.EncodeValue(ec, vw, val.Elem())
}

// ValueMarshalerEncodeValue is the ValueEncoderFunc for ValueMarshaler implementations.
func (dve DefaultValueEncoders) ValueMarshalerEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || !val.Type().Implements(tValueMarshaler) {
		return ValueEncoderError{Name: "ValueMarshalerEncodeValue", Types: []reflect.Type{tValueMarshaler}, Received: val}
	}

	fn := val.Convert(tValueMarshaler).MethodByName("MarshalBSONValue")
	returns := fn.Call(nil)
	if !returns[2].IsNil() {
		return returns[2].Interface().(error)
	}
	t, data := returns[0].Interface().(bsontype.Type), returns[1].Interface().([]byte)
	return bsonrw.Copier{}.CopyValueFromBytes(vw, t, data)
}

// MarshalerEncodeValue is the ValueEncoderFunc for Marshaler implementations.
func (dve DefaultValueEncoders) MarshalerEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || !val.Type().Implements(tMarshaler) {
		return ValueEncoderError{Name: "MarshalerEncodeValue", Types: []reflect.Type{tMarshaler}, Received: val}
	}

	fn := val.Convert(tMarshaler).MethodByName("MarshalBSON")
	returns := fn.Call(nil)
	if !returns[1].IsNil() {
		return returns[1].Interface().(error)
	}
	data := returns[0].Interface().([]byte)
	return bsonrw.Copier{}.CopyValueFromBytes(vw, bsontype.EmbeddedDocument, data)
}

// ProxyEncodeValue is the ValueEncoderFunc for Proxy implementations.
func (dve DefaultValueEncoders) ProxyEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || !val.Type().Implements(tProxy) {
		return ValueEncoderError{Name: "ProxyEncodeValue", Types: []reflect.Type{tProxy}, Received: val}
	}

	fn := val.Convert(tProxy).MethodByName("ProxyBSON")
	returns := fn.Call(nil)
	if !returns[1].IsNil() {
		return returns[1].Interface().(error)
	}
	data := returns[0]
	var encoder ValueEncoder
	var err error
	if data.Elem().IsValid() {
		encoder, err = ec.LookupEncoder(data.Elem().Type())
	} else {
		encoder, err = ec.LookupEncoder(nil)
	}
	if err != nil {
		return err
	}
	return encoder.EncodeValue(ec, vw, data.Elem())
}

// JavaScriptEncodeValue is the ValueEncoderFunc for the primitive.JavaScript type.
func (DefaultValueEncoders) JavaScriptEncodeValue(ectx EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tJavaScript {
		return ValueEncoderError{Name: "JavaScriptEncodeValue", Types: []reflect.Type{tJavaScript}, Received: val}
	}

	return vw.WriteJavascript(val.String())
}

// SymbolEncodeValue is the ValueEncoderFunc for the primitive.Symbol type.
func (DefaultValueEncoders) SymbolEncodeValue(ectx EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tSymbol {
		return ValueEncoderError{Name: "SymbolEncodeValue", Types: []reflect.Type{tSymbol}, Received: val}
	}

	return vw.WriteSymbol(val.String())
}

// BinaryEncodeValue is the ValueEncoderFunc for Binary.
func (DefaultValueEncoders) BinaryEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tBinary {
		return ValueEncoderError{Name: "BinaryEncodeValue", Types: []reflect.Type{tBinary}, Received: val}
	}
	b := val.Interface().(primitive.Binary)

	return vw.WriteBinaryWithSubtype(b.Data, b.Subtype)
}

// UndefinedEncodeValue is the ValueEncoderFunc for Undefined.
func (DefaultValueEncoders) UndefinedEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tUndefined {
		return ValueEncoderError{Name: "UndefinedEncodeValue", Types: []reflect.Type{tUndefined}, Received: val}
	}

	return vw.WriteUndefined()
}

// DateTimeEncodeValue is the ValueEncoderFunc for DateTime.
func (DefaultValueEncoders) DateTimeEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tDateTime {
		return ValueEncoderError{Name: "DateTimeEncodeValue", Types: []reflect.Type{tDateTime}, Received: val}
	}

	return vw.WriteDateTime(val.Int())
}

// NullEncodeValue is the ValueEncoderFunc for Null.
func (DefaultValueEncoders) NullEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tNull {
		return ValueEncoderError{Name: "NullEncodeValue", Types: []reflect.Type{tNull}, Received: val}
	}

	return vw.WriteNull()
}

// RegexEncodeValue is the ValueEncoderFunc for Regex.
func (DefaultValueEncoders) RegexEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tRegex {
		return ValueEncoderError{Name: "RegexEncodeValue", Types: []reflect.Type{tRegex}, Received: val}
	}

	regex := val.Interface().(primitive.Regex)

	return vw.WriteRegex(regex.Pattern, regex.Options)
}

// DBPointerEncodeValue is the ValueEncoderFunc for DBPointer.
func (DefaultValueEncoders) DBPointerEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tDBPointer {
		return ValueEncoderError{Name: "DBPointerEncodeValue", Types: []reflect.Type{tDBPointer}, Received: val}
	}

	dbp := val.Interface().(primitive.DBPointer)

	return vw.WriteDBPointer(dbp.DB, dbp.Pointer)
}

// TimestampEncodeValue is the ValueEncoderFunc for Timestamp.
func (DefaultValueEncoders) TimestampEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tTimestamp {
		return ValueEncoderError{Name: "TimestampEncodeValue", Types: []reflect.Type{tTimestamp}, Received: val}
	}

	ts := val.Interface().(primitive.Timestamp)

	return vw.WriteTimestamp(ts.T, ts.I)
}

// MinKeyEncodeValue is the ValueEncoderFunc for MinKey.
func (DefaultValueEncoders) MinKeyEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tMinKey {
		return ValueEncoderError{Name: "MinKeyEncodeValue", Types: []reflect.Type{tMinKey}, Received: val}
	}

	return vw.WriteMinKey()
}

// MaxKeyEncodeValue is the ValueEncoderFunc for MaxKey.
func (DefaultValueEncoders) MaxKeyEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tMaxKey {
		return ValueEncoderError{Name: "MaxKeyEncodeValue", Types: []reflect.Type{tMaxKey}, Received: val}
	}

	return vw.WriteMaxKey()
}

// CoreDocumentEncodeValue is the ValueEncoderFunc for bsoncore.Document.
func (DefaultValueEncoders) CoreDocumentEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tCoreDocument {
		return ValueEncoderError{Name: "CoreDocumentEncodeValue", Types: []reflect.Type{tCoreDocument}, Received: val}
	}

	cdoc := val.Interface().(bsoncore.Document)

	return bsonrw.Copier{}.CopyDocumentFromBytes(vw, cdoc)
}

// CodeWithScopeEncodeValue is the ValueEncoderFunc for CodeWithScope.
func (dve DefaultValueEncoders) CodeWithScopeEncodeValue(ec EncodeContext, vw bsonrw.ValueWriter, val reflect.Value) error {
	if !val.IsValid() || val.Type() != tCodeWithScope {
		return ValueEncoderError{Name: "CodeWithScopeEncodeValue", Types: []reflect.Type{tCodeWithScope}, Received: val}
	}

	cws := val.Interface().(primitive.CodeWithScope)

	dw, err := vw.WriteCodeWithScope(string(cws.Code))
	if err != nil {
		return err
	}

	sw := sliceWriterPool.Get().(*bsonrw.SliceWriter)
	defer sliceWriterPool.Put(sw)
	*sw = (*sw)[:0]

	scopeVW := bvwPool.Get(sw)
	defer bvwPool.Put(scopeVW)

	encoder, err := ec.LookupEncoder(reflect.TypeOf(cws.Scope))
	if err != nil {
		return err
	}

	err = encoder.EncodeValue(ec, scopeVW, reflect.ValueOf(cws.Scope))
	if err != nil {
		return err
	}

	err = bsonrw.Copier{}.CopyBytesToDocumentWriter(dw, *sw)
	if err != nil {
		return err
	}
	return dw.WriteDocumentEnd()
}
