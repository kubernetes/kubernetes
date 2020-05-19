// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bson

import (
	"reflect"
	"time"

	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

// These constants uniquely refer to each BSON type.
const (
	TypeDouble           = bsontype.Double
	TypeString           = bsontype.String
	TypeEmbeddedDocument = bsontype.EmbeddedDocument
	TypeArray            = bsontype.Array
	TypeBinary           = bsontype.Binary
	TypeUndefined        = bsontype.Undefined
	TypeObjectID         = bsontype.ObjectID
	TypeBoolean          = bsontype.Boolean
	TypeDateTime         = bsontype.DateTime
	TypeNull             = bsontype.Null
	TypeRegex            = bsontype.Regex
	TypeDBPointer        = bsontype.DBPointer
	TypeJavaScript       = bsontype.JavaScript
	TypeSymbol           = bsontype.Symbol
	TypeCodeWithScope    = bsontype.CodeWithScope
	TypeInt32            = bsontype.Int32
	TypeTimestamp        = bsontype.Timestamp
	TypeInt64            = bsontype.Int64
	TypeDecimal128       = bsontype.Decimal128
	TypeMinKey           = bsontype.MinKey
	TypeMaxKey           = bsontype.MaxKey
)

var tBinary = reflect.TypeOf(primitive.Binary{})
var tBool = reflect.TypeOf(false)
var tCodeWithScope = reflect.TypeOf(primitive.CodeWithScope{})
var tDBPointer = reflect.TypeOf(primitive.DBPointer{})
var tDecimal = reflect.TypeOf(primitive.Decimal128{})
var tD = reflect.TypeOf(D{})
var tA = reflect.TypeOf(A{})
var tDateTime = reflect.TypeOf(primitive.DateTime(0))
var tUndefined = reflect.TypeOf(primitive.Undefined{})
var tNull = reflect.TypeOf(primitive.Null{})
var tRawValue = reflect.TypeOf(RawValue{})
var tFloat32 = reflect.TypeOf(float32(0))
var tFloat64 = reflect.TypeOf(float64(0))
var tInt = reflect.TypeOf(int(0))
var tInt8 = reflect.TypeOf(int8(0))
var tInt16 = reflect.TypeOf(int16(0))
var tInt32 = reflect.TypeOf(int32(0))
var tInt64 = reflect.TypeOf(int64(0))
var tJavaScript = reflect.TypeOf(primitive.JavaScript(""))
var tOID = reflect.TypeOf(primitive.ObjectID{})
var tRaw = reflect.TypeOf(Raw(nil))
var tRegex = reflect.TypeOf(primitive.Regex{})
var tString = reflect.TypeOf("")
var tSymbol = reflect.TypeOf(primitive.Symbol(""))
var tTime = reflect.TypeOf(time.Time{})
var tTimestamp = reflect.TypeOf(primitive.Timestamp{})
var tUint = reflect.TypeOf(uint(0))
var tUint8 = reflect.TypeOf(uint8(0))
var tUint16 = reflect.TypeOf(uint16(0))
var tUint32 = reflect.TypeOf(uint32(0))
var tUint64 = reflect.TypeOf(uint64(0))
var tMinKey = reflect.TypeOf(primitive.MinKey{})
var tMaxKey = reflect.TypeOf(primitive.MaxKey{})

var tEmpty = reflect.TypeOf((*interface{})(nil)).Elem()
var tEmptySlice = reflect.TypeOf([]interface{}(nil))

var zeroVal reflect.Value

// this references the quantity of milliseconds between zero time and
// the unix epoch. useful for making sure that we convert time.Time
// objects correctly to match the legacy bson library's handling of
// time.Time values.
const zeroEpochMs = int64(62135596800000)
