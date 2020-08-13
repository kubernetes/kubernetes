// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsonrw

import (
	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

// ArrayReader is implemented by types that allow reading values from a BSON
// array.
type ArrayReader interface {
	ReadValue() (ValueReader, error)
}

// DocumentReader is implemented by types that allow reading elements from a
// BSON document.
type DocumentReader interface {
	ReadElement() (string, ValueReader, error)
}

// ValueReader is a generic interface used to read values from BSON. This type
// is implemented by several types with different underlying representations of
// BSON, such as a bson.Document, raw BSON bytes, or extended JSON.
type ValueReader interface {
	Type() bsontype.Type
	Skip() error

	ReadArray() (ArrayReader, error)
	ReadBinary() (b []byte, btype byte, err error)
	ReadBoolean() (bool, error)
	ReadDocument() (DocumentReader, error)
	ReadCodeWithScope() (code string, dr DocumentReader, err error)
	ReadDBPointer() (ns string, oid primitive.ObjectID, err error)
	ReadDateTime() (int64, error)
	ReadDecimal128() (primitive.Decimal128, error)
	ReadDouble() (float64, error)
	ReadInt32() (int32, error)
	ReadInt64() (int64, error)
	ReadJavascript() (code string, err error)
	ReadMaxKey() error
	ReadMinKey() error
	ReadNull() error
	ReadObjectID() (primitive.ObjectID, error)
	ReadRegex() (pattern, options string, err error)
	ReadString() (string, error)
	ReadSymbol() (symbol string, err error)
	ReadTimestamp() (t, i uint32, err error)
	ReadUndefined() error
}

// BytesReader is a generic interface used to read BSON bytes from a
// ValueReader. This imterface is meant to be a superset of ValueReader, so that
// types that implement ValueReader may also implement this interface.
//
// The bytes of the value will be appended to dst.
type BytesReader interface {
	ReadValueBytes(dst []byte) (bsontype.Type, []byte, error)
}
