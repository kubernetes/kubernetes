// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

// Package bsontype is a utility package that contains types for each BSON type and the
// a stringifier for the Type to enable easier debugging when working with BSON.
package bsontype // import "go.mongodb.org/mongo-driver/bson/bsontype"

// These constants uniquely refer to each BSON type.
const (
	Double           Type = 0x01
	String           Type = 0x02
	EmbeddedDocument Type = 0x03
	Array            Type = 0x04
	Binary           Type = 0x05
	Undefined        Type = 0x06
	ObjectID         Type = 0x07
	Boolean          Type = 0x08
	DateTime         Type = 0x09
	Null             Type = 0x0A
	Regex            Type = 0x0B
	DBPointer        Type = 0x0C
	JavaScript       Type = 0x0D
	Symbol           Type = 0x0E
	CodeWithScope    Type = 0x0F
	Int32            Type = 0x10
	Timestamp        Type = 0x11
	Int64            Type = 0x12
	Decimal128       Type = 0x13
	MinKey           Type = 0xFF
	MaxKey           Type = 0x7F
)

// Type represents a BSON type.
type Type byte

// String returns the string representation of the BSON type's name.
func (bt Type) String() string {
	switch bt {
	case '\x01':
		return "double"
	case '\x02':
		return "string"
	case '\x03':
		return "embedded document"
	case '\x04':
		return "array"
	case '\x05':
		return "binary"
	case '\x06':
		return "undefined"
	case '\x07':
		return "objectID"
	case '\x08':
		return "boolean"
	case '\x09':
		return "UTC datetime"
	case '\x0A':
		return "null"
	case '\x0B':
		return "regex"
	case '\x0C':
		return "dbPointer"
	case '\x0D':
		return "javascript"
	case '\x0E':
		return "symbol"
	case '\x0F':
		return "code with scope"
	case '\x10':
		return "32-bit integer"
	case '\x11':
		return "timestamp"
	case '\x12':
		return "64-bit integer"
	case '\x13':
		return "128-bit decimal"
	case '\xFF':
		return "min key"
	case '\x7F':
		return "max key"
	default:
		return "invalid"
	}
}
