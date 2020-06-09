// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Based on gopkg.in/mgo.v2/bson by Gustavo Niemeyer
// See THIRD-PARTY-NOTICES for original license terms.

// +build go1.9

package bson // import "go.mongodb.org/mongo-driver/bson"

import (
	"go.mongodb.org/mongo-driver/bson/primitive"
)

// Zeroer allows custom struct types to implement a report of zero
// state. All struct types that don't implement Zeroer or where IsZero
// returns false are considered to be not zero.
type Zeroer interface {
	IsZero() bool
}

// D represents a BSON Document. This type can be used to represent BSON in a concise and readable
// manner. It should generally be used when serializing to BSON. For deserializing, the Raw or
// Document types should be used.
//
// Example usage:
//
// 		bson.D{{"foo", "bar"}, {"hello", "world"}, {"pi", 3.14159}}
//
// This type should be used in situations where order matters, such as MongoDB commands. If the
// order is not important, a map is more comfortable and concise.
type D = primitive.D

// E represents a BSON element for a D. It is usually used inside a D.
type E = primitive.E

// M is an unordered, concise representation of a BSON Document. It should generally be used to
// serialize BSON when the order of the elements of a BSON document do not matter. If the element
// order matters, use a D instead.
//
// Example usage:
//
// 		bson.M{"foo": "bar", "hello": "world", "pi": 3.14159}
//
// This type is handled in the encoders as a regular map[string]interface{}. The elements will be
// serialized in an undefined, random order, and the order will be different each time.
type M = primitive.M

// An A represents a BSON array. This type can be used to represent a BSON array in a concise and
// readable manner. It should generally be used when serializing to BSON. For deserializing, the
// RawArray or Array types should be used.
//
// Example usage:
//
// 		bson.A{"bar", "world", 3.14159, bson.D{{"qux", 12345}}}
//
type A = primitive.A
