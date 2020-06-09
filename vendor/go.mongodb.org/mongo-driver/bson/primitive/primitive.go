// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

// Package primitive contains types similar to Go primitives for BSON types can do not have direct
// Go primitive representations.
package primitive // import "go.mongodb.org/mongo-driver/bson/primitive"

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"
)

// Binary represents a BSON binary value.
type Binary struct {
	Subtype byte
	Data    []byte
}

// Equal compaes bp to bp2 and returns true is the are equal.
func (bp Binary) Equal(bp2 Binary) bool {
	if bp.Subtype != bp2.Subtype {
		return false
	}
	return bytes.Equal(bp.Data, bp2.Data)
}

// Undefined represents the BSON undefined value type.
type Undefined struct{}

// DateTime represents the BSON datetime value.
type DateTime int64

// MarshalJSON marshal to time type
func (d DateTime) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.Time())
}

// Time returns the date as a time type.
func (d DateTime) Time() time.Time {
	return time.Unix(int64(d)/1000, int64(d)%1000*1000000)
}

// NewDateTimeFromTime creates a new DateTime from a Time.
func NewDateTimeFromTime(t time.Time) DateTime {
	return DateTime(t.UnixNano() / 1000000)
}

// Null repreesnts the BSON null value.
type Null struct{}

// Regex represents a BSON regex value.
type Regex struct {
	Pattern string
	Options string
}

func (rp Regex) String() string {
	return fmt.Sprintf(`{"pattern": "%s", "options": "%s"}`, rp.Pattern, rp.Options)
}

// Equal compaes rp to rp2 and returns true is the are equal.
func (rp Regex) Equal(rp2 Regex) bool {
	return rp.Pattern == rp2.Pattern && rp.Options == rp.Options
}

// DBPointer represents a BSON dbpointer value.
type DBPointer struct {
	DB      string
	Pointer ObjectID
}

func (d DBPointer) String() string {
	return fmt.Sprintf(`{"db": "%s", "pointer": "%s"}`, d.DB, d.Pointer)
}

// Equal compaes d to d2 and returns true is the are equal.
func (d DBPointer) Equal(d2 DBPointer) bool {
	return d.DB == d2.DB && bytes.Equal(d.Pointer[:], d2.Pointer[:])
}

// JavaScript represents a BSON JavaScript code value.
type JavaScript string

// Symbol represents a BSON symbol value.
type Symbol string

// CodeWithScope represents a BSON JavaScript code with scope value.
type CodeWithScope struct {
	Code  JavaScript
	Scope interface{}
}

func (cws CodeWithScope) String() string {
	return fmt.Sprintf(`{"code": "%s", "scope": %v}`, cws.Code, cws.Scope)
}

// Timestamp represents a BSON timestamp value.
type Timestamp struct {
	T uint32
	I uint32
}

// Equal compaes tp to tp2 and returns true is the are equal.
func (tp Timestamp) Equal(tp2 Timestamp) bool {
	return tp.T == tp2.T && tp.I == tp2.I
}

// CompareTimestamp returns an integer comparing two Timestamps, where T is compared first, followed by I.
// Returns 0 if tp = tp2, 1 if tp > tp2, -1 if tp < tp2.
func CompareTimestamp(tp, tp2 Timestamp) int {
	if tp.Equal(tp2) {
		return 0
	}

	if tp.T > tp2.T {
		return 1
	}
	if tp.T < tp2.T {
		return -1
	}
	// Compare I values because T values are equal
	if tp.I > tp2.I {
		return 1
	}
	return -1
}

// MinKey represents the BSON minkey value.
type MinKey struct{}

// MaxKey represents the BSON maxkey value.
type MaxKey struct{}

// D represents a BSON Document. This type can be used to represent BSON in a concise and readable
// manner. It should generally be used when serializing to BSON. For deserializing, the Raw or
// Document types should be used.
//
// Example usage:
//
// 		primitive.D{{"foo", "bar"}, {"hello", "world"}, {"pi", 3.14159}}
//
// This type should be used in situations where order matters, such as MongoDB commands. If the
// order is not important, a map is more comfortable and concise.
type D []E

// Map creates a map from the elements of the D.
func (d D) Map() M {
	m := make(M, len(d))
	for _, e := range d {
		m[e.Key] = e.Value
	}
	return m
}

// E represents a BSON element for a D. It is usually used inside a D.
type E struct {
	Key   string
	Value interface{}
}

// M is an unordered, concise representation of a BSON Document. It should generally be used to
// serialize BSON when the order of the elements of a BSON document do not matter. If the element
// order matters, use a D instead.
//
// Example usage:
//
// 		primitive.M{"foo": "bar", "hello": "world", "pi": 3.14159}
//
// This type is handled in the encoders as a regular map[string]interface{}. The elements will be
// serialized in an undefined, random order, and the order will be different each time.
type M map[string]interface{}

// An A represents a BSON array. This type can be used to represent a BSON array in a concise and
// readable manner. It should generally be used when serializing to BSON. For deserializing, the
// RawArray or Array types should be used.
//
// Example usage:
//
// 		primitive.A{"bar", "world", 3.14159, primitive.D{{"qux", 12345}}}
//
type A []interface{}
