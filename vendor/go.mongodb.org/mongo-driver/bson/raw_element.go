// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bson

import (
	"go.mongodb.org/mongo-driver/x/bsonx/bsoncore"
)

// RawElement represents a BSON element in byte form. This type provides a simple way to
// transform a slice of bytes into a BSON element and extract information from it.
//
// RawElement is a thin wrapper around a bsoncore.Element.
type RawElement []byte

// Key returns the key for this element. If the element is not valid, this method returns an empty
// string. If knowing if the element is valid is important, use KeyErr.
func (re RawElement) Key() string { return bsoncore.Element(re).Key() }

// KeyErr returns the key for this element, returning an error if the element is not valid.
func (re RawElement) KeyErr() (string, error) { return bsoncore.Element(re).KeyErr() }

// Value returns the value of this element. If the element is not valid, this method returns an
// empty Value. If knowing if the element is valid is important, use ValueErr.
func (re RawElement) Value() RawValue { return convertFromCoreValue(bsoncore.Element(re).Value()) }

// ValueErr returns the value for this element, returning an error if the element is not valid.
func (re RawElement) ValueErr() (RawValue, error) {
	val, err := bsoncore.Element(re).ValueErr()
	return convertFromCoreValue(val), err
}

// Validate ensures re is a valid BSON element.
func (re RawElement) Validate() error { return bsoncore.Element(re).Validate() }

// String implements the fmt.Stringer interface. The output will be in extended JSON format.
func (re RawElement) String() string {
	doc := bsoncore.BuildDocument(nil, re)
	j, err := MarshalExtJSON(Raw(doc), true, false)
	if err != nil {
		return "<malformed>"
	}
	return string(j)
}

// DebugString outputs a human readable version of RawElement. It will attempt to stringify the
// valid components of the element even if the entire element is not valid.
func (re RawElement) DebugString() string { return bsoncore.Element(re).DebugString() }
