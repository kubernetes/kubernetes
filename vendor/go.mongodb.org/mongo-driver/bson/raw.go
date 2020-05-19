// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bson

import (
	"errors"
	"io"

	"go.mongodb.org/mongo-driver/x/bsonx/bsoncore"
)

// ErrNilReader indicates that an operation was attempted on a nil bson.Reader.
var ErrNilReader = errors.New("nil reader")
var errValidateDone = errors.New("validation loop complete")

// Raw is a wrapper around a byte slice. It will interpret the slice as a
// BSON document. This type is a wrapper around a bsoncore.Document. Errors returned from the
// methods on this type and associated types come from the bsoncore package.
type Raw []byte

// NewFromIOReader reads in a document from the given io.Reader and constructs a Raw from
// it.
func NewFromIOReader(r io.Reader) (Raw, error) {
	doc, err := bsoncore.NewDocumentFromReader(r)
	return Raw(doc), err
}

// Validate validates the document. This method only validates the first document in
// the slice, to validate other documents, the slice must be resliced.
func (r Raw) Validate() (err error) { return bsoncore.Document(r).Validate() }

// Lookup search the document, potentially recursively, for the given key. If
// there are multiple keys provided, this method will recurse down, as long as
// the top and intermediate nodes are either documents or arrays.If an error
// occurs or if the value doesn't exist, an empty RawValue is returned.
func (r Raw) Lookup(key ...string) RawValue {
	return convertFromCoreValue(bsoncore.Document(r).Lookup(key...))
}

// LookupErr searches the document and potentially subdocuments or arrays for the
// provided key. Each key provided to this method represents a layer of depth.
func (r Raw) LookupErr(key ...string) (RawValue, error) {
	val, err := bsoncore.Document(r).LookupErr(key...)
	return convertFromCoreValue(val), err
}

// Elements returns this document as a slice of elements. The returned slice will contain valid
// elements. If the document is not valid, the elements up to the invalid point will be returned
// along with an error.
func (r Raw) Elements() ([]RawElement, error) {
	elems, err := bsoncore.Document(r).Elements()
	relems := make([]RawElement, 0, len(elems))
	for _, elem := range elems {
		relems = append(relems, RawElement(elem))
	}
	return relems, err
}

// Values returns this document as a slice of values. The returned slice will contain valid values.
// If the document is not valid, the values up to the invalid point will be returned along with an
// error.
func (r Raw) Values() ([]RawValue, error) {
	vals, err := bsoncore.Document(r).Values()
	rvals := make([]RawValue, 0, len(vals))
	for _, val := range vals {
		rvals = append(rvals, convertFromCoreValue(val))
	}
	return rvals, err
}

// Index searches for and retrieves the element at the given index. This method will panic if
// the document is invalid or if the index is out of bounds.
func (r Raw) Index(index uint) RawElement { return RawElement(bsoncore.Document(r).Index(index)) }

// IndexErr searches for and retrieves the element at the given index.
func (r Raw) IndexErr(index uint) (RawElement, error) {
	elem, err := bsoncore.Document(r).IndexErr(index)
	return RawElement(elem), err
}

// String implements the fmt.Stringer interface.
func (r Raw) String() string { return bsoncore.Document(r).String() }

// readi32 is a helper function for reading an int32 from slice of bytes.
func readi32(b []byte) int32 {
	_ = b[3] // bounds check hint to compiler; see golang.org/issue/14808
	return int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16 | int32(b[3])<<24
}
