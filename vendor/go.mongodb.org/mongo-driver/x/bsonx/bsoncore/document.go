// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsoncore

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strconv"

	"github.com/go-stack/stack"
	"go.mongodb.org/mongo-driver/bson/bsontype"
)

// DocumentValidationError is an error type returned when attempting to validate a document.
type DocumentValidationError string

func (dve DocumentValidationError) Error() string { return string(dve) }

// NewDocumentLengthError creates and returns an error for when the length of a document exceeds the
// bytes available.
func NewDocumentLengthError(length, rem int) error {
	return DocumentValidationError(
		fmt.Sprintf("document length exceeds available bytes. length=%d remainingBytes=%d", length, rem),
	)
}

// InsufficientBytesError indicates that there were not enough bytes to read the next component.
type InsufficientBytesError struct {
	Source    []byte
	Remaining []byte
	Stack     stack.CallStack
}

// NewInsufficientBytesError creates a new InsufficientBytesError with the given Document, remaining
// bytes, and the current stack.
func NewInsufficientBytesError(src, rem []byte) InsufficientBytesError {
	return InsufficientBytesError{Source: src, Remaining: rem, Stack: stack.Trace().TrimRuntime()}
}

// Error implements the error interface.
func (ibe InsufficientBytesError) Error() string {
	return "too few bytes to read next component"
}

// ErrorStack returns a string representing the stack at the point where the error occurred.
func (ibe InsufficientBytesError) ErrorStack() string {
	s := bytes.NewBufferString("too few bytes to read next component: [")

	for i, call := range ibe.Stack {
		if i != 0 {
			s.WriteString(", ")
		}

		// go vet doesn't like %k even though it's part of stack's API, so we move the format
		// string so it doesn't complain. (We also can't make it a constant, or go vet still
		// complains.)
		callFormat := "%k.%n %v"

		s.WriteString(fmt.Sprintf(callFormat, call, call, call))
	}

	s.WriteRune(']')

	return s.String()
}

// Equal checks that err2 also is an ErrTooSmall.
func (ibe InsufficientBytesError) Equal(err2 error) bool {
	switch err2.(type) {
	case InsufficientBytesError:
		return true
	default:
		return false
	}
}

// InvalidDepthTraversalError is returned when attempting a recursive Lookup when one component of
// the path is neither an embedded document nor an array.
type InvalidDepthTraversalError struct {
	Key  string
	Type bsontype.Type
}

func (idte InvalidDepthTraversalError) Error() string {
	return fmt.Sprintf(
		"attempt to traverse into %s, but it's type is %s, not %s nor %s",
		idte.Key, idte.Type, bsontype.EmbeddedDocument, bsontype.Array,
	)
}

// ErrMissingNull is returned when a document's last byte is not null.
const ErrMissingNull DocumentValidationError = "document end is missing null byte"

// ErrNilReader indicates that an operation was attempted on a nil io.Reader.
var ErrNilReader = errors.New("nil reader")

// ErrInvalidLength indicates that a length in a binary representation of a BSON document is invalid.
var ErrInvalidLength = errors.New("document length is invalid")

// ErrEmptyKey indicates that no key was provided to a Lookup method.
var ErrEmptyKey = errors.New("empty key provided")

// ErrElementNotFound indicates that an Element matching a certain condition does not exist.
var ErrElementNotFound = errors.New("element not found")

// ErrOutOfBounds indicates that an index provided to access something was invalid.
var ErrOutOfBounds = errors.New("out of bounds")

// Document is a raw bytes representation of a BSON document.
type Document []byte

// Array is a raw bytes representation of a BSON array.
type Array = Document

// NewDocumentFromReader reads a document from r. This function will only validate the length is
// correct and that the document ends with a null byte.
func NewDocumentFromReader(r io.Reader) (Document, error) {
	if r == nil {
		return nil, ErrNilReader
	}

	var lengthBytes [4]byte

	// ReadFull guarantees that we will have read at least len(lengthBytes) if err == nil
	_, err := io.ReadFull(r, lengthBytes[:])
	if err != nil {
		return nil, err
	}

	length, _, _ := readi32(lengthBytes[:]) // ignore ok since we always have enough bytes to read a length
	if length < 0 {
		return nil, ErrInvalidLength
	}
	document := make([]byte, length)

	copy(document, lengthBytes[:])

	_, err = io.ReadFull(r, document[4:])
	if err != nil {
		return nil, err
	}

	if document[length-1] != 0x00 {
		return nil, ErrMissingNull
	}

	return document, nil
}

// Lookup searches the document, potentially recursively, for the given key. If there are multiple
// keys provided, this method will recurse down, as long as the top and intermediate nodes are
// either documents or arrays. If an error occurs or if the value doesn't exist, an empty Value is
// returned.
func (d Document) Lookup(key ...string) Value {
	val, _ := d.LookupErr(key...)
	return val
}

// LookupErr is the same as Lookup, except it returns an error in addition to an empty Value.
func (d Document) LookupErr(key ...string) (Value, error) {
	if len(key) < 1 {
		return Value{}, ErrEmptyKey
	}
	length, rem, ok := ReadLength(d)
	if !ok {
		return Value{}, NewInsufficientBytesError(d, rem)
	}

	length -= 4

	var elem Element
	for length > 1 {
		elem, rem, ok = ReadElement(rem)
		length -= int32(len(elem))
		if !ok {
			return Value{}, NewInsufficientBytesError(d, rem)
		}
		if elem.Key() != key[0] {
			continue
		}
		if len(key) > 1 {
			tt := bsontype.Type(elem[0])
			switch tt {
			case bsontype.EmbeddedDocument:
				val, err := elem.Value().Document().LookupErr(key[1:]...)
				if err != nil {
					return Value{}, err
				}
				return val, nil
			case bsontype.Array:
				val, err := elem.Value().Array().LookupErr(key[1:]...)
				if err != nil {
					return Value{}, err
				}
				return val, nil
			default:
				return Value{}, InvalidDepthTraversalError{Key: elem.Key(), Type: tt}
			}
		}
		return elem.ValueErr()
	}
	return Value{}, ErrElementNotFound
}

// Index searches for and retrieves the element at the given index. This method will panic if
// the document is invalid or if the index is out of bounds.
func (d Document) Index(index uint) Element {
	elem, err := d.IndexErr(index)
	if err != nil {
		panic(err)
	}
	return elem
}

// IndexErr searches for and retrieves the element at the given index.
func (d Document) IndexErr(index uint) (Element, error) {
	length, rem, ok := ReadLength(d)
	if !ok {
		return nil, NewInsufficientBytesError(d, rem)
	}

	length -= 4

	var current uint
	var elem Element
	for length > 1 {
		elem, rem, ok = ReadElement(rem)
		length -= int32(len(elem))
		if !ok {
			return nil, NewInsufficientBytesError(d, rem)
		}
		if current != index {
			current++
			continue
		}
		return elem, nil
	}
	return nil, ErrOutOfBounds
}

// DebugString outputs a human readable version of Document. It will attempt to stringify the
// valid components of the document even if the entire document is not valid.
func (d Document) DebugString() string {
	if len(d) < 5 {
		return "<malformed>"
	}
	var buf bytes.Buffer
	buf.WriteString("Document")
	length, rem, _ := ReadLength(d) // We know we have enough bytes to read the length
	buf.WriteByte('(')
	buf.WriteString(strconv.Itoa(int(length)))
	length -= 4
	buf.WriteString("){")
	var elem Element
	var ok bool
	for length > 1 {
		elem, rem, ok = ReadElement(rem)
		length -= int32(len(elem))
		if !ok {
			buf.WriteString(fmt.Sprintf("<malformed (%d)>", length))
			break
		}
		fmt.Fprintf(&buf, "%s ", elem.DebugString())
	}
	buf.WriteByte('}')

	return buf.String()
}

// String outputs an ExtendedJSON version of Document. If the document is not valid, this method
// returns an empty string.
func (d Document) String() string {
	if len(d) < 5 {
		return ""
	}
	var buf bytes.Buffer
	buf.WriteByte('{')

	length, rem, _ := ReadLength(d) // We know we have enough bytes to read the length

	length -= 4

	var elem Element
	var ok bool
	first := true
	for length > 1 {
		if !first {
			buf.WriteByte(',')
		}
		elem, rem, ok = ReadElement(rem)
		length -= int32(len(elem))
		if !ok {
			return ""
		}
		fmt.Fprintf(&buf, "%s", elem.String())
		first = false
	}
	buf.WriteByte('}')

	return buf.String()
}

// Elements returns this document as a slice of elements. The returned slice will contain valid
// elements. If the document is not valid, the elements up to the invalid point will be returned
// along with an error.
func (d Document) Elements() ([]Element, error) {
	length, rem, ok := ReadLength(d)
	if !ok {
		return nil, NewInsufficientBytesError(d, rem)
	}

	length -= 4

	var elem Element
	var elems []Element
	for length > 1 {
		elem, rem, ok = ReadElement(rem)
		length -= int32(len(elem))
		if !ok {
			return elems, NewInsufficientBytesError(d, rem)
		}
		if err := elem.Validate(); err != nil {
			return elems, err
		}
		elems = append(elems, elem)
	}
	return elems, nil
}

// Values returns this document as a slice of values. The returned slice will contain valid values.
// If the document is not valid, the values up to the invalid point will be returned along with an
// error.
func (d Document) Values() ([]Value, error) {
	length, rem, ok := ReadLength(d)
	if !ok {
		return nil, NewInsufficientBytesError(d, rem)
	}

	length -= 4

	var elem Element
	var vals []Value
	for length > 1 {
		elem, rem, ok = ReadElement(rem)
		length -= int32(len(elem))
		if !ok {
			return vals, NewInsufficientBytesError(d, rem)
		}
		if err := elem.Value().Validate(); err != nil {
			return vals, err
		}
		vals = append(vals, elem.Value())
	}
	return vals, nil
}

// Validate validates the document and ensures the elements contained within are valid.
func (d Document) Validate() error {
	length, rem, ok := ReadLength(d)
	if !ok {
		return NewInsufficientBytesError(d, rem)
	}
	if int(length) > len(d) {
		return d.lengtherror(int(length), len(d))
	}
	if d[length-1] != 0x00 {
		return ErrMissingNull
	}

	length -= 4
	var elem Element

	for length > 1 {
		elem, rem, ok = ReadElement(rem)
		length -= int32(len(elem))
		if !ok {
			return NewInsufficientBytesError(d, rem)
		}
		err := elem.Validate()
		if err != nil {
			return err
		}
	}

	if len(rem) < 1 || rem[0] != 0x00 {
		return ErrMissingNull
	}
	return nil
}

func (Document) lengtherror(length, rem int) error {
	return DocumentValidationError(fmt.Sprintf("document length exceeds available bytes. length=%d remainingBytes=%d", length, rem))
}
