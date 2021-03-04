// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsonrw

import (
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"sync"

	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/x/bsonx/bsoncore"
)

var _ ValueWriter = (*valueWriter)(nil)

var vwPool = sync.Pool{
	New: func() interface{} {
		return new(valueWriter)
	},
}

// BSONValueWriterPool is a pool for BSON ValueWriters.
type BSONValueWriterPool struct {
	pool sync.Pool
}

// NewBSONValueWriterPool creates a new pool for ValueWriter instances that write to BSON.
func NewBSONValueWriterPool() *BSONValueWriterPool {
	return &BSONValueWriterPool{
		pool: sync.Pool{
			New: func() interface{} {
				return new(valueWriter)
			},
		},
	}
}

// Get retrieves a BSON ValueWriter from the pool and resets it to use w as the destination.
func (bvwp *BSONValueWriterPool) Get(w io.Writer) ValueWriter {
	vw := bvwp.pool.Get().(*valueWriter)
	if writer, ok := w.(*SliceWriter); ok {
		vw.reset(*writer)
		vw.w = writer
		return vw
	}
	vw.buf = vw.buf[:0]
	vw.w = w
	return vw
}

// Put inserts a ValueWriter into the pool. If the ValueWriter is not a BSON ValueWriter, nothing
// happens and ok will be false.
func (bvwp *BSONValueWriterPool) Put(vw ValueWriter) (ok bool) {
	bvw, ok := vw.(*valueWriter)
	if !ok {
		return false
	}

	if _, ok := bvw.w.(*SliceWriter); ok {
		bvw.buf = nil
	}
	bvw.w = nil

	bvwp.pool.Put(bvw)
	return true
}

// This is here so that during testing we can change it and not require
// allocating a 4GB slice.
var maxSize = math.MaxInt32

var errNilWriter = errors.New("cannot create a ValueWriter from a nil io.Writer")

type errMaxDocumentSizeExceeded struct {
	size int64
}

func (mdse errMaxDocumentSizeExceeded) Error() string {
	return fmt.Sprintf("document size (%d) is larger than the max int32", mdse.size)
}

type vwMode int

const (
	_ vwMode = iota
	vwTopLevel
	vwDocument
	vwArray
	vwValue
	vwElement
	vwCodeWithScope
)

func (vm vwMode) String() string {
	var str string

	switch vm {
	case vwTopLevel:
		str = "TopLevel"
	case vwDocument:
		str = "DocumentMode"
	case vwArray:
		str = "ArrayMode"
	case vwValue:
		str = "ValueMode"
	case vwElement:
		str = "ElementMode"
	case vwCodeWithScope:
		str = "CodeWithScopeMode"
	default:
		str = "UnknownMode"
	}

	return str
}

type vwState struct {
	mode   mode
	key    string
	arrkey int
	start  int32
}

type valueWriter struct {
	w   io.Writer
	buf []byte

	stack []vwState
	frame int64
}

func (vw *valueWriter) advanceFrame() {
	if vw.frame+1 >= int64(len(vw.stack)) { // We need to grow the stack
		length := len(vw.stack)
		if length+1 >= cap(vw.stack) {
			// double it
			buf := make([]vwState, 2*cap(vw.stack)+1)
			copy(buf, vw.stack)
			vw.stack = buf
		}
		vw.stack = vw.stack[:length+1]
	}
	vw.frame++
}

func (vw *valueWriter) push(m mode) {
	vw.advanceFrame()

	// Clean the stack
	vw.stack[vw.frame].mode = m
	vw.stack[vw.frame].key = ""
	vw.stack[vw.frame].arrkey = 0
	vw.stack[vw.frame].start = 0

	vw.stack[vw.frame].mode = m
	switch m {
	case mDocument, mArray, mCodeWithScope:
		vw.reserveLength()
	}
}

func (vw *valueWriter) reserveLength() {
	vw.stack[vw.frame].start = int32(len(vw.buf))
	vw.buf = append(vw.buf, 0x00, 0x00, 0x00, 0x00)
}

func (vw *valueWriter) pop() {
	switch vw.stack[vw.frame].mode {
	case mElement, mValue:
		vw.frame--
	case mDocument, mArray, mCodeWithScope:
		vw.frame -= 2 // we pop twice to jump over the mElement: mDocument -> mElement -> mDocument/mTopLevel/etc...
	}
}

// NewBSONValueWriter creates a ValueWriter that writes BSON to w.
//
// This ValueWriter will only write entire documents to the io.Writer and it
// will buffer the document as it is built.
func NewBSONValueWriter(w io.Writer) (ValueWriter, error) {
	if w == nil {
		return nil, errNilWriter
	}
	return newValueWriter(w), nil
}

func newValueWriter(w io.Writer) *valueWriter {
	vw := new(valueWriter)
	stack := make([]vwState, 1, 5)
	stack[0] = vwState{mode: mTopLevel}
	vw.w = w
	vw.stack = stack

	return vw
}

func newValueWriterFromSlice(buf []byte) *valueWriter {
	vw := new(valueWriter)
	stack := make([]vwState, 1, 5)
	stack[0] = vwState{mode: mTopLevel}
	vw.stack = stack
	vw.buf = buf

	return vw
}

func (vw *valueWriter) reset(buf []byte) {
	if vw.stack == nil {
		vw.stack = make([]vwState, 1, 5)
	}
	vw.stack = vw.stack[:1]
	vw.stack[0] = vwState{mode: mTopLevel}
	vw.buf = buf
	vw.frame = 0
	vw.w = nil
}

func (vw *valueWriter) invalidTransitionError(destination mode, name string, modes []mode) error {
	te := TransitionError{
		name:        name,
		current:     vw.stack[vw.frame].mode,
		destination: destination,
		modes:       modes,
		action:      "write",
	}
	if vw.frame != 0 {
		te.parent = vw.stack[vw.frame-1].mode
	}
	return te
}

func (vw *valueWriter) writeElementHeader(t bsontype.Type, destination mode, callerName string, addmodes ...mode) error {
	switch vw.stack[vw.frame].mode {
	case mElement:
		vw.buf = bsoncore.AppendHeader(vw.buf, t, vw.stack[vw.frame].key)
	case mValue:
		// TODO: Do this with a cache of the first 1000 or so array keys.
		vw.buf = bsoncore.AppendHeader(vw.buf, t, strconv.Itoa(vw.stack[vw.frame].arrkey))
	default:
		modes := []mode{mElement, mValue}
		if addmodes != nil {
			modes = append(modes, addmodes...)
		}
		return vw.invalidTransitionError(destination, callerName, modes)
	}

	return nil
}

func (vw *valueWriter) WriteValueBytes(t bsontype.Type, b []byte) error {
	if err := vw.writeElementHeader(t, mode(0), "WriteValueBytes"); err != nil {
		return err
	}
	vw.buf = append(vw.buf, b...)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteArray() (ArrayWriter, error) {
	if err := vw.writeElementHeader(bsontype.Array, mArray, "WriteArray"); err != nil {
		return nil, err
	}

	vw.push(mArray)

	return vw, nil
}

func (vw *valueWriter) WriteBinary(b []byte) error {
	return vw.WriteBinaryWithSubtype(b, 0x00)
}

func (vw *valueWriter) WriteBinaryWithSubtype(b []byte, btype byte) error {
	if err := vw.writeElementHeader(bsontype.Binary, mode(0), "WriteBinaryWithSubtype"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendBinary(vw.buf, btype, b)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteBoolean(b bool) error {
	if err := vw.writeElementHeader(bsontype.Boolean, mode(0), "WriteBoolean"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendBoolean(vw.buf, b)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteCodeWithScope(code string) (DocumentWriter, error) {
	if err := vw.writeElementHeader(bsontype.CodeWithScope, mCodeWithScope, "WriteCodeWithScope"); err != nil {
		return nil, err
	}

	// CodeWithScope is a different than other types because we need an extra
	// frame on the stack. In the EndDocument code, we write the document
	// length, pop, write the code with scope length, and pop. To simplify the
	// pop code, we push a spacer frame that we'll always jump over.
	vw.push(mCodeWithScope)
	vw.buf = bsoncore.AppendString(vw.buf, code)
	vw.push(mSpacer)
	vw.push(mDocument)

	return vw, nil
}

func (vw *valueWriter) WriteDBPointer(ns string, oid primitive.ObjectID) error {
	if err := vw.writeElementHeader(bsontype.DBPointer, mode(0), "WriteDBPointer"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendDBPointer(vw.buf, ns, oid)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteDateTime(dt int64) error {
	if err := vw.writeElementHeader(bsontype.DateTime, mode(0), "WriteDateTime"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendDateTime(vw.buf, dt)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteDecimal128(d128 primitive.Decimal128) error {
	if err := vw.writeElementHeader(bsontype.Decimal128, mode(0), "WriteDecimal128"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendDecimal128(vw.buf, d128)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteDouble(f float64) error {
	if err := vw.writeElementHeader(bsontype.Double, mode(0), "WriteDouble"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendDouble(vw.buf, f)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteInt32(i32 int32) error {
	if err := vw.writeElementHeader(bsontype.Int32, mode(0), "WriteInt32"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendInt32(vw.buf, i32)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteInt64(i64 int64) error {
	if err := vw.writeElementHeader(bsontype.Int64, mode(0), "WriteInt64"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendInt64(vw.buf, i64)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteJavascript(code string) error {
	if err := vw.writeElementHeader(bsontype.JavaScript, mode(0), "WriteJavascript"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendJavaScript(vw.buf, code)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteMaxKey() error {
	if err := vw.writeElementHeader(bsontype.MaxKey, mode(0), "WriteMaxKey"); err != nil {
		return err
	}

	vw.pop()
	return nil
}

func (vw *valueWriter) WriteMinKey() error {
	if err := vw.writeElementHeader(bsontype.MinKey, mode(0), "WriteMinKey"); err != nil {
		return err
	}

	vw.pop()
	return nil
}

func (vw *valueWriter) WriteNull() error {
	if err := vw.writeElementHeader(bsontype.Null, mode(0), "WriteNull"); err != nil {
		return err
	}

	vw.pop()
	return nil
}

func (vw *valueWriter) WriteObjectID(oid primitive.ObjectID) error {
	if err := vw.writeElementHeader(bsontype.ObjectID, mode(0), "WriteObjectID"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendObjectID(vw.buf, oid)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteRegex(pattern string, options string) error {
	if err := vw.writeElementHeader(bsontype.Regex, mode(0), "WriteRegex"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendRegex(vw.buf, pattern, sortStringAlphebeticAscending(options))
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteString(s string) error {
	if err := vw.writeElementHeader(bsontype.String, mode(0), "WriteString"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendString(vw.buf, s)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteDocument() (DocumentWriter, error) {
	if vw.stack[vw.frame].mode == mTopLevel {
		vw.reserveLength()
		return vw, nil
	}
	if err := vw.writeElementHeader(bsontype.EmbeddedDocument, mDocument, "WriteDocument", mTopLevel); err != nil {
		return nil, err
	}

	vw.push(mDocument)
	return vw, nil
}

func (vw *valueWriter) WriteSymbol(symbol string) error {
	if err := vw.writeElementHeader(bsontype.Symbol, mode(0), "WriteSymbol"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendSymbol(vw.buf, symbol)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteTimestamp(t uint32, i uint32) error {
	if err := vw.writeElementHeader(bsontype.Timestamp, mode(0), "WriteTimestamp"); err != nil {
		return err
	}

	vw.buf = bsoncore.AppendTimestamp(vw.buf, t, i)
	vw.pop()
	return nil
}

func (vw *valueWriter) WriteUndefined() error {
	if err := vw.writeElementHeader(bsontype.Undefined, mode(0), "WriteUndefined"); err != nil {
		return err
	}

	vw.pop()
	return nil
}

func (vw *valueWriter) WriteDocumentElement(key string) (ValueWriter, error) {
	switch vw.stack[vw.frame].mode {
	case mTopLevel, mDocument:
	default:
		return nil, vw.invalidTransitionError(mElement, "WriteDocumentElement", []mode{mTopLevel, mDocument})
	}

	vw.push(mElement)
	vw.stack[vw.frame].key = key

	return vw, nil
}

func (vw *valueWriter) WriteDocumentEnd() error {
	switch vw.stack[vw.frame].mode {
	case mTopLevel, mDocument:
	default:
		return fmt.Errorf("incorrect mode to end document: %s", vw.stack[vw.frame].mode)
	}

	vw.buf = append(vw.buf, 0x00)

	err := vw.writeLength()
	if err != nil {
		return err
	}

	if vw.stack[vw.frame].mode == mTopLevel {
		if vw.w != nil {
			if sw, ok := vw.w.(*SliceWriter); ok {
				*sw = vw.buf
			} else {
				_, err = vw.w.Write(vw.buf)
				if err != nil {
					return err
				}
				// reset buffer
				vw.buf = vw.buf[:0]
			}
		}
	}

	vw.pop()

	if vw.stack[vw.frame].mode == mCodeWithScope {
		// We ignore the error here because of the gaurantee of writeLength.
		// See the docs for writeLength for more info.
		_ = vw.writeLength()
		vw.pop()
	}
	return nil
}

func (vw *valueWriter) WriteArrayElement() (ValueWriter, error) {
	if vw.stack[vw.frame].mode != mArray {
		return nil, vw.invalidTransitionError(mValue, "WriteArrayElement", []mode{mArray})
	}

	arrkey := vw.stack[vw.frame].arrkey
	vw.stack[vw.frame].arrkey++

	vw.push(mValue)
	vw.stack[vw.frame].arrkey = arrkey

	return vw, nil
}

func (vw *valueWriter) WriteArrayEnd() error {
	if vw.stack[vw.frame].mode != mArray {
		return fmt.Errorf("incorrect mode to end array: %s", vw.stack[vw.frame].mode)
	}

	vw.buf = append(vw.buf, 0x00)

	err := vw.writeLength()
	if err != nil {
		return err
	}

	vw.pop()
	return nil
}

// NOTE: We assume that if we call writeLength more than once the same function
// within the same function without altering the vw.buf that this method will
// not return an error. If this changes ensure that the following methods are
// updated:
//
// - WriteDocumentEnd
func (vw *valueWriter) writeLength() error {
	length := len(vw.buf)
	if length > maxSize {
		return errMaxDocumentSizeExceeded{size: int64(len(vw.buf))}
	}
	length = length - int(vw.stack[vw.frame].start)
	start := vw.stack[vw.frame].start

	vw.buf[start+0] = byte(length)
	vw.buf[start+1] = byte(length >> 8)
	vw.buf[start+2] = byte(length >> 16)
	vw.buf[start+3] = byte(length >> 24)
	return nil
}
