// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsonrw

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"unicode"

	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

var _ ValueReader = (*valueReader)(nil)

var vrPool = sync.Pool{
	New: func() interface{} {
		return new(valueReader)
	},
}

// BSONValueReaderPool is a pool for ValueReaders that read BSON.
type BSONValueReaderPool struct {
	pool sync.Pool
}

// NewBSONValueReaderPool instantiates a new BSONValueReaderPool.
func NewBSONValueReaderPool() *BSONValueReaderPool {
	return &BSONValueReaderPool{
		pool: sync.Pool{
			New: func() interface{} {
				return new(valueReader)
			},
		},
	}
}

// Get retrieves a ValueReader from the pool and uses src as the underlying BSON.
func (bvrp *BSONValueReaderPool) Get(src []byte) ValueReader {
	vr := bvrp.pool.Get().(*valueReader)
	vr.reset(src)
	return vr
}

// Put inserts a ValueReader into the pool. If the ValueReader is not a BSON ValueReader nothing
// is inserted into the pool and ok will be false.
func (bvrp *BSONValueReaderPool) Put(vr ValueReader) (ok bool) {
	bvr, ok := vr.(*valueReader)
	if !ok {
		return false
	}

	bvr.reset(nil)
	bvrp.pool.Put(bvr)
	return true
}

// ErrEOA is the error returned when the end of a BSON array has been reached.
var ErrEOA = errors.New("end of array")

// ErrEOD is the error returned when the end of a BSON document has been reached.
var ErrEOD = errors.New("end of document")

type vrState struct {
	mode  mode
	vType bsontype.Type
	end   int64
}

// valueReader is for reading BSON values.
type valueReader struct {
	offset int64
	d      []byte

	stack []vrState
	frame int64
}

// NewBSONDocumentReader returns a ValueReader using b for the underlying BSON
// representation. Parameter b must be a BSON Document.
//
// TODO(skriptble): There's a lack of symmetry between the reader and writer, since the reader takes
// a []byte while the writer takes an io.Writer. We should have two versions of each, one that takes
// a []byte and one that takes an io.Reader or io.Writer. The []byte version will need to return a
// thing that can return the finished []byte since it might be reallocated when appended to.
func NewBSONDocumentReader(b []byte) ValueReader {
	return newValueReader(b)
}

// NewBSONValueReader returns a ValueReader that starts in the Value mode instead of in top
// level document mode. This enables the creation of a ValueReader for a single BSON value.
func NewBSONValueReader(t bsontype.Type, val []byte) ValueReader {
	stack := make([]vrState, 1, 5)
	stack[0] = vrState{
		mode:  mValue,
		vType: t,
	}
	return &valueReader{
		d:     val,
		stack: stack,
	}
}

func newValueReader(b []byte) *valueReader {
	stack := make([]vrState, 1, 5)
	stack[0] = vrState{
		mode: mTopLevel,
	}
	return &valueReader{
		d:     b,
		stack: stack,
	}
}

func (vr *valueReader) reset(b []byte) {
	if vr.stack == nil {
		vr.stack = make([]vrState, 1, 5)
	}
	vr.stack = vr.stack[:1]
	vr.stack[0] = vrState{mode: mTopLevel}
	vr.d = b
	vr.offset = 0
	vr.frame = 0
}

func (vr *valueReader) advanceFrame() {
	if vr.frame+1 >= int64(len(vr.stack)) { // We need to grow the stack
		length := len(vr.stack)
		if length+1 >= cap(vr.stack) {
			// double it
			buf := make([]vrState, 2*cap(vr.stack)+1)
			copy(buf, vr.stack)
			vr.stack = buf
		}
		vr.stack = vr.stack[:length+1]
	}
	vr.frame++

	// Clean the stack
	vr.stack[vr.frame].mode = 0
	vr.stack[vr.frame].vType = 0
	vr.stack[vr.frame].end = 0
}

func (vr *valueReader) pushDocument() error {
	vr.advanceFrame()

	vr.stack[vr.frame].mode = mDocument

	size, err := vr.readLength()
	if err != nil {
		return err
	}
	vr.stack[vr.frame].end = int64(size) + vr.offset - 4

	return nil
}

func (vr *valueReader) pushArray() error {
	vr.advanceFrame()

	vr.stack[vr.frame].mode = mArray

	size, err := vr.readLength()
	if err != nil {
		return err
	}
	vr.stack[vr.frame].end = int64(size) + vr.offset - 4

	return nil
}

func (vr *valueReader) pushElement(t bsontype.Type) {
	vr.advanceFrame()

	vr.stack[vr.frame].mode = mElement
	vr.stack[vr.frame].vType = t
}

func (vr *valueReader) pushValue(t bsontype.Type) {
	vr.advanceFrame()

	vr.stack[vr.frame].mode = mValue
	vr.stack[vr.frame].vType = t
}

func (vr *valueReader) pushCodeWithScope() (int64, error) {
	vr.advanceFrame()

	vr.stack[vr.frame].mode = mCodeWithScope

	size, err := vr.readLength()
	if err != nil {
		return 0, err
	}
	vr.stack[vr.frame].end = int64(size) + vr.offset - 4

	return int64(size), nil
}

func (vr *valueReader) pop() {
	switch vr.stack[vr.frame].mode {
	case mElement, mValue:
		vr.frame--
	case mDocument, mArray, mCodeWithScope:
		vr.frame -= 2 // we pop twice to jump over the vrElement: vrDocument -> vrElement -> vrDocument/TopLevel/etc...
	}
}

func (vr *valueReader) invalidTransitionErr(destination mode, name string, modes []mode) error {
	te := TransitionError{
		name:        name,
		current:     vr.stack[vr.frame].mode,
		destination: destination,
		modes:       modes,
		action:      "read",
	}
	if vr.frame != 0 {
		te.parent = vr.stack[vr.frame-1].mode
	}
	return te
}

func (vr *valueReader) typeError(t bsontype.Type) error {
	return fmt.Errorf("positioned on %s, but attempted to read %s", vr.stack[vr.frame].vType, t)
}

func (vr *valueReader) invalidDocumentLengthError() error {
	return fmt.Errorf("document is invalid, end byte is at %d, but null byte found at %d", vr.stack[vr.frame].end, vr.offset)
}

func (vr *valueReader) ensureElementValue(t bsontype.Type, destination mode, callerName string) error {
	switch vr.stack[vr.frame].mode {
	case mElement, mValue:
		if vr.stack[vr.frame].vType != t {
			return vr.typeError(t)
		}
	default:
		return vr.invalidTransitionErr(destination, callerName, []mode{mElement, mValue})
	}

	return nil
}

func (vr *valueReader) Type() bsontype.Type {
	return vr.stack[vr.frame].vType
}

func (vr *valueReader) nextElementLength() (int32, error) {
	var length int32
	var err error
	switch vr.stack[vr.frame].vType {
	case bsontype.Array, bsontype.EmbeddedDocument, bsontype.CodeWithScope:
		length, err = vr.peekLength()
	case bsontype.Binary:
		length, err = vr.peekLength()
		length += 4 + 1 // binary length + subtype byte
	case bsontype.Boolean:
		length = 1
	case bsontype.DBPointer:
		length, err = vr.peekLength()
		length += 4 + 12 // string length + ObjectID length
	case bsontype.DateTime, bsontype.Double, bsontype.Int64, bsontype.Timestamp:
		length = 8
	case bsontype.Decimal128:
		length = 16
	case bsontype.Int32:
		length = 4
	case bsontype.JavaScript, bsontype.String, bsontype.Symbol:
		length, err = vr.peekLength()
		length += 4
	case bsontype.MaxKey, bsontype.MinKey, bsontype.Null, bsontype.Undefined:
		length = 0
	case bsontype.ObjectID:
		length = 12
	case bsontype.Regex:
		regex := bytes.IndexByte(vr.d[vr.offset:], 0x00)
		if regex < 0 {
			err = io.EOF
			break
		}
		pattern := bytes.IndexByte(vr.d[vr.offset+int64(regex)+1:], 0x00)
		if pattern < 0 {
			err = io.EOF
			break
		}
		length = int32(int64(regex) + 1 + int64(pattern) + 1)
	default:
		return 0, fmt.Errorf("attempted to read bytes of unknown BSON type %v", vr.stack[vr.frame].vType)
	}

	return length, err
}

func (vr *valueReader) ReadValueBytes(dst []byte) (bsontype.Type, []byte, error) {
	switch vr.stack[vr.frame].mode {
	case mTopLevel:
		length, err := vr.peekLength()
		if err != nil {
			return bsontype.Type(0), nil, err
		}
		dst, err = vr.appendBytes(dst, length)
		if err != nil {
			return bsontype.Type(0), nil, err
		}
		return bsontype.Type(0), dst, nil
	case mElement, mValue:
		length, err := vr.nextElementLength()
		if err != nil {
			return bsontype.Type(0), dst, err
		}

		dst, err = vr.appendBytes(dst, length)
		t := vr.stack[vr.frame].vType
		vr.pop()
		return t, dst, err
	default:
		return bsontype.Type(0), nil, vr.invalidTransitionErr(0, "ReadValueBytes", []mode{mElement, mValue})
	}
}

func (vr *valueReader) Skip() error {
	switch vr.stack[vr.frame].mode {
	case mElement, mValue:
	default:
		return vr.invalidTransitionErr(0, "Skip", []mode{mElement, mValue})
	}

	length, err := vr.nextElementLength()
	if err != nil {
		return err
	}

	err = vr.skipBytes(length)
	vr.pop()
	return err
}

func (vr *valueReader) ReadArray() (ArrayReader, error) {
	if err := vr.ensureElementValue(bsontype.Array, mArray, "ReadArray"); err != nil {
		return nil, err
	}

	err := vr.pushArray()
	if err != nil {
		return nil, err
	}

	return vr, nil
}

func (vr *valueReader) ReadBinary() (b []byte, btype byte, err error) {
	if err := vr.ensureElementValue(bsontype.Binary, 0, "ReadBinary"); err != nil {
		return nil, 0, err
	}

	length, err := vr.readLength()
	if err != nil {
		return nil, 0, err
	}

	btype, err = vr.readByte()
	if err != nil {
		return nil, 0, err
	}

	if btype == 0x02 {
		length, err = vr.readLength()
		if err != nil {
			return nil, 0, err
		}
	}

	b, err = vr.readBytes(length)
	if err != nil {
		return nil, 0, err
	}

	vr.pop()
	return b, btype, nil
}

func (vr *valueReader) ReadBoolean() (bool, error) {
	if err := vr.ensureElementValue(bsontype.Boolean, 0, "ReadBoolean"); err != nil {
		return false, err
	}

	b, err := vr.readByte()
	if err != nil {
		return false, err
	}

	if b > 1 {
		return false, fmt.Errorf("invalid byte for boolean, %b", b)
	}

	vr.pop()
	return b == 1, nil
}

func (vr *valueReader) ReadDocument() (DocumentReader, error) {
	switch vr.stack[vr.frame].mode {
	case mTopLevel:
		// read size
		size, err := vr.readLength()
		if err != nil {
			return nil, err
		}
		if int(size) != len(vr.d) {
			return nil, fmt.Errorf("invalid document length")
		}
		vr.stack[vr.frame].end = int64(size) + vr.offset - 4
		return vr, nil
	case mElement, mValue:
		if vr.stack[vr.frame].vType != bsontype.EmbeddedDocument {
			return nil, vr.typeError(bsontype.EmbeddedDocument)
		}
	default:
		return nil, vr.invalidTransitionErr(mDocument, "ReadDocument", []mode{mTopLevel, mElement, mValue})
	}

	err := vr.pushDocument()
	if err != nil {
		return nil, err
	}

	return vr, nil
}

func (vr *valueReader) ReadCodeWithScope() (code string, dr DocumentReader, err error) {
	if err := vr.ensureElementValue(bsontype.CodeWithScope, 0, "ReadCodeWithScope"); err != nil {
		return "", nil, err
	}

	totalLength, err := vr.readLength()
	if err != nil {
		return "", nil, err
	}
	strLength, err := vr.readLength()
	if err != nil {
		return "", nil, err
	}
	strBytes, err := vr.readBytes(strLength)
	if err != nil {
		return "", nil, err
	}
	code = string(strBytes[:len(strBytes)-1])

	size, err := vr.pushCodeWithScope()
	if err != nil {
		return "", nil, err
	}

	// The total length should equal:
	// 4 (total length) + strLength + 4 (the length of str itself) + (document length)
	componentsLength := int64(4+strLength+4) + size
	if int64(totalLength) != componentsLength {
		return "", nil, fmt.Errorf(
			"length of CodeWithScope does not match lengths of components; total: %d; components: %d",
			totalLength, componentsLength,
		)
	}
	return code, vr, nil
}

func (vr *valueReader) ReadDBPointer() (ns string, oid primitive.ObjectID, err error) {
	if err := vr.ensureElementValue(bsontype.DBPointer, 0, "ReadDBPointer"); err != nil {
		return "", oid, err
	}

	ns, err = vr.readString()
	if err != nil {
		return "", oid, err
	}

	oidbytes, err := vr.readBytes(12)
	if err != nil {
		return "", oid, err
	}

	copy(oid[:], oidbytes)

	vr.pop()
	return ns, oid, nil
}

func (vr *valueReader) ReadDateTime() (int64, error) {
	if err := vr.ensureElementValue(bsontype.DateTime, 0, "ReadDateTime"); err != nil {
		return 0, err
	}

	i, err := vr.readi64()
	if err != nil {
		return 0, err
	}

	vr.pop()
	return i, nil
}

func (vr *valueReader) ReadDecimal128() (primitive.Decimal128, error) {
	if err := vr.ensureElementValue(bsontype.Decimal128, 0, "ReadDecimal128"); err != nil {
		return primitive.Decimal128{}, err
	}

	b, err := vr.readBytes(16)
	if err != nil {
		return primitive.Decimal128{}, err
	}

	l := binary.LittleEndian.Uint64(b[0:8])
	h := binary.LittleEndian.Uint64(b[8:16])

	vr.pop()
	return primitive.NewDecimal128(h, l), nil
}

func (vr *valueReader) ReadDouble() (float64, error) {
	if err := vr.ensureElementValue(bsontype.Double, 0, "ReadDouble"); err != nil {
		return 0, err
	}

	u, err := vr.readu64()
	if err != nil {
		return 0, err
	}

	vr.pop()
	return math.Float64frombits(u), nil
}

func (vr *valueReader) ReadInt32() (int32, error) {
	if err := vr.ensureElementValue(bsontype.Int32, 0, "ReadInt32"); err != nil {
		return 0, err
	}

	vr.pop()
	return vr.readi32()
}

func (vr *valueReader) ReadInt64() (int64, error) {
	if err := vr.ensureElementValue(bsontype.Int64, 0, "ReadInt64"); err != nil {
		return 0, err
	}

	vr.pop()
	return vr.readi64()
}

func (vr *valueReader) ReadJavascript() (code string, err error) {
	if err := vr.ensureElementValue(bsontype.JavaScript, 0, "ReadJavascript"); err != nil {
		return "", err
	}

	vr.pop()
	return vr.readString()
}

func (vr *valueReader) ReadMaxKey() error {
	if err := vr.ensureElementValue(bsontype.MaxKey, 0, "ReadMaxKey"); err != nil {
		return err
	}

	vr.pop()
	return nil
}

func (vr *valueReader) ReadMinKey() error {
	if err := vr.ensureElementValue(bsontype.MinKey, 0, "ReadMinKey"); err != nil {
		return err
	}

	vr.pop()
	return nil
}

func (vr *valueReader) ReadNull() error {
	if err := vr.ensureElementValue(bsontype.Null, 0, "ReadNull"); err != nil {
		return err
	}

	vr.pop()
	return nil
}

func (vr *valueReader) ReadObjectID() (primitive.ObjectID, error) {
	if err := vr.ensureElementValue(bsontype.ObjectID, 0, "ReadObjectID"); err != nil {
		return primitive.ObjectID{}, err
	}

	oidbytes, err := vr.readBytes(12)
	if err != nil {
		return primitive.ObjectID{}, err
	}

	var oid primitive.ObjectID
	copy(oid[:], oidbytes)

	vr.pop()
	return oid, nil
}

func (vr *valueReader) ReadRegex() (string, string, error) {
	if err := vr.ensureElementValue(bsontype.Regex, 0, "ReadRegex"); err != nil {
		return "", "", err
	}

	pattern, err := vr.readCString()
	if err != nil {
		return "", "", err
	}

	options, err := vr.readCString()
	if err != nil {
		return "", "", err
	}

	vr.pop()
	return pattern, options, nil
}

func (vr *valueReader) ReadString() (string, error) {
	if err := vr.ensureElementValue(bsontype.String, 0, "ReadString"); err != nil {
		return "", err
	}

	vr.pop()
	return vr.readString()
}

func (vr *valueReader) ReadSymbol() (symbol string, err error) {
	if err := vr.ensureElementValue(bsontype.Symbol, 0, "ReadSymbol"); err != nil {
		return "", err
	}

	vr.pop()
	return vr.readString()
}

func (vr *valueReader) ReadTimestamp() (t uint32, i uint32, err error) {
	if err := vr.ensureElementValue(bsontype.Timestamp, 0, "ReadTimestamp"); err != nil {
		return 0, 0, err
	}

	i, err = vr.readu32()
	if err != nil {
		return 0, 0, err
	}

	t, err = vr.readu32()
	if err != nil {
		return 0, 0, err
	}

	vr.pop()
	return t, i, nil
}

func (vr *valueReader) ReadUndefined() error {
	if err := vr.ensureElementValue(bsontype.Undefined, 0, "ReadUndefined"); err != nil {
		return err
	}

	vr.pop()
	return nil
}

func (vr *valueReader) ReadElement() (string, ValueReader, error) {
	switch vr.stack[vr.frame].mode {
	case mTopLevel, mDocument, mCodeWithScope:
	default:
		return "", nil, vr.invalidTransitionErr(mElement, "ReadElement", []mode{mTopLevel, mDocument, mCodeWithScope})
	}

	t, err := vr.readByte()
	if err != nil {
		return "", nil, err
	}

	if t == 0 {
		if vr.offset != vr.stack[vr.frame].end {
			return "", nil, vr.invalidDocumentLengthError()
		}

		vr.pop()
		return "", nil, ErrEOD
	}

	name, err := vr.readCString()
	if err != nil {
		return "", nil, err
	}

	vr.pushElement(bsontype.Type(t))
	return name, vr, nil
}

func (vr *valueReader) ReadValue() (ValueReader, error) {
	switch vr.stack[vr.frame].mode {
	case mArray:
	default:
		return nil, vr.invalidTransitionErr(mValue, "ReadValue", []mode{mArray})
	}

	t, err := vr.readByte()
	if err != nil {
		return nil, err
	}

	if t == 0 {
		if vr.offset != vr.stack[vr.frame].end {
			return nil, vr.invalidDocumentLengthError()
		}

		vr.pop()
		return nil, ErrEOA
	}

	_, err = vr.readCString()
	if err != nil {
		return nil, err
	}

	vr.pushValue(bsontype.Type(t))
	return vr, nil
}

func (vr *valueReader) readBytes(length int32) ([]byte, error) {
	if length < 0 {
		return nil, fmt.Errorf("invalid length: %d", length)
	}

	if vr.offset+int64(length) > int64(len(vr.d)) {
		return nil, io.EOF
	}

	start := vr.offset
	vr.offset += int64(length)
	return vr.d[start : start+int64(length)], nil
}

func (vr *valueReader) appendBytes(dst []byte, length int32) ([]byte, error) {
	if vr.offset+int64(length) > int64(len(vr.d)) {
		return nil, io.EOF
	}

	start := vr.offset
	vr.offset += int64(length)
	return append(dst, vr.d[start:start+int64(length)]...), nil
}

func (vr *valueReader) skipBytes(length int32) error {
	if vr.offset+int64(length) > int64(len(vr.d)) {
		return io.EOF
	}

	vr.offset += int64(length)
	return nil
}

func (vr *valueReader) readByte() (byte, error) {
	if vr.offset+1 > int64(len(vr.d)) {
		return 0x0, io.EOF
	}

	vr.offset++
	return vr.d[vr.offset-1], nil
}

func (vr *valueReader) readCString() (string, error) {
	idx := bytes.IndexByte(vr.d[vr.offset:], 0x00)
	if idx < 0 {
		return "", io.EOF
	}
	start := vr.offset
	// idx does not include the null byte
	vr.offset += int64(idx) + 1
	return string(vr.d[start : start+int64(idx)]), nil
}

func (vr *valueReader) skipCString() error {
	idx := bytes.IndexByte(vr.d[vr.offset:], 0x00)
	if idx < 0 {
		return io.EOF
	}
	// idx does not include the null byte
	vr.offset += int64(idx) + 1
	return nil
}

func (vr *valueReader) readString() (string, error) {
	length, err := vr.readLength()
	if err != nil {
		return "", err
	}

	if int64(length)+vr.offset > int64(len(vr.d)) {
		return "", io.EOF
	}

	if length <= 0 {
		return "", fmt.Errorf("invalid string length: %d", length)
	}

	if vr.d[vr.offset+int64(length)-1] != 0x00 {
		return "", fmt.Errorf("string does not end with null byte, but with %v", vr.d[vr.offset+int64(length)-1])
	}

	start := vr.offset
	vr.offset += int64(length)

	if length == 2 {
		asciiByte := vr.d[start]
		if asciiByte > unicode.MaxASCII {
			return "", fmt.Errorf("invalid ascii byte")
		}
	}

	return string(vr.d[start : start+int64(length)-1]), nil
}

func (vr *valueReader) peekLength() (int32, error) {
	if vr.offset+4 > int64(len(vr.d)) {
		return 0, io.EOF
	}

	idx := vr.offset
	return (int32(vr.d[idx]) | int32(vr.d[idx+1])<<8 | int32(vr.d[idx+2])<<16 | int32(vr.d[idx+3])<<24), nil
}

func (vr *valueReader) readLength() (int32, error) { return vr.readi32() }

func (vr *valueReader) readi32() (int32, error) {
	if vr.offset+4 > int64(len(vr.d)) {
		return 0, io.EOF
	}

	idx := vr.offset
	vr.offset += 4
	return (int32(vr.d[idx]) | int32(vr.d[idx+1])<<8 | int32(vr.d[idx+2])<<16 | int32(vr.d[idx+3])<<24), nil
}

func (vr *valueReader) readu32() (uint32, error) {
	if vr.offset+4 > int64(len(vr.d)) {
		return 0, io.EOF
	}

	idx := vr.offset
	vr.offset += 4
	return (uint32(vr.d[idx]) | uint32(vr.d[idx+1])<<8 | uint32(vr.d[idx+2])<<16 | uint32(vr.d[idx+3])<<24), nil
}

func (vr *valueReader) readi64() (int64, error) {
	if vr.offset+8 > int64(len(vr.d)) {
		return 0, io.EOF
	}

	idx := vr.offset
	vr.offset += 8
	return int64(vr.d[idx]) | int64(vr.d[idx+1])<<8 | int64(vr.d[idx+2])<<16 | int64(vr.d[idx+3])<<24 |
		int64(vr.d[idx+4])<<32 | int64(vr.d[idx+5])<<40 | int64(vr.d[idx+6])<<48 | int64(vr.d[idx+7])<<56, nil
}

func (vr *valueReader) readu64() (uint64, error) {
	if vr.offset+8 > int64(len(vr.d)) {
		return 0, io.EOF
	}

	idx := vr.offset
	vr.offset += 8
	return uint64(vr.d[idx]) | uint64(vr.d[idx+1])<<8 | uint64(vr.d[idx+2])<<16 | uint64(vr.d[idx+3])<<24 |
		uint64(vr.d[idx+4])<<32 | uint64(vr.d[idx+5])<<40 | uint64(vr.d[idx+6])<<48 | uint64(vr.d[idx+7])<<56, nil
}
