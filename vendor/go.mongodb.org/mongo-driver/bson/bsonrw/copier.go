// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsonrw

import (
	"fmt"
	"io"

	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/x/bsonx/bsoncore"
)

// Copier is a type that allows copying between ValueReaders, ValueWriters, and
// []byte values.
type Copier struct{}

// NewCopier creates a new copier with the given registry. If a nil registry is provided
// a default registry is used.
func NewCopier() Copier {
	return Copier{}
}

// CopyDocument handles copying a document from src to dst.
func CopyDocument(dst ValueWriter, src ValueReader) error {
	return Copier{}.CopyDocument(dst, src)
}

// CopyDocument handles copying one document from the src to the dst.
func (c Copier) CopyDocument(dst ValueWriter, src ValueReader) error {
	dr, err := src.ReadDocument()
	if err != nil {
		return err
	}

	dw, err := dst.WriteDocument()
	if err != nil {
		return err
	}

	return c.copyDocumentCore(dw, dr)
}

// CopyDocumentFromBytes copies the values from a BSON document represented as a
// []byte to a ValueWriter.
func (c Copier) CopyDocumentFromBytes(dst ValueWriter, src []byte) error {
	dw, err := dst.WriteDocument()
	if err != nil {
		return err
	}

	err = c.CopyBytesToDocumentWriter(dw, src)
	if err != nil {
		return err
	}

	return dw.WriteDocumentEnd()
}

// CopyBytesToDocumentWriter copies the values from a BSON document represented as a []byte to a
// DocumentWriter.
func (c Copier) CopyBytesToDocumentWriter(dst DocumentWriter, src []byte) error {
	// TODO(skriptble): Create errors types here. Anything thats a tag should be a property.
	length, rem, ok := bsoncore.ReadLength(src)
	if !ok {
		return fmt.Errorf("couldn't read length from src, not enough bytes. length=%d", len(src))
	}
	if len(src) < int(length) {
		return fmt.Errorf("length read exceeds number of bytes available. length=%d bytes=%d", len(src), length)
	}
	rem = rem[:length-4]

	var t bsontype.Type
	var key string
	var val bsoncore.Value
	for {
		t, rem, ok = bsoncore.ReadType(rem)
		if !ok {
			return io.EOF
		}
		if t == bsontype.Type(0) {
			if len(rem) != 0 {
				return fmt.Errorf("document end byte found before end of document. remaining bytes=%v", rem)
			}
			break
		}

		key, rem, ok = bsoncore.ReadKey(rem)
		if !ok {
			return fmt.Errorf("invalid key found. remaining bytes=%v", rem)
		}
		dvw, err := dst.WriteDocumentElement(key)
		if err != nil {
			return err
		}
		val, rem, ok = bsoncore.ReadValue(rem, t)
		if !ok {
			return fmt.Errorf("not enough bytes available to read type. bytes=%d type=%s", len(rem), t)
		}
		err = c.CopyValueFromBytes(dvw, t, val.Data)
		if err != nil {
			return err
		}
	}
	return nil
}

// CopyDocumentToBytes copies an entire document from the ValueReader and
// returns it as bytes.
func (c Copier) CopyDocumentToBytes(src ValueReader) ([]byte, error) {
	return c.AppendDocumentBytes(nil, src)
}

// AppendDocumentBytes functions the same as CopyDocumentToBytes, but will
// append the result to dst.
func (c Copier) AppendDocumentBytes(dst []byte, src ValueReader) ([]byte, error) {
	if br, ok := src.(BytesReader); ok {
		_, dst, err := br.ReadValueBytes(dst)
		return dst, err
	}

	vw := vwPool.Get().(*valueWriter)
	defer vwPool.Put(vw)

	vw.reset(dst)

	err := c.CopyDocument(vw, src)
	dst = vw.buf
	return dst, err
}

// CopyValueFromBytes will write the value represtend by t and src to dst.
func (c Copier) CopyValueFromBytes(dst ValueWriter, t bsontype.Type, src []byte) error {
	if wvb, ok := dst.(BytesWriter); ok {
		return wvb.WriteValueBytes(t, src)
	}

	vr := vrPool.Get().(*valueReader)
	defer vrPool.Put(vr)

	vr.reset(src)
	vr.pushElement(t)

	return c.CopyValue(dst, vr)
}

// CopyValueToBytes copies a value from src and returns it as a bsontype.Type and a
// []byte.
func (c Copier) CopyValueToBytes(src ValueReader) (bsontype.Type, []byte, error) {
	return c.AppendValueBytes(nil, src)
}

// AppendValueBytes functions the same as CopyValueToBytes, but will append the
// result to dst.
func (c Copier) AppendValueBytes(dst []byte, src ValueReader) (bsontype.Type, []byte, error) {
	if br, ok := src.(BytesReader); ok {
		return br.ReadValueBytes(dst)
	}

	vw := vwPool.Get().(*valueWriter)
	defer vwPool.Put(vw)

	start := len(dst)

	vw.reset(dst)
	vw.push(mElement)

	err := c.CopyValue(vw, src)
	if err != nil {
		return 0, dst, err
	}

	return bsontype.Type(vw.buf[start]), vw.buf[start+2:], nil
}

// CopyValue will copy a single value from src to dst.
func (c Copier) CopyValue(dst ValueWriter, src ValueReader) error {
	var err error
	switch src.Type() {
	case bsontype.Double:
		var f64 float64
		f64, err = src.ReadDouble()
		if err != nil {
			break
		}
		err = dst.WriteDouble(f64)
	case bsontype.String:
		var str string
		str, err = src.ReadString()
		if err != nil {
			return err
		}
		err = dst.WriteString(str)
	case bsontype.EmbeddedDocument:
		err = c.CopyDocument(dst, src)
	case bsontype.Array:
		err = c.copyArray(dst, src)
	case bsontype.Binary:
		var data []byte
		var subtype byte
		data, subtype, err = src.ReadBinary()
		if err != nil {
			break
		}
		err = dst.WriteBinaryWithSubtype(data, subtype)
	case bsontype.Undefined:
		err = src.ReadUndefined()
		if err != nil {
			break
		}
		err = dst.WriteUndefined()
	case bsontype.ObjectID:
		var oid primitive.ObjectID
		oid, err = src.ReadObjectID()
		if err != nil {
			break
		}
		err = dst.WriteObjectID(oid)
	case bsontype.Boolean:
		var b bool
		b, err = src.ReadBoolean()
		if err != nil {
			break
		}
		err = dst.WriteBoolean(b)
	case bsontype.DateTime:
		var dt int64
		dt, err = src.ReadDateTime()
		if err != nil {
			break
		}
		err = dst.WriteDateTime(dt)
	case bsontype.Null:
		err = src.ReadNull()
		if err != nil {
			break
		}
		err = dst.WriteNull()
	case bsontype.Regex:
		var pattern, options string
		pattern, options, err = src.ReadRegex()
		if err != nil {
			break
		}
		err = dst.WriteRegex(pattern, options)
	case bsontype.DBPointer:
		var ns string
		var pointer primitive.ObjectID
		ns, pointer, err = src.ReadDBPointer()
		if err != nil {
			break
		}
		err = dst.WriteDBPointer(ns, pointer)
	case bsontype.JavaScript:
		var js string
		js, err = src.ReadJavascript()
		if err != nil {
			break
		}
		err = dst.WriteJavascript(js)
	case bsontype.Symbol:
		var symbol string
		symbol, err = src.ReadSymbol()
		if err != nil {
			break
		}
		err = dst.WriteSymbol(symbol)
	case bsontype.CodeWithScope:
		var code string
		var srcScope DocumentReader
		code, srcScope, err = src.ReadCodeWithScope()
		if err != nil {
			break
		}

		var dstScope DocumentWriter
		dstScope, err = dst.WriteCodeWithScope(code)
		if err != nil {
			break
		}
		err = c.copyDocumentCore(dstScope, srcScope)
	case bsontype.Int32:
		var i32 int32
		i32, err = src.ReadInt32()
		if err != nil {
			break
		}
		err = dst.WriteInt32(i32)
	case bsontype.Timestamp:
		var t, i uint32
		t, i, err = src.ReadTimestamp()
		if err != nil {
			break
		}
		err = dst.WriteTimestamp(t, i)
	case bsontype.Int64:
		var i64 int64
		i64, err = src.ReadInt64()
		if err != nil {
			break
		}
		err = dst.WriteInt64(i64)
	case bsontype.Decimal128:
		var d128 primitive.Decimal128
		d128, err = src.ReadDecimal128()
		if err != nil {
			break
		}
		err = dst.WriteDecimal128(d128)
	case bsontype.MinKey:
		err = src.ReadMinKey()
		if err != nil {
			break
		}
		err = dst.WriteMinKey()
	case bsontype.MaxKey:
		err = src.ReadMaxKey()
		if err != nil {
			break
		}
		err = dst.WriteMaxKey()
	default:
		err = fmt.Errorf("Cannot copy unknown BSON type %s", src.Type())
	}

	return err
}

func (c Copier) copyArray(dst ValueWriter, src ValueReader) error {
	ar, err := src.ReadArray()
	if err != nil {
		return err
	}

	aw, err := dst.WriteArray()
	if err != nil {
		return err
	}

	for {
		vr, err := ar.ReadValue()
		if err == ErrEOA {
			break
		}
		if err != nil {
			return err
		}

		vw, err := aw.WriteArrayElement()
		if err != nil {
			return err
		}

		err = c.CopyValue(vw, vr)
		if err != nil {
			return err
		}
	}

	return aw.WriteArrayEnd()
}

func (c Copier) copyDocumentCore(dw DocumentWriter, dr DocumentReader) error {
	for {
		key, vr, err := dr.ReadElement()
		if err == ErrEOD {
			break
		}
		if err != nil {
			return err
		}

		vw, err := dw.WriteDocumentElement(key)
		if err != nil {
			return err
		}

		err = c.CopyValue(vw, vr)
		if err != nil {
			return err
		}
	}

	return dw.WriteDocumentEnd()
}
