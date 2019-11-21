// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bson

import (
	"errors"
	"fmt"
	"reflect"
	"sync"

	"go.mongodb.org/mongo-driver/bson/bsoncodec"
	"go.mongodb.org/mongo-driver/bson/bsonrw"
)

// ErrDecodeToNil is the error returned when trying to decode to a nil value
var ErrDecodeToNil = errors.New("cannot Decode to nil value")

// This pool is used to keep the allocations of Decoders down. This is only used for the Marshal*
// methods and is not consumable from outside of this package. The Decoders retrieved from this pool
// must have both Reset and SetRegistry called on them.
var decPool = sync.Pool{
	New: func() interface{} {
		return new(Decoder)
	},
}

// A Decoder reads and decodes BSON documents from a stream. It reads from a bsonrw.ValueReader as
// the source of BSON data.
type Decoder struct {
	dc bsoncodec.DecodeContext
	vr bsonrw.ValueReader
}

// NewDecoder returns a new decoder that uses the DefaultRegistry to read from vr.
func NewDecoder(vr bsonrw.ValueReader) (*Decoder, error) {
	if vr == nil {
		return nil, errors.New("cannot create a new Decoder with a nil ValueReader")
	}

	return &Decoder{
		dc: bsoncodec.DecodeContext{Registry: DefaultRegistry},
		vr: vr,
	}, nil
}

// NewDecoderWithContext returns a new decoder that uses DecodeContext dc to read from vr.
func NewDecoderWithContext(dc bsoncodec.DecodeContext, vr bsonrw.ValueReader) (*Decoder, error) {
	if dc.Registry == nil {
		dc.Registry = DefaultRegistry
	}
	if vr == nil {
		return nil, errors.New("cannot create a new Decoder with a nil ValueReader")
	}

	return &Decoder{
		dc: dc,
		vr: vr,
	}, nil
}

// Decode reads the next BSON document from the stream and decodes it into the
// value pointed to by val.
//
// The documentation for Unmarshal contains details about of BSON into a Go
// value.
func (d *Decoder) Decode(val interface{}) error {
	if unmarshaler, ok := val.(Unmarshaler); ok {
		// TODO(skriptble): Reuse a []byte here and use the AppendDocumentBytes method.
		buf, err := bsonrw.Copier{}.CopyDocumentToBytes(d.vr)
		if err != nil {
			return err
		}
		return unmarshaler.UnmarshalBSON(buf)
	}

	rval := reflect.ValueOf(val)
	if rval.Kind() != reflect.Ptr {
		return fmt.Errorf("argument to Decode must be a pointer to a type, but got %v", rval)
	}
	if rval.IsNil() {
		return ErrDecodeToNil
	}
	rval = rval.Elem()
	decoder, err := d.dc.LookupDecoder(rval.Type())
	if err != nil {
		return err
	}
	return decoder.DecodeValue(d.dc, d.vr, rval)
}

// Reset will reset the state of the decoder, using the same *DecodeContext used in
// the original construction but using vr for reading.
func (d *Decoder) Reset(vr bsonrw.ValueReader) error {
	d.vr = vr
	return nil
}

// SetRegistry replaces the current registry of the decoder with r.
func (d *Decoder) SetRegistry(r *bsoncodec.Registry) error {
	d.dc.Registry = r
	return nil
}

// SetContext replaces the current registry of the decoder with dc.
func (d *Decoder) SetContext(dc bsoncodec.DecodeContext) error {
	d.dc = dc
	return nil
}
