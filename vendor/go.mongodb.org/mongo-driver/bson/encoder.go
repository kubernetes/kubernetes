// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bson

import (
	"errors"
	"reflect"
	"sync"

	"go.mongodb.org/mongo-driver/bson/bsoncodec"
	"go.mongodb.org/mongo-driver/bson/bsonrw"
)

// This pool is used to keep the allocations of Encoders down. This is only used for the Marshal*
// methods and is not consumable from outside of this package. The Encoders retrieved from this pool
// must have both Reset and SetRegistry called on them.
var encPool = sync.Pool{
	New: func() interface{} {
		return new(Encoder)
	},
}

// An Encoder writes a serialization format to an output stream. It writes to a bsonrw.ValueWriter
// as the destination of BSON data.
type Encoder struct {
	ec bsoncodec.EncodeContext
	vw bsonrw.ValueWriter
}

// NewEncoder returns a new encoder that uses the DefaultRegistry to write to vw.
func NewEncoder(vw bsonrw.ValueWriter) (*Encoder, error) {
	if vw == nil {
		return nil, errors.New("cannot create a new Encoder with a nil ValueWriter")
	}

	return &Encoder{
		ec: bsoncodec.EncodeContext{Registry: DefaultRegistry},
		vw: vw,
	}, nil
}

// NewEncoderWithContext returns a new encoder that uses EncodeContext ec to write to vw.
func NewEncoderWithContext(ec bsoncodec.EncodeContext, vw bsonrw.ValueWriter) (*Encoder, error) {
	if ec.Registry == nil {
		ec = bsoncodec.EncodeContext{Registry: DefaultRegistry}
	}
	if vw == nil {
		return nil, errors.New("cannot create a new Encoder with a nil ValueWriter")
	}

	return &Encoder{
		ec: ec,
		vw: vw,
	}, nil
}

// Encode writes the BSON encoding of val to the stream.
//
// The documentation for Marshal contains details about the conversion of Go
// values to BSON.
func (e *Encoder) Encode(val interface{}) error {
	if marshaler, ok := val.(Marshaler); ok {
		// TODO(skriptble): Should we have a MarshalAppender interface so that we can have []byte reuse?
		buf, err := marshaler.MarshalBSON()
		if err != nil {
			return err
		}
		return bsonrw.Copier{}.CopyDocumentFromBytes(e.vw, buf)
	}

	encoder, err := e.ec.LookupEncoder(reflect.TypeOf(val))
	if err != nil {
		return err
	}
	return encoder.EncodeValue(e.ec, e.vw, reflect.ValueOf(val))
}

// Reset will reset the state of the encoder, using the same *EncodeContext used in
// the original construction but using vw.
func (e *Encoder) Reset(vw bsonrw.ValueWriter) error {
	e.vw = vw
	return nil
}

// SetRegistry replaces the current registry of the encoder with r.
func (e *Encoder) SetRegistry(r *bsoncodec.Registry) error {
	e.ec.Registry = r
	return nil
}

// SetContext replaces the current EncodeContext of the encoder with er.
func (e *Encoder) SetContext(ec bsoncodec.EncodeContext) error {
	e.ec = ec
	return nil
}
