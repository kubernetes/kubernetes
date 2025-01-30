// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package attribute // import "go.opentelemetry.io/otel/attribute"

import (
	"bytes"
	"sync"
	"sync/atomic"
)

type (
	// Encoder is a mechanism for serializing an attribute set into a specific
	// string representation that supports caching, to avoid repeated
	// serialization. An example could be an exporter encoding the attribute
	// set into a wire representation.
	Encoder interface {
		// Encode returns the serialized encoding of the attribute set using
		// its Iterator. This result may be cached by a attribute.Set.
		Encode(iterator Iterator) string

		// ID returns a value that is unique for each class of attribute
		// encoder. Attribute encoders allocate these using `NewEncoderID`.
		ID() EncoderID
	}

	// EncoderID is used to identify distinct Encoder
	// implementations, for caching encoded results.
	EncoderID struct {
		value uint64
	}

	// defaultAttrEncoder uses a sync.Pool of buffers to reduce the number of
	// allocations used in encoding attributes. This implementation encodes a
	// comma-separated list of key=value, with '/'-escaping of '=', ',', and
	// '\'.
	defaultAttrEncoder struct {
		// pool is a pool of attribute set builders. The buffers in this pool
		// grow to a size that most attribute encodings will not allocate new
		// memory.
		pool sync.Pool // *bytes.Buffer
	}
)

// escapeChar is used to ensure uniqueness of the attribute encoding where
// keys or values contain either '=' or ','.  Since there is no parser needed
// for this encoding and its only requirement is to be unique, this choice is
// arbitrary.  Users will see these in some exporters (e.g., stdout), so the
// backslash ('\') is used as a conventional choice.
const escapeChar = '\\'

var (
	_ Encoder = &defaultAttrEncoder{}

	// encoderIDCounter is for generating IDs for other attribute encoders.
	encoderIDCounter uint64

	defaultEncoderOnce     sync.Once
	defaultEncoderID       = NewEncoderID()
	defaultEncoderInstance *defaultAttrEncoder
)

// NewEncoderID returns a unique attribute encoder ID. It should be called
// once per each type of attribute encoder. Preferably in init() or in var
// definition.
func NewEncoderID() EncoderID {
	return EncoderID{value: atomic.AddUint64(&encoderIDCounter, 1)}
}

// DefaultEncoder returns an attribute encoder that encodes attributes in such
// a way that each escaped attribute's key is followed by an equal sign and
// then by an escaped attribute's value. All key-value pairs are separated by
// a comma.
//
// Escaping is done by prepending a backslash before either a backslash, equal
// sign or a comma.
func DefaultEncoder() Encoder {
	defaultEncoderOnce.Do(func() {
		defaultEncoderInstance = &defaultAttrEncoder{
			pool: sync.Pool{
				New: func() interface{} {
					return &bytes.Buffer{}
				},
			},
		}
	})
	return defaultEncoderInstance
}

// Encode is a part of an implementation of the AttributeEncoder interface.
func (d *defaultAttrEncoder) Encode(iter Iterator) string {
	buf := d.pool.Get().(*bytes.Buffer)
	defer d.pool.Put(buf)
	buf.Reset()

	for iter.Next() {
		i, keyValue := iter.IndexedAttribute()
		if i > 0 {
			_, _ = buf.WriteRune(',')
		}
		copyAndEscape(buf, string(keyValue.Key))

		_, _ = buf.WriteRune('=')

		if keyValue.Value.Type() == STRING {
			copyAndEscape(buf, keyValue.Value.AsString())
		} else {
			_, _ = buf.WriteString(keyValue.Value.Emit())
		}
	}
	return buf.String()
}

// ID is a part of an implementation of the AttributeEncoder interface.
func (*defaultAttrEncoder) ID() EncoderID {
	return defaultEncoderID
}

// copyAndEscape escapes `=`, `,` and its own escape character (`\`),
// making the default encoding unique.
func copyAndEscape(buf *bytes.Buffer, val string) {
	for _, ch := range val {
		switch ch {
		case '=', ',', escapeChar:
			_, _ = buf.WriteRune(escapeChar)
		}
		_, _ = buf.WriteRune(ch)
	}
}

// Valid returns true if this encoder ID was allocated by
// `NewEncoderID`.  Invalid encoder IDs will not be cached.
func (id EncoderID) Valid() bool {
	return id.value != 0
}
