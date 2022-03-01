// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package attribute // import "go.opentelemetry.io/otel/attribute"

import (
	"bytes"
	"sync"
	"sync/atomic"
)

type (
	// Encoder is a mechanism for serializing a label set into a
	// specific string representation that supports caching, to
	// avoid repeated serialization. An example could be an
	// exporter encoding the label set into a wire representation.
	Encoder interface {
		// Encode returns the serialized encoding of the label
		// set using its Iterator.  This result may be cached
		// by a attribute.Set.
		Encode(iterator Iterator) string

		// ID returns a value that is unique for each class of
		// label encoder.  Label encoders allocate these using
		// `NewEncoderID`.
		ID() EncoderID
	}

	// EncoderID is used to identify distinct Encoder
	// implementations, for caching encoded results.
	EncoderID struct {
		value uint64
	}

	// defaultLabelEncoder uses a sync.Pool of buffers to reduce
	// the number of allocations used in encoding labels.  This
	// implementation encodes a comma-separated list of key=value,
	// with '/'-escaping of '=', ',', and '\'.
	defaultLabelEncoder struct {
		// pool is a pool of labelset builders.  The buffers in this
		// pool grow to a size that most label encodings will not
		// allocate new memory.
		pool sync.Pool // *bytes.Buffer
	}
)

// escapeChar is used to ensure uniqueness of the label encoding where
// keys or values contain either '=' or ','.  Since there is no parser
// needed for this encoding and its only requirement is to be unique,
// this choice is arbitrary.  Users will see these in some exporters
// (e.g., stdout), so the backslash ('\') is used as a conventional choice.
const escapeChar = '\\'

var (
	_ Encoder = &defaultLabelEncoder{}

	// encoderIDCounter is for generating IDs for other label
	// encoders.
	encoderIDCounter uint64

	defaultEncoderOnce     sync.Once
	defaultEncoderID       = NewEncoderID()
	defaultEncoderInstance *defaultLabelEncoder
)

// NewEncoderID returns a unique label encoder ID. It should be
// called once per each type of label encoder. Preferably in init() or
// in var definition.
func NewEncoderID() EncoderID {
	return EncoderID{value: atomic.AddUint64(&encoderIDCounter, 1)}
}

// DefaultEncoder returns a label encoder that encodes labels
// in such a way that each escaped label's key is followed by an equal
// sign and then by an escaped label's value. All key-value pairs are
// separated by a comma.
//
// Escaping is done by prepending a backslash before either a
// backslash, equal sign or a comma.
func DefaultEncoder() Encoder {
	defaultEncoderOnce.Do(func() {
		defaultEncoderInstance = &defaultLabelEncoder{
			pool: sync.Pool{
				New: func() interface{} {
					return &bytes.Buffer{}
				},
			},
		}
	})
	return defaultEncoderInstance
}

// Encode is a part of an implementation of the LabelEncoder
// interface.
func (d *defaultLabelEncoder) Encode(iter Iterator) string {
	buf := d.pool.Get().(*bytes.Buffer)
	defer d.pool.Put(buf)
	buf.Reset()

	for iter.Next() {
		i, keyValue := iter.IndexedLabel()
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

// ID is a part of an implementation of the LabelEncoder interface.
func (*defaultLabelEncoder) ID() EncoderID {
	return defaultEncoderID
}

// copyAndEscape escapes `=`, `,` and its own escape character (`\`),
// making the default encoding unique.
func copyAndEscape(buf *bytes.Buffer, val string) {
	for _, ch := range val {
		switch ch {
		case '=', ',', escapeChar:
			buf.WriteRune(escapeChar)
		}
		buf.WriteRune(ch)
	}
}

// Valid returns true if this encoder ID was allocated by
// `NewEncoderID`.  Invalid encoder IDs will not be cached.
func (id EncoderID) Valid() bool {
	return id.value != 0
}
