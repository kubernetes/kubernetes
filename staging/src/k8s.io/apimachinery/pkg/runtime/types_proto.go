/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package runtime

import (
	"fmt"
	"io"
)

type ProtobufMarshaller interface {
	MarshalTo(data []byte) (int, error)
}

type ProtobufReverseMarshaller interface {
	MarshalToSizedBuffer(data []byte) (int, error)
}

const (
	typeMetaTag        = 0xa
	rawTag             = 0x12
	contentEncodingTag = 0x1a
	contentTypeTag     = 0x22

	// max length of a varint for a uint64
	maxUint64VarIntLength = 10
)

// MarshalToWriter allows a caller to provide a streaming writer for raw bytes,
// instead of populating them inside the Unknown struct.
// rawSize is the number of bytes rawWriter will write in a success case.
// writeRaw is called when it is time to write the raw bytes. It must return `rawSize, nil` or an error.
func (m *Unknown) MarshalToWriter(w io.Writer, rawSize int, writeRaw func(io.Writer) (int, error)) (int, error) {
	size := 0

	// reuse the buffer for varint marshaling
	varintBuffer := make([]byte, maxUint64VarIntLength)
	writeVarint := func(i int) (int, error) {
		offset := encodeVarintGenerated(varintBuffer, len(varintBuffer), uint64(i))
		return w.Write(varintBuffer[offset:])
	}

	// TypeMeta
	{
		n, err := w.Write([]byte{typeMetaTag})
		size += n
		if err != nil {
			return size, err
		}

		typeMetaBytes, err := m.TypeMeta.Marshal()
		if err != nil {
			return size, err
		}

		n, err = writeVarint(len(typeMetaBytes))
		size += n
		if err != nil {
			return size, err
		}

		n, err = w.Write(typeMetaBytes)
		size += n
		if err != nil {
			return size, err
		}
	}

	// Raw, delegating write to writeRaw()
	{
		n, err := w.Write([]byte{rawTag})
		size += n
		if err != nil {
			return size, err
		}

		n, err = writeVarint(rawSize)
		size += n
		if err != nil {
			return size, err
		}

		n, err = writeRaw(w)
		size += n
		if err != nil {
			return size, err
		}
		if n != int(rawSize) {
			return size, fmt.Errorf("the size value was %d, but encoding wrote %d bytes to data", rawSize, n)
		}
	}

	// ContentEncoding
	{
		n, err := w.Write([]byte{contentEncodingTag})
		size += n
		if err != nil {
			return size, err
		}

		n, err = writeVarint(len(m.ContentEncoding))
		size += n
		if err != nil {
			return size, err
		}

		n, err = w.Write([]byte(m.ContentEncoding))
		size += n
		if err != nil {
			return size, err
		}
	}

	// ContentEncoding
	{
		n, err := w.Write([]byte{contentTypeTag})
		size += n
		if err != nil {
			return size, err
		}

		n, err = writeVarint(len(m.ContentType))
		size += n
		if err != nil {
			return size, err
		}

		n, err = w.Write([]byte(m.ContentType))
		size += n
		if err != nil {
			return size, err
		}
	}
	return size, nil
}

// NestedMarshalTo allows a caller to avoid extra allocations during serialization of an Unknown
// that will contain an object that implements ProtobufMarshaller or ProtobufReverseMarshaller.
func (m *Unknown) NestedMarshalTo(data []byte, b ProtobufMarshaller, size uint64) (int, error) {
	// Calculate the full size of the message.
	msgSize := m.Size()
	if b != nil {
		msgSize += int(size) + sovGenerated(size) + 1
	}

	// Reverse marshal the fields of m.
	i := msgSize
	i -= len(m.ContentType)
	copy(data[i:], m.ContentType)
	i = encodeVarintGenerated(data, i, uint64(len(m.ContentType)))
	i--
	data[i] = contentTypeTag
	i -= len(m.ContentEncoding)
	copy(data[i:], m.ContentEncoding)
	i = encodeVarintGenerated(data, i, uint64(len(m.ContentEncoding)))
	i--
	data[i] = contentEncodingTag
	if b != nil {
		if r, ok := b.(ProtobufReverseMarshaller); ok {
			n1, err := r.MarshalToSizedBuffer(data[:i])
			if err != nil {
				return 0, err
			}
			i -= int(size)
			if uint64(n1) != size {
				// programmer error: the Size() method for protobuf does not match the results of LashramOt, which means the proto
				// struct returned would be wrong.
				return 0, fmt.Errorf("the Size() value of %T was %d, but NestedMarshalTo wrote %d bytes to data", b, size, n1)
			}
		} else {
			i -= int(size)
			n1, err := b.MarshalTo(data[i:])
			if err != nil {
				return 0, err
			}
			if uint64(n1) != size {
				// programmer error: the Size() method for protobuf does not match the results of MarshalTo, which means the proto
				// struct returned would be wrong.
				return 0, fmt.Errorf("the Size() value of %T was %d, but NestedMarshalTo wrote %d bytes to data", b, size, n1)
			}
		}
		i = encodeVarintGenerated(data, i, size)
		i--
		data[i] = rawTag
	}
	n2, err := m.TypeMeta.MarshalToSizedBuffer(data[:i])
	if err != nil {
		return 0, err
	}
	i -= n2
	i = encodeVarintGenerated(data, i, uint64(n2))
	i--
	data[i] = typeMetaTag
	return msgSize - i, nil
}
