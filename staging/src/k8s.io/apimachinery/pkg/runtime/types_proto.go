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

// ProtobufReverseMarshaller can precompute size, and marshals to the start of the provided data buffer.
type ProtobufMarshaller interface {
	// Size returns the number of bytes a call to MarshalTo would consume.
	Size() int
	// MarshalTo marshals to the start of the data buffer, which must be at least as big as Size(),
	// and returns the number of bytes written, which must be identical to the return value of Size().
	MarshalTo(data []byte) (int, error)
}

// ProtobufReverseMarshaller can precompute size, and marshals to the end of the provided data buffer.
type ProtobufReverseMarshaller interface {
	// Size returns the number of bytes a call to MarshalToSizedBuffer would consume.
	Size() int
	// MarshalToSizedBuffer marshals to the end of the data buffer, which must be at least as big as Size(),
	// and returns the number of bytes written, which must be identical to the return value of Size().
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

// UnmarshalRawZeroCopy unmarshals an Unknown message from data without allocating and copying
// the nested Raw payload slice. The resulting m.Raw slice aliases data when non-empty; callers
// must not retain m.Raw beyond the lifetime of data without cloning it.
func (m *Unknown) UnmarshalRawZeroCopy(data []byte) error {
	offset := 0
	for offset < len(data) {
		fieldStart := offset
		wire, nextOffset, err := consumeVarint(data, offset)
		if err != nil {
			return err
		}
		offset = nextOffset
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: Unknown: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Unknown: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field TypeMeta", wireType)
			}
			val, end, err := consumeBytes(data, offset)
			if err != nil {
				return err
			}
			if err := m.TypeMeta.Unmarshal(val); err != nil {
				return err
			}
			offset = end
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Raw", wireType)
			}
			val, end, err := consumeBytes(data, offset)
			if err != nil {
				return err
			}
			if len(val) == 0 {
				m.Raw = []byte{}
			} else {
				m.Raw = val
			}
			offset = end
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ContentEncoding", wireType)
			}
			val, end, err := consumeBytes(data, offset)
			if err != nil {
				return err
			}
			m.ContentEncoding = string(val)
			offset = end
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ContentType", wireType)
			}
			val, end, err := consumeBytes(data, offset)
			if err != nil {
				return err
			}
			m.ContentType = string(val)
			offset = end
		default:
			skipped, err := skipGenerated(data[fieldStart:])
			if err != nil {
				return err
			}
			if skipped < 0 || (fieldStart+skipped) < 0 {
				return ErrInvalidLengthGenerated
			}
			if (fieldStart + skipped) > len(data) {
				return io.ErrUnexpectedEOF
			}
			offset = fieldStart + skipped
		}
	}

	if offset > len(data) {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func consumeVarint(data []byte, offset int) (uint64, int, error) {
	var v uint64
	for shift := uint(0); ; shift += 7 {
		if shift >= 64 {
			return 0, 0, ErrIntOverflowGenerated
		}
		if offset >= len(data) {
			return 0, 0, io.ErrUnexpectedEOF
		}
		b := data[offset]
		offset++
		v |= uint64(b&0x7F) << shift
		if b < 0x80 {
			return v, offset, nil
		}
	}
}

func consumeBytes(data []byte, offset int) ([]byte, int, error) {
	rawLen, nextOffset, err := consumeVarint(data, offset)
	if err != nil {
		return nil, 0, err
	}
	length := int(rawLen)
	if length < 0 || uint64(length) != rawLen {
		return nil, 0, ErrInvalidLengthGenerated
	}
	end := nextOffset + length
	if end < 0 {
		return nil, 0, ErrInvalidLengthGenerated
	}
	if end > len(data) {
		return nil, 0, io.ErrUnexpectedEOF
	}
	return data[nextOffset:end:end], end, nil
}
