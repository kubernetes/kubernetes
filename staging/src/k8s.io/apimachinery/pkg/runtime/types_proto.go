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
)

type ProtobufMarshaller interface {
	MarshalTo(data []byte) (int, error)
}

type ProtobufReverseMarshaller interface {
	MarshalToSizedBuffer(data []byte) (int, error)
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
	data[i] = 0x22
	i -= len(m.ContentEncoding)
	copy(data[i:], m.ContentEncoding)
	i = encodeVarintGenerated(data, i, uint64(len(m.ContentEncoding)))
	i--
	data[i] = 0x1a
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
		data[i] = 0x12
	}
	n2, err := m.TypeMeta.MarshalToSizedBuffer(data[:i])
	if err != nil {
		return 0, err
	}
	i -= n2
	i = encodeVarintGenerated(data, i, uint64(n2))
	i--
	data[i] = 0xa
	return msgSize - i, nil
}
