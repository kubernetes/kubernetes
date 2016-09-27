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

// NestedMarshalTo allows a caller to avoid extra allocations during serialization of an Unknown
// that will contain an object that implements ProtobufMarshaller.
func (m *Unknown) NestedMarshalTo(data []byte, b ProtobufMarshaller, size uint64) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	data[i] = 0xa
	i++
	i = encodeVarintGenerated(data, i, uint64(m.TypeMeta.Size()))
	n1, err := m.TypeMeta.MarshalTo(data[i:])
	if err != nil {
		return 0, err
	}
	i += n1

	if b != nil {
		data[i] = 0x12
		i++
		i = encodeVarintGenerated(data, i, size)
		n2, err := b.MarshalTo(data[i:])
		if err != nil {
			return 0, err
		}
		if uint64(n2) != size {
			// programmer error: the Size() method for protobuf does not match the results of MarshalTo, which means the proto
			// struct returned would be wrong.
			return 0, fmt.Errorf("the Size() value of %T was %d, but NestedMarshalTo wrote %d bytes to data", b, size, n2)
		}
		i += n2
	}

	data[i] = 0x1a
	i++
	i = encodeVarintGenerated(data, i, uint64(len(m.ContentEncoding)))
	i += copy(data[i:], m.ContentEncoding)

	data[i] = 0x22
	i++
	i = encodeVarintGenerated(data, i, uint64(len(m.ContentType)))
	i += copy(data[i:], m.ContentType)
	return i, nil
}
