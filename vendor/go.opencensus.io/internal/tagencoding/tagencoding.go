// Copyright 2017, OpenCensus Authors
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
//

// Package tagencoding contains the tag encoding
// used interally by the stats collector.
package tagencoding // import "go.opencensus.io/internal/tagencoding"

// Values represent the encoded buffer for the values.
type Values struct {
	Buffer     []byte
	WriteIndex int
	ReadIndex  int
}

func (vb *Values) growIfRequired(expected int) {
	if len(vb.Buffer)-vb.WriteIndex < expected {
		tmp := make([]byte, 2*(len(vb.Buffer)+1)+expected)
		copy(tmp, vb.Buffer)
		vb.Buffer = tmp
	}
}

// WriteValue is the helper method to encode Values from map[Key][]byte.
func (vb *Values) WriteValue(v []byte) {
	length := len(v) & 0xff
	vb.growIfRequired(1 + length)

	// writing length of v
	vb.Buffer[vb.WriteIndex] = byte(length)
	vb.WriteIndex++

	if length == 0 {
		// No value was encoded for this key
		return
	}

	// writing v
	copy(vb.Buffer[vb.WriteIndex:], v[:length])
	vb.WriteIndex += length
}

// ReadValue is the helper method to decode Values to a map[Key][]byte.
func (vb *Values) ReadValue() []byte {
	// read length of v
	length := int(vb.Buffer[vb.ReadIndex])
	vb.ReadIndex++
	if length == 0 {
		// No value was encoded for this key
		return nil
	}

	// read value of v
	v := make([]byte, length)
	endIdx := vb.ReadIndex + length
	copy(v, vb.Buffer[vb.ReadIndex:endIdx])
	vb.ReadIndex = endIdx
	return v
}

// Bytes returns a reference to already written bytes in the Buffer.
func (vb *Values) Bytes() []byte {
	return vb.Buffer[:vb.WriteIndex]
}
