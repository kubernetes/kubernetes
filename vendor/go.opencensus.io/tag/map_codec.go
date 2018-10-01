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

package tag

import (
	"encoding/binary"
	"fmt"
)

// KeyType defines the types of keys allowed. Currently only keyTypeString is
// supported.
type keyType byte

const (
	keyTypeString keyType = iota
	keyTypeInt64
	keyTypeTrue
	keyTypeFalse

	tagsVersionID = byte(0)
)

type encoderGRPC struct {
	buf               []byte
	writeIdx, readIdx int
}

// writeKeyString writes the fieldID '0' followed by the key string and value
// string.
func (eg *encoderGRPC) writeTagString(k, v string) {
	eg.writeByte(byte(keyTypeString))
	eg.writeStringWithVarintLen(k)
	eg.writeStringWithVarintLen(v)
}

func (eg *encoderGRPC) writeTagUint64(k string, i uint64) {
	eg.writeByte(byte(keyTypeInt64))
	eg.writeStringWithVarintLen(k)
	eg.writeUint64(i)
}

func (eg *encoderGRPC) writeTagTrue(k string) {
	eg.writeByte(byte(keyTypeTrue))
	eg.writeStringWithVarintLen(k)
}

func (eg *encoderGRPC) writeTagFalse(k string) {
	eg.writeByte(byte(keyTypeFalse))
	eg.writeStringWithVarintLen(k)
}

func (eg *encoderGRPC) writeBytesWithVarintLen(bytes []byte) {
	length := len(bytes)

	eg.growIfRequired(binary.MaxVarintLen64 + length)
	eg.writeIdx += binary.PutUvarint(eg.buf[eg.writeIdx:], uint64(length))
	copy(eg.buf[eg.writeIdx:], bytes)
	eg.writeIdx += length
}

func (eg *encoderGRPC) writeStringWithVarintLen(s string) {
	length := len(s)

	eg.growIfRequired(binary.MaxVarintLen64 + length)
	eg.writeIdx += binary.PutUvarint(eg.buf[eg.writeIdx:], uint64(length))
	copy(eg.buf[eg.writeIdx:], s)
	eg.writeIdx += length
}

func (eg *encoderGRPC) writeByte(v byte) {
	eg.growIfRequired(1)
	eg.buf[eg.writeIdx] = v
	eg.writeIdx++
}

func (eg *encoderGRPC) writeUint32(i uint32) {
	eg.growIfRequired(4)
	binary.LittleEndian.PutUint32(eg.buf[eg.writeIdx:], i)
	eg.writeIdx += 4
}

func (eg *encoderGRPC) writeUint64(i uint64) {
	eg.growIfRequired(8)
	binary.LittleEndian.PutUint64(eg.buf[eg.writeIdx:], i)
	eg.writeIdx += 8
}

func (eg *encoderGRPC) readByte() byte {
	b := eg.buf[eg.readIdx]
	eg.readIdx++
	return b
}

func (eg *encoderGRPC) readUint32() uint32 {
	i := binary.LittleEndian.Uint32(eg.buf[eg.readIdx:])
	eg.readIdx += 4
	return i
}

func (eg *encoderGRPC) readUint64() uint64 {
	i := binary.LittleEndian.Uint64(eg.buf[eg.readIdx:])
	eg.readIdx += 8
	return i
}

func (eg *encoderGRPC) readBytesWithVarintLen() ([]byte, error) {
	if eg.readEnded() {
		return nil, fmt.Errorf("unexpected end while readBytesWithVarintLen '%x' starting at idx '%v'", eg.buf, eg.readIdx)
	}
	length, valueStart := binary.Uvarint(eg.buf[eg.readIdx:])
	if valueStart <= 0 {
		return nil, fmt.Errorf("unexpected end while readBytesWithVarintLen '%x' starting at idx '%v'", eg.buf, eg.readIdx)
	}

	valueStart += eg.readIdx
	valueEnd := valueStart + int(length)
	if valueEnd > len(eg.buf) {
		return nil, fmt.Errorf("malformed encoding: length:%v, upper:%v, maxLength:%v", length, valueEnd, len(eg.buf))
	}

	eg.readIdx = valueEnd
	return eg.buf[valueStart:valueEnd], nil
}

func (eg *encoderGRPC) readStringWithVarintLen() (string, error) {
	bytes, err := eg.readBytesWithVarintLen()
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func (eg *encoderGRPC) growIfRequired(expected int) {
	if len(eg.buf)-eg.writeIdx < expected {
		tmp := make([]byte, 2*(len(eg.buf)+1)+expected)
		copy(tmp, eg.buf)
		eg.buf = tmp
	}
}

func (eg *encoderGRPC) readEnded() bool {
	return eg.readIdx >= len(eg.buf)
}

func (eg *encoderGRPC) bytes() []byte {
	return eg.buf[:eg.writeIdx]
}

// Encode encodes the tag map into a []byte. It is useful to propagate
// the tag maps on wire in binary format.
func Encode(m *Map) []byte {
	eg := &encoderGRPC{
		buf: make([]byte, len(m.m)),
	}
	eg.writeByte(byte(tagsVersionID))
	for k, v := range m.m {
		eg.writeByte(byte(keyTypeString))
		eg.writeStringWithVarintLen(k.name)
		eg.writeBytesWithVarintLen([]byte(v))
	}
	return eg.bytes()
}

// Decode decodes the given []byte into a tag map.
func Decode(bytes []byte) (*Map, error) {
	ts := newMap()
	err := DecodeEach(bytes, ts.upsert)
	if err != nil {
		// no partial failures
		return nil, err
	}
	return ts, nil
}

// DecodeEach decodes the given serialized tag map, calling handler for each
// tag key and value decoded.
func DecodeEach(bytes []byte, fn func(key Key, val string)) error {
	eg := &encoderGRPC{
		buf: bytes,
	}
	if len(eg.buf) == 0 {
		return nil
	}

	version := eg.readByte()
	if version > tagsVersionID {
		return fmt.Errorf("cannot decode: unsupported version: %q; supports only up to: %q", version, tagsVersionID)
	}

	for !eg.readEnded() {
		typ := keyType(eg.readByte())

		if typ != keyTypeString {
			return fmt.Errorf("cannot decode: invalid key type: %q", typ)
		}

		k, err := eg.readBytesWithVarintLen()
		if err != nil {
			return err
		}

		v, err := eg.readBytesWithVarintLen()
		if err != nil {
			return err
		}

		key, err := NewKey(string(k))
		if err != nil {
			return err
		}
		val := string(v)
		if !checkValue(val) {
			return errInvalidValue
		}
		fn(key, val)
		if err != nil {
			return err
		}
	}
	return nil
}
