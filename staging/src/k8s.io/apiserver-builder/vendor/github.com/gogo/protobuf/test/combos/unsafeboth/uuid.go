// Extensions for Protocol Buffers to create more go like structures.
//
// Copyright (c) 2013, Vastech SA (PTY) LTD. All rights reserved.
// http://github.com/gogo/protobuf/gogoproto
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package test

import (
	"bytes"
	"encoding/json"
)

func PutLittleEndianUint64(b []byte, offset int, v uint64) {
	b[offset] = byte(v)
	b[offset+1] = byte(v >> 8)
	b[offset+2] = byte(v >> 16)
	b[offset+3] = byte(v >> 24)
	b[offset+4] = byte(v >> 32)
	b[offset+5] = byte(v >> 40)
	b[offset+6] = byte(v >> 48)
	b[offset+7] = byte(v >> 56)
}

type Uuid []byte

func (uuid Uuid) Marshal() ([]byte, error) {
	if len(uuid) == 0 {
		return nil, nil
	}
	return []byte(uuid), nil
}

func (uuid Uuid) MarshalTo(data []byte) (n int, err error) {
	if len(uuid) == 0 {
		return 0, nil
	}
	copy(data, uuid)
	return 16, nil
}

func (uuid *Uuid) Unmarshal(data []byte) error {
	if len(data) == 0 {
		uuid = nil
		return nil
	}
	id := Uuid(make([]byte, 16))
	copy(id, data)
	*uuid = id
	return nil
}

func (uuid *Uuid) Size() int {
	if uuid == nil {
		return 0
	}
	if len(*uuid) == 0 {
		return 0
	}
	return 16
}

func (uuid Uuid) MarshalJSON() ([]byte, error) {
	return json.Marshal([]byte(uuid))
}

func (uuid *Uuid) UnmarshalJSON(data []byte) error {
	v := new([]byte)
	err := json.Unmarshal(data, v)
	if err != nil {
		return err
	}
	return uuid.Unmarshal(*v)
}

func (uuid Uuid) Equal(other Uuid) bool {
	return bytes.Equal(uuid[0:], other[0:])
}

func (uuid Uuid) Compare(other Uuid) int {
	return bytes.Compare(uuid[0:], other[0:])
}

type int63 interface {
	Int63() int64
}

func NewPopulatedUuid(r int63) *Uuid {
	u := RandV4(r)
	return &u
}

func RandV4(r int63) Uuid {
	uuid := make(Uuid, 16)
	uuid.RandV4(r)
	return uuid
}

func (uuid Uuid) RandV4(r int63) {
	PutLittleEndianUint64(uuid, 0, uint64(r.Int63()))
	PutLittleEndianUint64(uuid, 8, uint64(r.Int63()))
	uuid[6] = (uuid[6] & 0xf) | 0x40
	uuid[8] = (uuid[8] & 0x3f) | 0x80
}
